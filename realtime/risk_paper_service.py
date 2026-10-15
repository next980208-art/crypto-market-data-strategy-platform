"""Hard risk gate and paper-order audit service. It never calls an exchange API."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaTimeoutError

from realtime.contracts import PAPER_EVENT_SCHEMA_VERSION, TOPIC_PAPER_EXECUTIONS, TOPIC_RISK_AUDIT, TOPIC_SIGNALS, decode_json, json_bytes, utc_now_iso
from realtime.idempotency import IdempotencyStore, offsets_after_records
from market_metadata import canonical_event_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply hard risk limits to paper-only trading signals.")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--consumer-group", default="realtime-risk-paper-v1")
    parser.add_argument("--account-balance-usdt", type=float, default=10_000.0)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.02)
    parser.add_argument("--leverage", type=float, default=10.0)
    parser.add_argument("--max-margin-pct", type=float, default=0.35)
    parser.add_argument("--audit-path", default="realtime_runtime/risk_audit.jsonl")
    parser.add_argument("--metrics-path", default="realtime_runtime/risk_metrics.json")
    parser.add_argument("--idempotency-db", default="realtime_runtime/risk_idempotency.sqlite3")
    parser.add_argument("--transactional-id", default="realtime-risk-paper-v1")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.account_balance_usdt <= 0 or args.leverage <= 0:
        raise ValueError("Account balance and leverage must be positive.")
    if not 0 < args.risk_per_trade_pct <= 0.02:
        raise ValueError("Risk policy violation: risk per trade must be greater than 0 and at most 2%.")
    if not 0 < args.max_margin_pct <= 1:
        raise ValueError("--max-margin-pct must be in (0, 1].")


def create_consumer(args: argparse.Namespace) -> KafkaConsumer:
    while True:
        try:
            return KafkaConsumer(
                TOPIC_SIGNALS,
                bootstrap_servers=args.bootstrap_servers,
                group_id=args.consumer_group,
                auto_offset_reset="latest",
                enable_auto_commit=False,
                isolation_level="read_committed",
            )
        except KafkaTimeoutError:
            print("Kafka not ready for risk paper service; retrying in 5 seconds.")
            time.sleep(5)


def audit_event(signal: dict[str, Any], decision: str, reason: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    metadata = canonical_event_metadata(
        symbol=str(signal.get("symbol", "")),
        market=str(signal.get("market", "usdm")),
        timeframe=str(signal.get("timeframe", "1m")),
        event_time_ms=int(signal.get("event_time_ms", 0)),
    )
    return {
        "schema_version": PAPER_EVENT_SCHEMA_VERSION,
        "audit_id": f"{signal.get('signal_id')}:risk_audit",
        "event_type": "risk_audit",
        "created_at_utc": utc_now_iso(),
        "signal_id": signal.get("signal_id"),
        "decision": decision,
        "reason": reason,
        "details": details or {},
        "paper_only": True,
        "exchange_order_api_called": False,
        **metadata,
    }


def evaluate(signal: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if signal.get("action") not in {"long", "short"}:
        return audit_event(signal, "rejected", str(signal.get("reason", "no_trade"))), None
    entry_price = float(signal.get("entry_price") or 0)
    stop_distance_pct = float(signal.get("stop_distance_pct") or 0)
    if entry_price <= 0 or stop_distance_pct <= 0:
        return audit_event(signal, "rejected", "missing_entry_or_stop"), None
    risk_budget = args.account_balance_usdt * args.risk_per_trade_pct
    notional = risk_budget / stop_distance_pct
    margin = notional / args.leverage
    if margin / args.account_balance_usdt > args.max_margin_pct:
        return audit_event(signal, "rejected", "max_margin_pct_exceeded"), None
    quantity = notional / entry_price
    side = str(signal["action"])
    stop_price = entry_price * (1 - stop_distance_pct if side == "long" else 1 + stop_distance_pct)
    metadata = canonical_event_metadata(
        symbol=str(signal["symbol"]),
        market=str(signal.get("market", "usdm")),
        timeframe=str(signal.get("timeframe", "1m")),
        event_time_ms=int(signal.get("event_time_ms", 0)),
    )
    paper_order = {
        "schema_version": PAPER_EVENT_SCHEMA_VERSION,
        "paper_order_id": f"{signal.get('signal_id')}:paper_order_plan",
        "event_type": "paper_order_plan",
        "created_at_utc": utc_now_iso(),
        "signal_id": signal.get("signal_id"),
        "side": side,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "quantity": quantity,
        "notional_usdt": notional,
        "margin_usdt": margin,
        "risk_budget_usdt": risk_budget,
        "status": "planned_not_sent_to_exchange",
        "paper_only": True,
        "exchange_order_api_called": False,
        **metadata,
    }
    return audit_event(signal, "accepted_for_paper", "risk_limits_passed", paper_order), paper_order


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_metrics(path: Path, counters: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "updated_at_utc": utc_now_iso(),
                "counters": dict(counters),
                "paper_only": True,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def create_transactional_producer(args: argparse.Namespace) -> KafkaProducer:
    producer = KafkaProducer(
        bootstrap_servers=args.bootstrap_servers,
        acks="all",
        retries=5,
        enable_idempotence=True,
        transactional_id=args.transactional_id,
    )
    producer.init_transactions()
    return producer


def main() -> None:
    args = parse_args()
    validate_args(args)
    consumer = create_consumer(args)
    producer = create_transactional_producer(args)
    store = IdempotencyStore(args.idempotency_db)
    counters: Counter[str] = Counter()
    try:
        while True:
            records = consumer.poll(timeout_ms=1_000, max_records=500)
            if not records:
                write_metrics(Path(args.metrics_path), counters)
                continue
            committed_ids: list[str] = []
            committed_audits: list[dict[str, Any]] = []
            committed_orders = 0
            batch_ids: set[str] = set()
            batch_duplicates = 0
            batch_invalid = 0
            producer.begin_transaction()
            try:
                for partition_records in records.values():
                    for record in partition_records:
                        signal = decode_json(record.value)
                        signal_id = str(signal.get("signal_id", ""))
                        if not signal_id:
                            batch_invalid += 1
                            continue
                        if signal_id in batch_ids or store.contains("signal", signal_id):
                            batch_duplicates += 1
                            continue
                        audit, paper_order = evaluate(signal, args)
                        producer.send(
                            TOPIC_RISK_AUDIT,
                            key=str(audit.get("symbol", "account")).encode("utf-8"),
                            value=json_bytes(audit),
                        )
                        if paper_order is not None:
                            producer.send(
                                TOPIC_PAPER_EXECUTIONS,
                                key=paper_order["symbol"].encode("utf-8"),
                                value=json_bytes(paper_order),
                            )
                            committed_orders += 1
                        batch_ids.add(signal_id)
                        committed_ids.append(signal_id)
                        committed_audits.append(audit)
                producer.send_offsets_to_transaction(
                    offsets_after_records(records),
                    consumer.group_metadata(),
                )
                producer.commit_transaction()
            except Exception:
                producer.abort_transaction()
                raise
            store.mark_many("signal", committed_ids)
            for audit in committed_audits:
                append_jsonl(Path(args.audit_path), audit)
                counters[audit["decision"]] += 1
            counters["signals_committed"] += len(committed_ids)
            counters["paper_orders"] += committed_orders
            counters["duplicate_signals_skipped"] += batch_duplicates
            counters["invalid_signals_skipped"] += batch_invalid
            write_metrics(Path(args.metrics_path), counters)
    finally:
        producer.close(timeout=20)
        consumer.close()
        store.close()


if __name__ == "__main__":
    main()
