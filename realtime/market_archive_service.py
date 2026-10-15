"""Durable compressed Kafka archive for depth and bookTicker (single writer)."""

from __future__ import annotations

import argparse
import json
import signal
import sqlite3
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

from kafka import KafkaConsumer, TopicPartition
from kafka.structs import OffsetAndMetadata

from realtime.contracts import TOPIC_BOOK_TICKER, TOPIC_DEPTH


class MarketArchive:
    """Commit raw bytes and Kafka positions together before acknowledging Kafka."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS records (
                topic TEXT NOT NULL, partition_id INTEGER NOT NULL,
                offset_id INTEGER NOT NULL, timestamp_ms INTEGER,
                key_bytes BLOB, value_zlib BLOB NOT NULL,
                PRIMARY KEY (topic, partition_id, offset_id)
            );
            CREATE TABLE IF NOT EXISTS positions (
                topic TEXT NOT NULL, partition_id INTEGER NOT NULL,
                next_offset INTEGER NOT NULL,
                PRIMARY KEY (topic, partition_id)
            );
        """)

    def position(self, topic, partition):
        row = self.connection.execute(
            "SELECT next_offset FROM positions WHERE topic=? AND partition_id=?",
            (topic, partition),
        ).fetchone()
        return None if row is None else row[0]

    def save(self, records):
        inserted = 0
        with self.connection:
            for record in records:
                identity = (record.topic, record.partition, record.offset)
                previous = self.connection.execute(
                    "SELECT key_bytes, value_zlib FROM records "
                    "WHERE topic=? AND partition_id=? AND offset_id=?", identity,
                ).fetchone()
                if previous is not None:
                    if previous[0] != record.key or zlib.decompress(previous[1]) != record.value:
                        raise RuntimeError(f"Conflicting Kafka identity: {identity}")
                    continue
                expected = self.position(record.topic, record.partition)
                if expected is not None and record.offset != expected:
                    raise RuntimeError(f"Archive offset gap: expected={expected}, actual={identity}")
                self.connection.execute(
                    "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?)",
                    (*identity, record.timestamp, record.key, zlib.compress(record.value)),
                )
                self.connection.execute(
                    "INSERT INTO positions VALUES (?, ?, ?) ON CONFLICT(topic, partition_id) "
                    "DO UPDATE SET next_offset=excluded.next_offset",
                    (record.topic, record.partition, record.offset + 1),
                )
                inserted += 1
        return inserted

    def summary(self):
        groups = self.connection.execute(
            "SELECT topic, partition_id, COUNT(*), MIN(offset_id), MAX(offset_id), "
            "SUM(LENGTH(value_zlib)) FROM records GROUP BY topic, partition_id"
        ).fetchall()
        return {
            "stored_rows": sum(row[2] for row in groups),
            "partitions": [dict(zip(
                ("topic", "partition", "rows", "first_offset", "last_offset", "compressed_bytes"), row
            )) for row in groups],
            "storage": "sqlite_wal_zlib", "retention": "keep_until_explicit_review",
        }

    def close(self):
        self.connection.close()


def run(args):
    archive = MarketArchive(args.archive_file)
    consumer = None
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    status, error, received, saved = "completed", None, 0, 0

    def write_report(state):
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": state, "error": error, "received_this_run": received,
            "inserted_this_run": saved, "elapsed_seconds": round(time.monotonic() - started, 3),
            "archive_file": str(args.archive_file), **archive.summary(),
            "max_archive_bytes": args.max_archive_bytes,
            "scope": "Kafka records; source downtime is not recoverable from this archive",
        }
        destination = Path(args.report_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(destination)
        return report

    try:
        consumer = KafkaConsumer(
            bootstrap_servers=args.bootstrap_servers, group_id=args.consumer_group,
            enable_auto_commit=False, auto_offset_reset="none", max_poll_records=2000,
        )
        partitions = []
        for topic in (TOPIC_DEPTH, TOPIC_BOOK_TICKER):
            ids = consumer.partitions_for_topic(topic)
            if not ids:
                raise RuntimeError(f"Topic unavailable: {topic}; start the collector first")
            partitions.extend(TopicPartition(topic, item) for item in sorted(ids))
        consumer.assign(partitions)
        beginning, end = consumer.beginning_offsets(partitions), consumer.end_offsets(partitions)
        for tp in partitions:
            position = archive.position(tp.topic, tp.partition)
            if position is None:
                position = beginning[tp]
            if not beginning[tp] <= position <= end[tp]:
                raise RuntimeError(f"Kafka retention gap or topic reset: {tp}, offset={position}")
            consumer.seek(tp, position)
        last_report = time.monotonic()
        write_report("running")
        while not stopped and (args.duration_seconds == 0 or time.monotonic() - started < args.duration_seconds):
            stored_bytes = sum(path.stat().st_size for path in
                               (args.archive_file, Path(str(args.archive_file) + "-wal")) if path.exists())
            if stored_bytes >= args.max_archive_bytes:
                raise RuntimeError("Archive storage budget reached; review or back up data before resuming")
            batches = consumer.poll(timeout_ms=1000)
            for tp, records in batches.items():
                received += len(records)
                saved += archive.save(records)
                # A crash here replays already durable offsets without adding rows.
                consumer.commit({tp: OffsetAndMetadata(records[-1].offset + 1, "")})
            if time.monotonic() - last_report >= 30:
                write_report("running")
                last_report = time.monotonic()
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report = write_report(status)
        print(json.dumps(report, ensure_ascii=True), flush=True)
        archive.close()
        if consumer is not None:
            consumer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--consumer-group", default="market-archive-v1")
    parser.add_argument("--archive-file", type=Path, default=Path("realtime_runtime/market_archive.sqlite3"))
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--max-archive-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--report-json", default="runtime_reports/market_archive_latest.json")
    args = parser.parse_args()
    if args.duration_seconds < 0:
        parser.error("duration must be nonnegative")
    if args.max_archive_bytes <= 0:
        parser.error("archive budget must be positive")
    run(args)


if __name__ == "__main__":
    main()
