"""Durable business-ID deduplication and Kafka transaction offset helpers."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from kafka.structs import OffsetAndMetadata, TopicPartition


class IdempotencyStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_messages (
                scope TEXT NOT NULL,
                message_id TEXT NOT NULL,
                completed_at_utc TEXT NOT NULL,
                PRIMARY KEY (scope, message_id)
            )
            """
        )
        self.connection.commit()

    def contains(self, scope: str, message_id: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM processed_messages WHERE scope = ? AND message_id = ?",
            (scope, message_id),
        ).fetchone()
        return row is not None

    def mark_many(self, scope: str, message_ids: Iterable[str]) -> int:
        unique_ids = sorted({str(message_id) for message_id in message_ids if str(message_id)})
        if not unique_ids:
            return 0
        completed_at = datetime.now(timezone.utc).isoformat()
        before = self.connection.total_changes
        with self.connection:
            self.connection.executemany(
                """
                INSERT OR IGNORE INTO processed_messages(scope, message_id, completed_at_utc)
                VALUES (?, ?, ?)
                """,
                [(scope, message_id, completed_at) for message_id in unique_ids],
            )
        return self.connection.total_changes - before

    def count(self, scope: str) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) FROM processed_messages WHERE scope = ?",
            (scope,),
        ).fetchone()
        return int(row[0])

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "IdempotencyStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


def offsets_after_records(
    records: Mapping[TopicPartition, Sequence[object]],
) -> dict[TopicPartition, OffsetAndMetadata]:
    offsets: dict[TopicPartition, OffsetAndMetadata] = {}
    for partition, partition_records in records.items():
        if not partition_records:
            continue
        next_offset = max(int(getattr(record, "offset")) for record in partition_records) + 1
        offsets[partition] = OffsetAndMetadata(next_offset, "", -1)
    return offsets
