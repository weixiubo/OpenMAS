"""SQLite-backed run/event/cache store."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from openagora.core.interfaces import CacheStore
from openagora.types import ArchitectureSpec, CacheEntry, RunEvent, TaskSignature


class SQLiteStore(CacheStore):
    """Persist run metadata, events, metrics and cache versions."""

    CURRENT_SCHEMA_VERSION = 3

    def __init__(self, db_path: str) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate_to_latest()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def migrate_to_latest(self) -> None:
        """Apply schema migrations sequentially with transaction rollback on failure."""
        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            version = self.schema_version(conn)

            try:
                if version < 1:
                    self._migration_v1(conn)
                    self._set_schema_version(conn, 1)
                    version = 1
                if version < 2:
                    self._migration_v2(conn)
                    self._set_schema_version(conn, 2)
                    version = 2
                if version < 3:
                    self._migration_v3(conn)
                    self._set_schema_version(conn, 3)
            except Exception:
                conn.rollback()
                raise

    def schema_version(self, conn: sqlite3.Connection | None = None) -> int:
        """Return current schema version."""
        if conn is None:
            with self._conn() as new_conn:
                return self.schema_version(new_conn)
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()
        if row is None:
            return 0
        try:
            return int(str(row["value"]))
        except ValueError:
            return 0

    def _set_schema_version(self, conn: sqlite3.Connection, version: int) -> None:
        conn.execute(
            """
            INSERT INTO schema_meta (key, value) VALUES ('schema_version', ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (str(version),),
        )

    def _migration_v1(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                started_at TEXT NOT NULL,
                status TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                events_path TEXT NOT NULL,
                result_path TEXT,
                ended_at TEXT
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS cache_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                version_id INTEGER NOT NULL,
                signature_json TEXT NOT NULL,
                architecture_json TEXT NOT NULL,
                avg_perf REAL NOT NULL,
                sample_count INTEGER NOT NULL,
                ot_alignment REAL NOT NULL,
                needs_revalidation INTEGER NOT NULL DEFAULT 0,
                is_current INTEGER NOT NULL DEFAULT 0,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

    def _migration_v2(self, conn: sqlite3.Connection) -> None:
        if not self._has_column(conn, "events", "schema_version"):
            conn.execute("ALTER TABLE events ADD COLUMN schema_version TEXT NOT NULL DEFAULT 'v2'")

    def _migration_v3(self, conn: sqlite3.Connection) -> None:
        if not self._has_column(conn, "runs", "artifact_hashes_path"):
            conn.execute("ALTER TABLE runs ADD COLUMN artifact_hashes_path TEXT")

    def _has_column(self, conn: sqlite3.Connection, table: str, column: str) -> bool:
        columns = conn.execute(f"PRAGMA table_info({table})").fetchall()  # noqa: S608
        return any(str(row["name"]) == column for row in columns)

    # run/event/metrics persistence
    def start_run(
        self,
        *,
        run_id: str,
        task_id: str,
        started_at: str,
        config_hash: str,
        manifest_path: str,
        events_path: str,
        artifact_hashes_path: str,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, task_id, started_at, status, config_hash,
                    manifest_path, events_path, artifact_hashes_path
                )
                VALUES (?, ?, ?, 'running', ?, ?, ?, ?)
                """,
                (run_id, task_id, started_at, config_hash, manifest_path, events_path, artifact_hashes_path),
            )

    def finish_run(self, *, run_id: str, ended_at: str, status: str, result_path: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, ended_at = ?, result_path = ?
                WHERE run_id = ?
                """,
                (status, ended_at, result_path, run_id),
            )

    def save_event(self, run_id: str, event: RunEvent) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO events (run_id, ts, schema_version, event_type, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    event.ts,
                    event.schema_version,
                    event.event_type,
                    json.dumps(event.payload, ensure_ascii=False),
                ),
            )

    def save_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        with self._conn() as conn:
            rows = [(run_id, key, float(value)) for key, value in metrics.items()]
            conn.executemany("INSERT INTO metrics (run_id, key, value) VALUES (?, ?, ?)", rows)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            return None if row is None else dict(row)

    def list_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT ts, schema_version, event_type, payload_json FROM events WHERE run_id = ? ORDER BY id ASC",
                (run_id,),
            ).fetchall()
        return [
            {
                "ts": row["ts"],
                "schema_version": row["schema_version"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    # cache store implementation
    def lookup(self, signature: TaskSignature, threshold: float, type_weight: float) -> CacheEntry | None:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM cache_versions
                WHERE task_type = ? AND is_current = 1 AND needs_revalidation = 0
                ORDER BY version_id DESC
                """,
                (signature.primary_type,),
            ).fetchall()
        best: CacheEntry | None = None
        best_score = -1.0
        for row in rows:
            candidate = self._row_to_cache_entry(row)
            score = _signature_similarity(signature, candidate.signature, type_weight)
            if score >= threshold and score > best_score:
                best = candidate
                best_score = score
        return best

    def upsert(
        self,
        *,
        signature: TaskSignature,
        architecture: ArchitectureSpec,
        performance: float,
        ot_alignment: float,
        source: str,
    ) -> CacheEntry:
        with self._conn() as conn:
            current = conn.execute(
                """
                SELECT * FROM cache_versions
                WHERE task_type = ? AND is_current = 1
                ORDER BY version_id DESC LIMIT 1
                """,
                (signature.primary_type,),
            ).fetchone()
            if current is None:
                version_id = 0
            else:
                version_id = int(current["version_id"]) + 1
                conn.execute(
                    "UPDATE cache_versions SET is_current = 0 WHERE task_type = ?",
                    (signature.primary_type,),
                )
            conn.execute(
                """
                INSERT INTO cache_versions (
                    task_type, version_id, signature_json, architecture_json,
                    avg_perf, sample_count, ot_alignment, needs_revalidation, is_current, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1, ?, datetime('now'))
                """,
                (
                    signature.primary_type,
                    version_id,
                    signature.model_dump_json(),
                    architecture.model_dump_json(),
                    performance,
                    1,
                    ot_alignment,
                    source,
                ),
            )
        return CacheEntry(
            task_type=signature.primary_type,
            version_id=version_id,
            signature=signature,
            architecture=architecture,
            avg_perf=performance,
            sample_count=1,
            ot_alignment=ot_alignment,
            needs_revalidation=False,
        )

    def mark_needs_revalidation(self, task_type: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE cache_versions
                SET needs_revalidation = 1
                WHERE task_type = ? AND is_current = 1
                """,
                (task_type,),
            )

    def rollback_if_needed(self, task_type: str) -> None:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM cache_versions
                WHERE task_type = ?
                ORDER BY version_id DESC LIMIT 2
                """,
                (task_type,),
            ).fetchall()
            if len(rows) < 2:
                return
            current, previous = rows[0], rows[1]
            if float(current["avg_perf"]) < float(previous["avg_perf"]):
                conn.execute("UPDATE cache_versions SET is_current = 0 WHERE id = ?", (current["id"],))
                conn.execute("UPDATE cache_versions SET is_current = 1 WHERE id = ?", (previous["id"],))

    def get_current(self, task_type: str) -> CacheEntry | None:
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT * FROM cache_versions
                WHERE task_type = ? AND is_current = 1
                ORDER BY version_id DESC LIMIT 1
                """,
                (task_type,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_cache_entry(row)

    def _row_to_cache_entry(self, row: sqlite3.Row) -> CacheEntry:
        signature = TaskSignature.model_validate_json(str(row["signature_json"]))
        architecture = ArchitectureSpec.model_validate_json(str(row["architecture_json"]))
        return CacheEntry(
            task_type=str(row["task_type"]),
            version_id=int(row["version_id"]),
            signature=signature,
            architecture=architecture,
            avg_perf=float(row["avg_perf"]),
            sample_count=int(row["sample_count"]),
            ot_alignment=float(row["ot_alignment"]),
            needs_revalidation=bool(row["needs_revalidation"]),
        )


def _signature_similarity(a: TaskSignature, b: TaskSignature, type_weight: float) -> float:
    same_type = 1.0 if a.primary_type == b.primary_type else 0.0
    char_match = sum(
        [
            1.0 if a.uncertainty == b.uncertainty else 0.0,
            1.0 if a.complexity == b.complexity else 0.0,
            1.0 if a.interdependence == b.interdependence else 0.0,
        ]
    )
    char_sim = char_match / 3.0
    return type_weight * same_type + (1.0 - type_weight) * char_sim
