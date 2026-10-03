from __future__ import annotations

import sqlite3

from openagora.storage.sqlite_store import SQLiteStore


def test_sqlite_store_migrates_from_v1_schema(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO schema_meta (key, value) VALUES ('schema_version', '1');

        CREATE TABLE runs (
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

        CREATE TABLE events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            ts TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );

        CREATE TABLE metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            key TEXT NOT NULL,
            value REAL NOT NULL
        );

        CREATE TABLE cache_versions (
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
    conn.commit()
    conn.close()

    store = SQLiteStore(str(db_path))
    assert store.schema_version() == SQLiteStore.CURRENT_SCHEMA_VERSION

    conn = sqlite3.connect(str(db_path))
    event_columns = {row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
    run_columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    conn.close()

    assert "schema_version" in event_columns
    assert "artifact_hashes_path" in run_columns
