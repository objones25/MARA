-- MARA leaf database schema
-- Applied at connection time via SQLiteLeafRepository.
--
-- PRAGMAs (not stored, applied on every connection):
--   PRAGMA journal_mode = WAL;
--   PRAGMA busy_timeout = 5000;
--   PRAGMA foreign_keys = ON;
--   PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,        -- UUID4
    query           TEXT NOT NULL,
    merkle_root     TEXT NOT NULL DEFAULT '',
    embedding_model TEXT NOT NULL,
    hash_algorithm  TEXT NOT NULL,
    started_at      TEXT NOT NULL,
    completed_at    TEXT                     -- NULL until certified_output runs
);

CREATE TABLE IF NOT EXISTS leaves (
    hash                TEXT PRIMARY KEY,    -- sha256 hex (64 chars)
    url                 TEXT NOT NULL,
    text                TEXT NOT NULL,       -- raw chunk text, exactly as hashed
    retrieved_at        TEXT NOT NULL,       -- ISO-8601, as used in hashing
    contextualized_text TEXT NOT NULL,       -- = text until contextual retrieval
    embedding           BLOB,               -- float32 ndarray.tobytes(), NULL until embedded
    embedding_model     TEXT,               -- model name, NULL until embedded
    parent_hash         TEXT REFERENCES leaves(hash) ON DELETE SET NULL,
    created_at          TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_leaves_url ON leaves(url, retrieved_at DESC);
CREATE INDEX IF NOT EXISTS idx_leaves_embedding_model ON leaves(embedding_model);

CREATE TABLE IF NOT EXISTS run_leaves (
    run_id          TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    leaf_hash       TEXT NOT NULL REFERENCES leaves(hash) ON DELETE CASCADE,
    position_index  INTEGER NOT NULL,        -- leaf's index within this run
    sub_query       TEXT NOT NULL,
    PRIMARY KEY (run_id, leaf_hash)
);

CREATE INDEX IF NOT EXISTS idx_run_leaves_run ON run_leaves(run_id, position_index);

CREATE VIRTUAL TABLE IF NOT EXISTS leaves_fts USING fts5(
    hash UNINDEXED,
    contextualized_text,
    content='leaves',
    content_rowid='rowid',
    tokenize='porter ascii'
);

-- Auto-sync triggers for FTS5
CREATE TRIGGER IF NOT EXISTS leaves_ai AFTER INSERT ON leaves BEGIN
    INSERT INTO leaves_fts(rowid, hash, contextualized_text)
    VALUES (new.rowid, new.hash, new.contextualized_text);
END;

CREATE TRIGGER IF NOT EXISTS leaves_au AFTER UPDATE OF contextualized_text ON leaves BEGIN
    INSERT INTO leaves_fts(leaves_fts, rowid, hash, contextualized_text)
        VALUES ('delete', old.rowid, old.hash, old.contextualized_text);
    INSERT INTO leaves_fts(rowid, hash, contextualized_text)
        VALUES (new.rowid, new.hash, new.contextualized_text);
END;

CREATE TRIGGER IF NOT EXISTS leaves_ad AFTER DELETE ON leaves BEGIN
    INSERT INTO leaves_fts(leaves_fts, rowid, hash, contextualized_text)
        VALUES ('delete', old.rowid, old.hash, old.contextualized_text);
END;
