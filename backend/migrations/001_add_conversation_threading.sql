-- 001_add_conversation_threading.sql
-- ───────────────────────────────────
-- Adds the conversation-threading and sharing columns introduced with the
-- Share feature (root_run_id, is_shared) to an existing analysis_runs table.
--
-- Why this file exists: the schema is otherwise created by
-- Base.metadata.create_all, which creates missing *tables* but never alters
-- an existing one. A database that predates these columns therefore starts
-- failing every INSERT into analysis_runs with UndefinedColumnError, while a
-- freshly created one works — so the breakage only shows up on developer
-- machines and deploys that have existing data.
--
-- Idempotent: safe to run against a database that already has the columns.
--
--   psql "$DATABASE_URL" -f backend/migrations/001_add_conversation_threading.sql

BEGIN;

ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS root_run_id VARCHAR(36);
ALTER TABLE analysis_runs ADD COLUMN IF NOT EXISTS is_shared BOOLEAN NOT NULL DEFAULT FALSE;

-- Every pre-existing run is the root of its own single-turn conversation.
UPDATE analysis_runs SET root_run_id = id WHERE root_run_id IS NULL;

ALTER TABLE analysis_runs ALTER COLUMN root_run_id SET NOT NULL;

DO $$
BEGIN
    ALTER TABLE analysis_runs
        ADD CONSTRAINT analysis_runs_root_run_id_fkey
        FOREIGN KEY (root_run_id) REFERENCES analysis_runs (id) ON DELETE CASCADE;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

CREATE INDEX IF NOT EXISTS ix_analysis_runs_root_run_id ON analysis_runs (root_run_id);

COMMIT;
