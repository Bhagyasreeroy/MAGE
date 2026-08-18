# Pulling `feat/m2-goal-orchestrator` — read this first

> Written 18 Aug 2026. Applies to anyone pulling this branch onto a machine that
> already has MAGE running. If you are starting from an empty database, none of
> this affects you — skip to §3.

---

## 1. The database needs five new columns

The merge from `integration/combined-features` added **version lineage** to the
`datasets` table (`root_id`, `parent_id`, `version`, `transform_type`,
`transform_params`) so the dataset workbench can track transforms.

**`Base.metadata.create_all` will not add them.** It creates *missing tables*; it
never `ALTER`s an existing one. So on an existing database the table stays as it
was, and every dataset write fails with:

```
asyncpg.exceptions.UndefinedColumnError:
column "root_id" of relation "datasets" does not exist
```

That surfaces as ~56 test failures and a broken upload path. It is not a code
bug — it is schema drift.

### Back up first

```bash
docker exec mage-postgres pg_dump -U mage -d mage > ~/mage-db-backup.sql
```

### Then either drop the table…

Fine if you do not care about your local data:

```bash
docker exec -i mage-postgres psql -U mage -d mage -c "DROP TABLE datasets CASCADE;"
```

…and restart the backend so `create_all` rebuilds it.

### …or migrate in place

Keeps your data. Existing rows are all original uploads, so each becomes its own
root at version 1:

```bash
docker exec -i mage-postgres psql -U mage -d mage <<'SQL'
BEGIN;
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS root_id          VARCHAR(36);
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS parent_id        VARCHAR(36);
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS version          INTEGER;
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS transform_type   VARCHAR(20);
ALTER TABLE datasets ADD COLUMN IF NOT EXISTS transform_params JSON;

UPDATE datasets SET root_id = id WHERE root_id IS NULL;
UPDATE datasets SET version = 1  WHERE version IS NULL;

ALTER TABLE datasets ALTER COLUMN root_id SET NOT NULL;
ALTER TABLE datasets ALTER COLUMN version SET NOT NULL;
ALTER TABLE datasets ALTER COLUMN version SET DEFAULT 1;

DO $$ BEGIN
  ALTER TABLE datasets ADD CONSTRAINT datasets_root_id_fkey
    FOREIGN KEY (root_id) REFERENCES datasets(id) ON DELETE CASCADE;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN
  ALTER TABLE datasets ADD CONSTRAINT datasets_parent_id_fkey
    FOREIGN KEY (parent_id) REFERENCES datasets(id) ON DELETE SET NULL;
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

CREATE INDEX IF NOT EXISTS ix_datasets_root_id   ON datasets(root_id);
CREATE INDEX IF NOT EXISTS ix_datasets_parent_id ON datasets(parent_id);
COMMIT;
SQL
```

---

## 2. New Python dependency

`duckdb` (the workbench SQL console). It is in `backend/pyproject.toml`, but your
local venv will not have it:

```bash
pip install -e "backend/[dev]"     # or: pip install duckdb
```

If you run in Docker, **rebuild** — a cached image predates it:

```bash
docker compose build backend
```

---

## 3. Gemini key (optional)

Set `GEMINI_API_KEY` in `.env` to enable LLM chat mode, "Explain further", and
ask-in-English → SQL.

**Everything works without it.** Every call site checks `is_configured` and falls
back to the deterministic RAG path. No key is a fully working system, just
without synthesis.

---

## 4. Running the tests

```bash
docker compose up -d postgres          # tests/backend/ needs it
PYTHONPATH=. pytest tests/ -q          # expect: 669 passed, 1 skipped
```

Without Postgres you will see ~100 failures in `tests/backend/` that mean nothing
about the code.

---

## 5. Two things to know before quoting numbers in the report

1. **Computation divergence is 0.836, not 0.940.** Broadening the `reporting`
   task's computations (commit `5e598f0`) made it overlap the other four task
   types. This was a deliberate, documented trade-off — see
   `docs/BUILD_LOG_2026-08-17.md` §7 for the reasoning and the three options that
   were measured. Precision moved 1.000 → 0.883 for the same reason. Quote the
   number *and* the reason together.
2. **`MAGE_Project_Report.pdf` is gitignored.** You will not get it from the repo
   — regenerate it with `python scripts/generate_project_report.py`, and only
   after the final harness run so its figures match.
