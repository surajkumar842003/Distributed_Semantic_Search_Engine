-- Post-Bulk-Load: Add tsvector column and build GIN index
-- Run this AFTER streaming binary COPY of all 20M chunks completes.
--
-- This script:
--   1. Adds a tsvector column (initially NULL)
--   2. Populates it in parallel batches using UPDATE
--   3. Builds the GIN index CONCURRENTLY (no exclusive table lock)
--
-- Expected time on 512GB RAM / 128 cores with maintenance_work_mem=32GB:
--   Step 1: instant
--   Step 2: ~2-5 minutes (parallel UPDATE across 16 workers)
--   Step 3: ~15-30 minutes (parallel GIN build)

-- Step 1: Add the tsvector column
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_tsv TSVECTOR;

-- Step 2: Populate tsvector from text column
-- Uses the 'english' dictionary for stemming, stop-word removal
UPDATE chunks SET text_tsv = to_tsvector('english', text);

-- Step 3: Build GIN index CONCURRENTLY
-- CONCURRENTLY avoids holding an exclusive lock on the table,
-- allowing read queries to proceed during index construction.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_text_tsv
    ON chunks USING GIN(text_tsv);

-- Step 4: Analyze the table to update planner statistics
ANALYZE chunks;

-- Verification: Report row count and index sizes
SELECT
    'chunks' AS table_name,
    pg_size_pretty(pg_total_relation_size('chunks')) AS total_size,
    pg_size_pretty(pg_relation_size('chunks')) AS table_size,
    pg_size_pretty(pg_indexes_size('chunks')) AS index_size,
    (SELECT count(*) FROM chunks) AS row_count;

