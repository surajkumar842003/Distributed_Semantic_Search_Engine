-- Migration 002: Add tsvector column, GIN index, and auto-update trigger for Full-Text Search

-- 1. Add tsvector column if not present
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS text_tsv TSVECTOR;

-- 2. Populate tsvector for any existing records
UPDATE chunks
SET text_tsv = to_tsvector('english', coalesce(text, ''))
WHERE text_tsv IS NULL;

-- 3. Create GIN index for high-speed full-text keyword retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_text_tsv ON chunks USING GIN(text_tsv);

-- 4. Trigger function to automatically maintain text_tsv on insert/update
CREATE OR REPLACE FUNCTION chunks_text_tsv_trigger() RETURNS trigger AS $$
BEGIN
    NEW.text_tsv := to_tsvector('english', coalesce(NEW.text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 5. Attach trigger to chunks table
DROP TRIGGER IF EXISTS trg_chunks_text_tsv ON chunks;
CREATE TRIGGER trg_chunks_text_tsv
    BEFORE INSERT OR UPDATE OF text ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_text_tsv_trigger();

