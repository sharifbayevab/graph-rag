-- advanced-rag-v2/graphrag/stores/pg_schema.sql — metadata, structure, μ (chunks.parent), inverted index
CREATE TABLE IF NOT EXISTS documents (
  doc_id  text PRIMARY KEY,
  section text NOT NULL,
  path    text NOT NULL,
  type    text,
  date    date,
  organ   text,
  lang    text[],
  status  text,
  title   text,
  text    text NOT NULL
);
CREATE TABLE IF NOT EXISTS nodes (
  node_id    text PRIMARY KEY,
  doc_id     text NOT NULL REFERENCES documents ON DELETE CASCADE,
  type       text NOT NULL,
  number     text,
  title      text,
  parent_id  text,
  span_start int NOT NULL,
  span_end   int NOT NULL
);
CREATE INDEX IF NOT EXISTS nodes_doc ON nodes(doc_id);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id   text PRIMARY KEY,
  doc_id     text NOT NULL REFERENCES documents ON DELETE CASCADE,
  parent     text NOT NULL REFERENCES nodes(node_id) ON DELETE CASCADE,
  par        int  NOT NULL,
  tokens     int  NOT NULL,
  src        text,
  text       text NOT NULL,
  text_norm  text NOT NULL,
  index_text text NOT NULL,          -- breadcrumb + "\n" + text_norm: what is indexed and embedded
  span_start int  NOT NULL,
  span_end   int  NOT NULL,
  n_terms    int  NOT NULL           -- BM25 document length |c| = sum of tf over postings
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS postings (
  term     text NOT NULL,
  chunk_id text NOT NULL REFERENCES chunks ON DELETE CASCADE,
  tf       int  NOT NULL,
  -- covering: BM25 reads (term → chunk_id, tf) by index-only scans (after VACUUM, see pg.analyze);
  -- one index instead of PK + a separate postings(term) INCLUDE (chunk_id, tf) duplicate
  CONSTRAINT postings_pkey PRIMARY KEY (term, chunk_id) INCLUDE (tf)
);
CREATE INDEX IF NOT EXISTS postings_chunk ON postings(chunk_id);
-- migrate a pre-covering layout (PRIMARY KEY (term, chunk_id) without INCLUDE) in place
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_index i
             WHERE i.indexrelid = to_regclass('postings_pkey') AND i.indnatts = i.indnkeyatts) THEN
    ALTER TABLE postings DROP CONSTRAINT postings_pkey,
      ADD CONSTRAINT postings_pkey PRIMARY KEY (term, chunk_id) INCLUDE (tf);
  END IF;
END $$;
