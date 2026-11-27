CREATE TABLE IF NOT EXISTS feedback (
  id TEXT PRIMARY KEY NOT NULL,
  alias TEXT NOT NULL DEFAULT '' CHECK (length(alias) <= 80),
  message TEXT NOT NULL CHECK (length(message) BETWEEN 1 AND 2000),
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS feedback_created_at_idx
  ON feedback (created_at DESC);