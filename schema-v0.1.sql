-- memkeeper SQLite schema v0.1 draft
-- Status: Draft 0.1
-- Date: 2026-05-25
-- Scope: deterministic local MVP; SQLite is canonical; vector/entity sidecars are rebuildable.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA busy_timeout = 5000;

-- Migration bookkeeping. The implementation should also set PRAGMA user_version = 1.
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  applied_at TEXT NOT NULL,
  checksum_sha256 TEXT
);

CREATE TABLE IF NOT EXISTS config_kv (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Space = Zep graph equivalent / top-level isolation boundary.
CREATE TABLE IF NOT EXISTS spaces (
  name TEXT PRIMARY KEY,
  display_name TEXT,
  description TEXT,
  default_silo TEXT NOT NULL DEFAULT 'durable',
  ontology TEXT,
  config_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

-- Silo = policy domain inside a space.
CREATE TABLE IF NOT EXISTS silos (
  space_name TEXT NOT NULL,
  name TEXT NOT NULL,
  description TEXT,
  retention_policy TEXT NOT NULL DEFAULT 'keep',
  default_scope TEXT NOT NULL DEFAULT 'workspace',
  config_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (space_name, name),
  FOREIGN KEY (space_name) REFERENCES spaces(name) ON DELETE CASCADE
);

-- Source episodes/documents are raw or lightly structured inputs.
-- Full canonical sources may still live in Git; this table stores chunks/provenance/queryable text.
CREATE TABLE IF NOT EXISTS source_episodes (
  id TEXT PRIMARY KEY,
  space_name TEXT NOT NULL,
  source_type TEXT NOT NULL, -- manual|host_session|manuscript|review_artifact|zep_export|tool|import
  source_uri TEXT,
  source_path TEXT,
  source_description TEXT,
  content TEXT,
  content_sha256 TEXT,
  chunk_index INTEGER NOT NULL DEFAULT 0,
  chunk_count INTEGER NOT NULL DEFAULT 1,
  authority TEXT, -- operational-memory|manuscript-canon|review-artifact-summary|...
  work_id TEXT,
  chapter_id TEXT,
  artifact_id TEXT,
  git_commit TEXT,
  tool_version TEXT,
  metadata_json TEXT,
  ingest_status TEXT NOT NULL DEFAULT 'indexed'
    CHECK (ingest_status IN ('pending','indexed','extracted','failed','ignored')),
  ingested_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (space_name) REFERENCES spaces(name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_source_episodes_space_status
  ON source_episodes(space_name, ingest_status, ingested_at DESC);
CREATE INDEX IF NOT EXISTS idx_source_episodes_hash
  ON source_episodes(space_name, content_sha256);
CREATE INDEX IF NOT EXISTS idx_source_episodes_path
  ON source_episodes(space_name, source_path, chunk_index);

-- Memory header/current projection. Text lives in memory_versions.
CREATE TABLE IF NOT EXISTS memories (
  id TEXT PRIMARY KEY,
  space_name TEXT NOT NULL,
  silo_name TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT 'workspace'
    CHECK (scope IN ('global','workspace','project','session','custom')),
  project_key TEXT,
  kind TEXT NOT NULL DEFAULT 'fact',
  entity_key TEXT,
  claim_key TEXT,
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active','superseded','conflicted','tombstoned','expired')),
  active_version_id TEXT,
  confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
  pinned INTEGER NOT NULL DEFAULT 0 CHECK (pinned IN (0,1)),
  source_episode_id TEXT,
  valid_from TEXT,
  valid_to TEXT,
  observed_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  accessed_at TEXT,
  expires_at TEXT,
  deleted_at TEXT,
  metadata_json TEXT,
  FOREIGN KEY (space_name) REFERENCES spaces(name) ON DELETE CASCADE,
  FOREIGN KEY (space_name, silo_name) REFERENCES silos(space_name, name),
  FOREIGN KEY (source_episode_id) REFERENCES source_episodes(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_space_status_kind
  ON memories(space_name, status, kind, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_space_status_updated
  ON memories(space_name, status, updated_at DESC, observed_at DESC, id ASC);
CREATE INDEX IF NOT EXISTS idx_memories_space_status_created
  ON memories(space_name, status, created_at DESC, updated_at DESC, id ASC);
CREATE INDEX IF NOT EXISTS idx_memories_silo_status
  ON memories(space_name, silo_name, status, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_project_status
  ON memories(space_name, project_key, status, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_keys
  ON memories(space_name, entity_key, claim_key, status);
CREATE INDEX IF NOT EXISTS idx_memories_source_episode
  ON memories(source_episode_id);
CREATE INDEX IF NOT EXISTS idx_memories_expires
  ON memories(status, expires_at);

-- Immutable memory text/version records.
CREATE TABLE IF NOT EXISTS memory_versions (
  id TEXT PRIMARY KEY,
  memory_id TEXT NOT NULL,
  version_num INTEGER NOT NULL,
  content TEXT NOT NULL,
  summary TEXT,
  content_sha256 TEXT NOT NULL,
  source_episode_id TEXT,
  source_ref_json TEXT,
  created_at TEXT NOT NULL,
  created_by TEXT,
  event_id TEXT,
  UNIQUE (memory_id, version_num),
  FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
  FOREIGN KEY (source_episode_id) REFERENCES source_episodes(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_versions_memory
  ON memory_versions(memory_id, version_num DESC);
CREATE INDEX IF NOT EXISTS idx_memory_versions_hash
  ON memory_versions(content_sha256);

-- Append-only-ish event log. Corrections should add events rather than mutate history.
CREATE TABLE IF NOT EXISTS memory_events (
  id TEXT PRIMARY KEY,
  memory_id TEXT,
  event_type TEXT NOT NULL, -- remember|update|forget|supersede|conflict|resolve|import|dream|...
  old_status TEXT,
  new_status TEXT,
  actor TEXT,
  reason TEXT,
  data_json TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_events_memory_time
  ON memory_events(memory_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_memory_events_type_time
  ON memory_events(event_type, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_tags (
  memory_id TEXT NOT NULL,
  tag TEXT NOT NULL,
  created_at TEXT NOT NULL,
  PRIMARY KEY (memory_id, tag),
  FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memory_tags_tag
  ON memory_tags(tag, memory_id);

CREATE TABLE IF NOT EXISTS memory_links (
  src_memory_id TEXT NOT NULL,
  dst_memory_id TEXT NOT NULL,
  link_type TEXT NOT NULL
    CHECK (link_type IN ('supersedes','superseded_by','contradicts','duplicates','derived_from','supports','related_to')),
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active','resolved','rejected')),
  confidence REAL CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
  event_id TEXT,
  metadata_json TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY (src_memory_id, dst_memory_id, link_type),
  FOREIGN KEY (src_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
  FOREIGN KEY (dst_memory_id) REFERENCES memories(id) ON DELETE CASCADE,
  FOREIGN KEY (event_id) REFERENCES memory_events(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memory_links_dst
  ON memory_links(dst_memory_id, link_type, status);

CREATE TABLE IF NOT EXISTS conflicts (
  id TEXT PRIMARY KEY,
  space_name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'open'
    CHECK (status IN ('open','resolved','ignored')),
  memory_a_id TEXT NOT NULL,
  memory_b_id TEXT NOT NULL,
  conflict_type TEXT NOT NULL DEFAULT 'contradiction',
  explanation TEXT,
  resolution TEXT,
  resolution_event_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  resolved_at TEXT,
  FOREIGN KEY (space_name) REFERENCES spaces(name) ON DELETE CASCADE,
  FOREIGN KEY (memory_a_id) REFERENCES memories(id) ON DELETE CASCADE,
  FOREIGN KEY (memory_b_id) REFERENCES memories(id) ON DELETE CASCADE,
  FOREIGN KEY (resolution_event_id) REFERENCES memory_events(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_conflicts_space_status
  ON conflicts(space_name, status, created_at DESC);

-- Lightweight entity/relationship projection. v0.1 can populate explicitly or by import.
CREATE TABLE IF NOT EXISTS entities (
  id TEXT PRIMARY KEY,
  space_name TEXT NOT NULL,
  entity_key TEXT NOT NULL,
  entity_type TEXT NOT NULL DEFAULT 'Entity',
  canonical_name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active','merged','tombstoned')),
  confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
  source_episode_id TEXT,
  metadata_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE (space_name, entity_key),
  FOREIGN KEY (space_name) REFERENCES spaces(name) ON DELETE CASCADE,
  FOREIGN KEY (source_episode_id) REFERENCES source_episodes(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_entities_space_type
  ON entities(space_name, entity_type, canonical_name);

CREATE TABLE IF NOT EXISTS entity_aliases (
  entity_id TEXT NOT NULL,
  alias TEXT NOT NULL,
  normalized_alias TEXT NOT NULL,
  source_episode_id TEXT,
  created_at TEXT NOT NULL,
  PRIMARY KEY (entity_id, normalized_alias),
  FOREIGN KEY (entity_id) REFERENCES entities(id) ON DELETE CASCADE,
  FOREIGN KEY (source_episode_id) REFERENCES source_episodes(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_entity_aliases_lookup
  ON entity_aliases(normalized_alias, entity_id);

CREATE TABLE IF NOT EXISTS relationships (
  id TEXT PRIMARY KEY,
  space_name TEXT NOT NULL,
  subject_entity_id TEXT NOT NULL,
  relation_type TEXT NOT NULL,
  object_entity_id TEXT NOT NULL,
  memory_id TEXT,
  source_episode_id TEXT,
  status TEXT NOT NULL DEFAULT 'active'
    CHECK (status IN ('active','superseded','conflicted','tombstoned')),
  confidence REAL NOT NULL DEFAULT 1.0 CHECK (confidence >= 0.0 AND confidence <= 1.0),
  observed_at TEXT,
  valid_from TEXT,
  valid_to TEXT,
  metadata_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (space_name) REFERENCES spaces(name) ON DELETE CASCADE,
  FOREIGN KEY (subject_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
  FOREIGN KEY (object_entity_id) REFERENCES entities(id) ON DELETE CASCADE,
  FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE SET NULL,
  FOREIGN KEY (source_episode_id) REFERENCES source_episodes(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_relationships_subject
  ON relationships(space_name, subject_entity_id, relation_type, status);
CREATE INDEX IF NOT EXISTS idx_relationships_object
  ON relationships(space_name, object_entity_id, relation_type, status);
CREATE INDEX IF NOT EXISTS idx_relationships_memory
  ON relationships(memory_id);

-- Background/idempotent job queue for post-MVP extraction, embeddings, graph projection, dream tasks.
-- v0.1 may create this table even if only used for diagnostics.
CREATE TABLE IF NOT EXISTS processing_jobs (
  id TEXT PRIMARY KEY,
  kind TEXT NOT NULL CHECK (kind IN ('extract','embedding','graph','dream','reindex','import','probe')),
  target_type TEXT NOT NULL,
  target_id TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued','running','succeeded','failed','cancelled')),
  priority INTEGER NOT NULL DEFAULT 100,
  attempts INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  run_after TEXT,
  locked_at TEXT,
  locked_by TEXT,
  input_json TEXT,
  output_json TEXT,
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE (kind, target_type, target_id)
);

CREATE INDEX IF NOT EXISTS idx_processing_jobs_ready
  ON processing_jobs(status, priority, run_after, created_at);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_target
  ON processing_jobs(target_type, target_id);

-- Optional embedding metadata/vector sidecar table. The vector blob is rebuildable.
CREATE TABLE IF NOT EXISTS embeddings (
  id TEXT PRIMARY KEY,
  memory_id TEXT,
  version_id TEXT,
  source_episode_id TEXT,
  embedding_model TEXT NOT NULL,
  dimensions INTEGER,
  vector_blob BLOB,
  status TEXT NOT NULL DEFAULT 'pending'
    CHECK (status IN ('pending','ready','stale','failed')),
  error TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  UNIQUE (version_id, embedding_model),
  FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
  FOREIGN KEY (version_id) REFERENCES memory_versions(id) ON DELETE CASCADE,
  FOREIGN KEY (source_episode_id) REFERENCES source_episodes(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_embeddings_status
  ON embeddings(status, embedding_model, updated_at);

CREATE TABLE IF NOT EXISTS dream_runs (
  id TEXT PRIMARY KEY,
  space_name TEXT,
  status TEXT NOT NULL DEFAULT 'running'
    CHECK (status IN ('running','succeeded','failed','cancelled')),
  started_at TEXT NOT NULL,
  finished_at TEXT,
  budget_json TEXT,
  summary_json TEXT,
  error TEXT,
  FOREIGN KEY (space_name) REFERENCES spaces(name) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS probe_runs (
  id TEXT PRIMARY KEY,
  space_name TEXT NOT NULL,
  probe_pack_name TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'running'
    CHECK (status IN ('running','succeeded','failed')),
  started_at TEXT NOT NULL,
  finished_at TEXT,
  result_json TEXT,
  FOREIGN KEY (space_name) REFERENCES spaces(name) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_probe_runs_space_time
  ON probe_runs(space_name, started_at DESC);

-- FTS tables are application-maintained in the same transaction as memory/source changes.
-- memory_fts should contain only currently searchable memory projections unless history is requested.
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
  memory_id UNINDEXED,
  version_id UNINDEXED,
  space_name UNINDEXED,
  silo_name UNINDEXED,
  status UNINDEXED,
  kind UNINDEXED,
  content,
  summary,
  tags,
  source_text,
  metadata_text,
  tokenize = 'unicode61 remove_diacritics 2'
);

-- Source-free FTS projection used by default include_source=false search so BM25
-- relevance never depends on source/provenance text.
CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts_public USING fts5(
  memory_id UNINDEXED,
  version_id UNINDEXED,
  space_name UNINDEXED,
  silo_name UNINDEXED,
  status UNINDEXED,
  kind UNINDEXED,
  content,
  summary,
  tags,
  metadata_text,
  tokenize = 'unicode61 remove_diacritics 2'
);

CREATE VIRTUAL TABLE IF NOT EXISTS source_episode_fts USING fts5(
  source_episode_id UNINDEXED,
  space_name UNINDEXED,
  source_type UNINDEXED,
  source_path UNINDEXED,
  source_description,
  content,
  metadata_text,
  tokenize = 'unicode61 remove_diacritics 2'
);

-- Default local spaces/silos for the general deterministic MVP. Project-specific
-- spaces such as nora-manuscript/nora-reviews belong to optional profiles,
-- imports, or P1.5 Zep-replacement migrations, not the base schema seed.
-- Timestamps should be supplied by the application in migrations; these
-- CURRENT_TIMESTAMP values are acceptable for initial manual bootstrap only.
INSERT OR IGNORE INTO spaces (name, display_name, description, default_silo, ontology, config_json, created_at, updated_at)
VALUES
  ('workspace-memory', 'Workspace Memory', 'Operational decisions, actions, lessons, reverts, preferences, and setup details.', 'durable', NULL, NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

INSERT OR IGNORE INTO silos (space_name, name, description, retention_policy, default_scope, config_json, created_at, updated_at)
VALUES
  ('workspace-memory', 'short-term', 'Working/session memory.', 'ttl', 'session', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP),
  ('workspace-memory', 'durable', 'High-value decisions and commitments.', 'keep', 'workspace', NULL, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP);

INSERT OR IGNORE INTO config_kv (key, value, updated_at)
VALUES
  ('protocol_version', 'memkeeper.v0.1', CURRENT_TIMESTAMP),
  ('schema_version', '1', CURRENT_TIMESTAMP),
  ('default_space', 'workspace-memory', CURRENT_TIMESTAMP);

INSERT OR IGNORE INTO schema_migrations (version, name, applied_at, checksum_sha256)
VALUES (1, 'schema-v0.1', CURRENT_TIMESTAMP, NULL);

PRAGMA user_version = 1;
