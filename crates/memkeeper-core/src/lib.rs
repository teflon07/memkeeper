#![forbid(unsafe_code)]

//! Host-agnostic domain constants for the local memory core.
//!
//! This crate intentionally has no host-adapter, database, model, or network
//! dependencies. Durable memory semantics belong here before Pi/MCP/IDE
//! adapters translate them to host-specific tools.

/// Default graph-equivalent memory space for operational workspace memory.
pub const DEFAULT_SPACE: &str = "workspace-memory";

/// Reserved sentinel in a read-side space filter meaning "every space". A
/// retrieval scoped to `["*"]` drops the `space_name` predicate entirely
/// (cross-space union) instead of collapsing to [`DEFAULT_SPACE`]. It is not a
/// valid space name: writes and space creation reject it so it can never collide
/// with a real space.
pub const ALL_SPACES: &str = "*";

/// Default durable silo for high-value operational decisions.
pub const DEFAULT_DURABLE_SILO: &str = "durable";

/// Supported memory status values for the v0.1 active projection.
pub mod status {
    /// Visible in default retrieval.
    pub const ACTIVE: &str = "active";
    /// Preserved as history but suppressed by default.
    pub const SUPERSEDED: &str = "superseded";
    /// Visible only when explicitly requested or reviewed.
    pub const CONFLICTED: &str = "conflicted";
    /// Soft-deleted; preserved for audit/export by default.
    pub const TOMBSTONED: &str = "tombstoned";
    /// Retention policy expired; suppressed by default.
    pub const EXPIRED: &str = "expired";
}

/// Supported default scopes for v0.1.
pub mod scope {
    /// Applies across all workspaces.
    pub const GLOBAL: &str = "global";
    /// Applies to the current workspace.
    pub const WORKSPACE: &str = "workspace";
    /// Applies to a project within a workspace.
    pub const PROJECT: &str = "project";
    /// Applies to a session/branch of work.
    pub const SESSION: &str = "session";
    /// Caller-defined scope.
    pub const CUSTOM: &str = "custom";
}

/// Supported default memory kinds for v0.1.
pub mod kind {
    /// General remembered fact.
    pub const FACT: &str = "fact";
    /// Accepted decision.
    pub const DECISION: &str = "decision";
    /// User/project preference.
    pub const PREFERENCE: &str = "preference";
    /// Lesson learned.
    pub const LESSON: &str = "lesson";
    /// Task memory.
    pub const TASK: &str = "task";
    /// Completed action memory.
    pub const ACTION: &str = "action";
    /// Fiction/project continuity memory that may intentionally coexist with conflicting variants.
    pub const CONTINUITY: &str = "continuity";
    /// Compact summary.
    pub const SUMMARY: &str = "summary";
    /// Reference/source note.
    pub const REFERENCE: &str = "reference";
    /// Explicit entity memory.
    pub const ENTITY: &str = "entity";
}

/// Returns true when a memory status should be included in default retrieval.
#[must_use]
pub const fn is_default_search_status(status: &str) -> bool {
    matches!(status.as_bytes(), b"active")
}

/// Deterministically infer a memory kind from an accepted workspace-memory prefix.
#[must_use]
pub fn infer_kind_from_prefix(content: &str) -> Option<&'static str> {
    let trimmed = content.trim_start();
    if trimmed.starts_with("remember:") || trimmed.starts_with("fact:") {
        Some(kind::FACT)
    } else if trimmed.starts_with("decision:") {
        Some(kind::DECISION)
    } else if trimmed.starts_with("preference:") {
        Some(kind::PREFERENCE)
    } else if trimmed.starts_with("action:") {
        Some(kind::ACTION)
    } else if trimmed.starts_with("lesson:") {
        Some(kind::LESSON)
    } else if trimmed.starts_with("revert:") {
        Some(kind::DECISION)
    } else {
        None
    }
}

/// Deterministic key derivation for `remember`'s `derive_keys` option.
///
/// When a caller stores a memory without supplying `entity_key`/`claim_key`,
/// these helpers fabricate stable keys from the memory text so the write still
/// participates in entity/claim grouping and exact-duplicate supersession
/// instead of being orphaned with `NULL` keys.
///
/// Derivation is intentionally conservative: an `entity_key` is a normalized
/// slug of the salient tokens, and a `claim_key` is a stable content
/// fingerprint. It is **not** a semantic matcher. A reworded restatement of the
/// same fact produces a different claim fingerprint and therefore coexists with
/// the original rather than being destructively superseded. Catching semantic
/// updates is out of scope for a deterministic, offline derivation.
pub mod derive {
    /// English stopwords dropped from entity slugs so the key reflects the
    /// memory's salient nouns/verbs, not connective tissue.
    const STOPWORDS: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "had", "has",
        "have", "he", "her", "his", "in", "into", "is", "it", "its", "of", "on", "or", "our",
        "that", "the", "their", "them", "then", "there", "these", "they", "this", "to", "was",
        "we", "were", "what", "when", "which", "who", "will", "with", "you",
    ];

    /// Maximum number of significant tokens kept in a derived entity slug.
    const ENTITY_SLUG_MAX_TOKENS: usize = 6;
    /// Hard cap on derived slug length (characters) to keep keys index-friendly.
    const SLUG_MAX_LEN: usize = 80;
    /// Prefix marking a key as machine-derived rather than caller-supplied.
    const DERIVED_PREFIX: &str = "auto:";

    /// Lowercase, strip a leading `word:` prefix (e.g. `decision:`), and split
    /// into alphanumeric tokens. Shared normalization for both derivations.
    fn tokens(text: &str) -> Vec<String> {
        let lowered = text.trim().to_lowercase();
        let body = match lowered.split_once(':') {
            Some((head, rest))
                if !head.is_empty() && head.chars().all(|c| c.is_ascii_alphabetic()) =>
            {
                rest
            }
            _ => lowered.as_str(),
        };
        body.split(|c: char| !c.is_alphanumeric())
            .filter(|token| !token.is_empty())
            .map(ToString::to_string)
            .collect()
    }

    /// Build a hyphen-joined slug from the first significant tokens of `text`.
    /// Returns `None` when no usable tokens remain (e.g. all stopwords/symbols).
    #[must_use]
    pub fn entity_slug(text: &str) -> Option<String> {
        let mut slug = String::new();
        for token in tokens(text)
            .into_iter()
            .filter(|token| token.len() >= 2 && !STOPWORDS.contains(&token.as_str()))
            .take(ENTITY_SLUG_MAX_TOKENS)
        {
            if !slug.is_empty() {
                slug.push('-');
            }
            slug.push_str(&token);
            if slug.len() >= SLUG_MAX_LEN {
                break;
            }
        }
        let mut end = SLUG_MAX_LEN.min(slug.len());
        while end > 0 && !slug.is_char_boundary(end) {
            end -= 1;
        }
        slug.truncate(end);
        while slug.ends_with('-') {
            slug.pop();
        }
        if slug.is_empty() {
            None
        } else {
            Some(slug)
        }
    }

    /// Stable 64-bit FNV-1a fingerprint of `text`, rendered as 16 lowercase hex
    /// digits. FNV-1a is used deliberately (not `std`'s `DefaultHasher`, whose
    /// output is not guaranteed stable across Rust releases) because the value
    /// is persisted as a key and must stay identical across upgrades. Internal
    /// whitespace is collapsed and case folded so trivially-different formatting
    /// of identical text collapses to the same fingerprint.
    #[must_use]
    pub fn fingerprint(text: &str) -> String {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
        let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let normalized = normalized.to_lowercase();
        let mut hash = FNV_OFFSET;
        for byte in normalized.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        format!("{hash:016x}")
    }

    /// Derive `(entity_key, claim_key)` for a memory.
    ///
    /// `summary` is preferred as the entity-slug source when present (it is the
    /// human-curated gist); otherwise the slug comes from `content`. The claim
    /// fingerprint is always taken from `content` so the claim key tracks the
    /// actual stored text. Returns `(None, None)` when `content` yields no
    /// usable slug tokens, leaving the memory key-less exactly as it would be
    /// without the option.
    #[must_use]
    pub fn keys(content: &str, summary: Option<&str>) -> (Option<String>, Option<String>) {
        let slug_source = summary
            .map(str::trim)
            .filter(|summary| !summary.is_empty())
            .unwrap_or(content);
        match entity_slug(slug_source) {
            Some(slug) => (
                Some(format!("{DERIVED_PREFIX}{slug}")),
                Some(format!("{DERIVED_PREFIX}{}", fingerprint(content))),
            ),
            None => (None, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{infer_kind_from_prefix, is_default_search_status, kind, status};

    #[test]
    fn active_is_default_search_status() {
        assert!(is_default_search_status(status::ACTIVE));
        assert!(!is_default_search_status(status::SUPERSEDED));
        assert!(!is_default_search_status(status::TOMBSTONED));
    }

    #[test]
    fn deterministic_prefixes_infer_kind() {
        assert_eq!(
            infer_kind_from_prefix("remember: use sqlite"),
            Some(kind::FACT)
        );
        assert_eq!(
            infer_kind_from_prefix("fact: sqlite is canonical"),
            Some(kind::FACT)
        );
        assert_eq!(
            infer_kind_from_prefix("decision: use sqlite"),
            Some(kind::DECISION)
        );
        assert_eq!(
            infer_kind_from_prefix("preference: keep output concise"),
            Some(kind::PREFERENCE)
        );
        assert_eq!(
            infer_kind_from_prefix(" action: ran benchmark"),
            Some(kind::ACTION)
        );
        assert_eq!(
            infer_kind_from_prefix("lesson: keep query fast"),
            Some(kind::LESSON)
        );
        assert_eq!(infer_kind_from_prefix("plain fact"), None);
    }
}

#[cfg(test)]
mod derive_tests {
    use super::derive;

    #[test]
    fn entity_slug_drops_stopwords_and_prefix() {
        assert_eq!(
            derive::entity_slug("decision: use SQLite as the canonical store"),
            Some("use-sqlite-canonical-store".to_string())
        );
    }

    #[test]
    fn entity_slug_caps_token_count() {
        let slug =
            derive::entity_slug("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda")
                .unwrap();
        assert_eq!(slug, "alpha-beta-gamma-delta-epsilon-zeta");
    }

    #[test]
    fn entity_slug_truncates_multibyte_text_on_a_char_boundary() {
        let slug = derive::entity_slug(&"界".repeat(30)).unwrap();

        assert!(slug.len() <= 80);
        assert!(slug.chars().all(|ch| ch == '界'));
    }

    #[test]
    fn entity_slug_returns_none_without_usable_tokens() {
        assert_eq!(derive::entity_slug("the and of to"), None);
        assert_eq!(derive::entity_slug("   --- !!! "), None);
    }

    #[test]
    fn fingerprint_is_stable_and_whitespace_insensitive() {
        let canonical = derive::fingerprint("Prefer concise answers");
        assert_eq!(canonical, derive::fingerprint("Prefer concise answers"));
        assert_eq!(
            canonical,
            derive::fingerprint("  prefer   concise\nanswers  ")
        );
        assert_ne!(canonical, derive::fingerprint("Prefer verbose answers"));
        assert_eq!(canonical.len(), 16);
        assert!(canonical.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn keys_are_prefixed_and_identical_for_identical_content() {
        let (entity, claim) = derive::keys("Voyage free tier trains on your data", None);
        let entity = entity.expect("entity key");
        let claim = claim.expect("claim key");
        assert!(entity.starts_with("auto:"));
        assert!(claim.starts_with("auto:"));
        // Identical content -> identical keys (the supersession hook).
        assert_eq!((entity.clone(), claim.clone()), {
            let (e, c) = derive::keys("Voyage free tier trains on your data", None);
            (e.unwrap(), c.unwrap())
        });
    }

    #[test]
    fn summary_drives_entity_but_content_drives_claim() {
        let (entity_a, claim_a) =
            derive::keys("long-form content about reranking", Some("rerank policy"));
        let (entity_b, claim_b) = derive::keys(
            "different long-form content entirely",
            Some("rerank policy"),
        );
        // Same summary -> same entity grouping.
        assert_eq!(entity_a, entity_b);
        assert_eq!(entity_a, Some("auto:rerank-policy".to_string()));
        // Different content -> different claim, so the two coexist (no supersede).
        assert_ne!(claim_a, claim_b);
    }

    #[test]
    fn keys_none_when_content_has_no_tokens() {
        assert_eq!(derive::keys("the of and", None), (None, None));
    }
}
