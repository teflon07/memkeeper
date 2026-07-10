//! Embedding and reranking provider traits with local (ONNX) and remote (API) backends.

#[cfg(feature = "local")]
use std::path::Path;

use anyhow::{Context, Result};
#[cfg(feature = "local")]
use ndarray::{Array2, ArrayViewD};
#[cfg(feature = "local")]
use ort::session::Session;
#[cfg(feature = "local")]
use ort::value::TensorRef;
#[cfg(feature = "local")]
use tokenizers::Tokenizer;

#[cfg(feature = "local")]
const MAX_SEQ_LEN: usize = 512;

/// Text embedding provider. Implementations may be local (ONNX) or remote (API).
pub trait Embedder: Send {
    /// Stable identifier of the active model, e.g. `mxbai-embed-large`.
    fn model_id(&self) -> &str;
    /// Embedding dimension produced by this model.
    fn dims(&self) -> usize;
    /// Embed a batch of texts, one unit-normalized vector per input.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying model invocation fails.
    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    /// Embed a single text.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying model invocation fails or yields no vector.
    fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        self.embed(&[text])?.pop().context("empty embedding result")
    }
}

/// Cross-encoder reranking provider. Implementations may be local or remote.
pub trait Reranker: Send {
    /// Stable identifier of the active reranker model.
    fn model_id(&self) -> &str;
    /// Score (query, document) pairs; one relevance score per document.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying model invocation fails.
    fn rerank(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>>;
}

#[cfg(feature = "api")]
mod api;
#[cfg(feature = "api")]
pub use api::{ApiEmbedder, ApiReranker};

mod colbert;
#[cfg(feature = "local")]
pub use colbert::ColBertModel;
pub use colbert::{colbert_from_env, TokenEmbedder};

/// Default embedding dimension of the shipping local model
/// (mxbai-embed-large-v1) and the default requested from API providers.
pub const DEFAULT_EMBED_DIMS: usize = 1024;

/// Construct the embedder selected by environment configuration, or `None`
/// (with a stderr warning) when the selected provider is unavailable in this
/// build or misconfigured.
///
/// Env vars: `MEMKEEPER_EMBED_PROVIDER` (`local` default, `openai`),
/// `MEMKEEPER_EMBED_MODEL_DIR` (local), `MEMKEEPER_EMBED_API_KEY`,
/// `MEMKEEPER_EMBED_MODEL`, `MEMKEEPER_EMBED_BASE_URL`, `MEMKEEPER_EMBED_DIMS`
/// (API), and `MEMKEEPER_EMBED_POOLING` (local, resolved at model load).
///
/// `openai` covers any `OpenAI`-compatible endpoint (e.g. `OpenRouter`) via
/// `MEMKEEPER_EMBED_BASE_URL`.
#[must_use]
pub fn embedder_from_env() -> Option<Box<dyn Embedder>> {
    let provider =
        std::env::var("MEMKEEPER_EMBED_PROVIDER").unwrap_or_else(|_| "local".to_string());
    #[cfg(feature = "local")]
    if provider == "local" {
        return local_embedder_from_env().map(|model| Box::new(model) as Box<dyn Embedder>);
    }
    #[cfg(feature = "api")]
    if provider == "openai" {
        return api_embedder_from_env();
    }
    eprintln!(
        "[memkeeper] embedding provider '{provider}' is not available in this build; embedding disabled"
    );
    None
}

/// Construct the reranker selected by environment configuration, or `None`
/// (with a stderr warning unless explicitly disabled with provider `none`).
///
/// Env vars: `MEMKEEPER_RERANK_PROVIDER` (`local` default, `none`, or a remote
/// API dialect `cohere` / `openrouter` / `api`), `MEMKEEPER_RERANK_MODEL_DIR`
/// (local), and for the remote dialects `MEMKEEPER_RERANK_API_KEY`,
/// `MEMKEEPER_RERANK_MODEL`, `MEMKEEPER_RERANK_BASE_URL`.
#[must_use]
pub fn reranker_from_env() -> Option<Box<dyn Reranker>> {
    let provider =
        std::env::var("MEMKEEPER_RERANK_PROVIDER").unwrap_or_else(|_| "local".to_string());
    if provider == "none" {
        return None;
    }
    #[cfg(feature = "local")]
    if provider == "local" {
        return local_reranker_from_env().map(|model| Box::new(model) as Box<dyn Reranker>);
    }
    #[cfg(not(feature = "local"))]
    if provider == "local" {
        // No local reranker compiled into this build; reranking is disabled.
        return None;
    }
    #[cfg(feature = "api")]
    if provider == "cohere" || provider == "openrouter" || provider == "api" {
        return api_reranker_from_env(&provider);
    }
    eprintln!(
        "[memkeeper] rerank provider '{provider}' is not available in this build; reranking disabled"
    );
    None
}

/// Build a remote reranker (Cohere `/rerank` dialect, also spoken by `OpenRouter`
/// and compatible proxies) from environment configuration. Returns `None` (with
/// a stderr warning) when the API key is missing.
#[cfg(feature = "api")]
fn api_reranker_from_env(provider: &str) -> Option<Box<dyn Reranker>> {
    let api_key = std::env::var("MEMKEEPER_RERANK_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        eprintln!("[memkeeper] MEMKEEPER_RERANK_API_KEY not set; API reranking disabled");
        return None;
    }
    let (default_endpoint, default_model) = if provider == "cohere" {
        ("https://api.cohere.com/v2/rerank", "rerank-v3.5")
    } else {
        ("https://openrouter.ai/api/v1/rerank", "cohere/rerank-v3.5")
    };
    let endpoint =
        std::env::var("MEMKEEPER_RERANK_BASE_URL").unwrap_or_else(|_| default_endpoint.to_string());
    let model =
        std::env::var("MEMKEEPER_RERANK_MODEL").unwrap_or_else(|_| default_model.to_string());
    Some(Box::new(ApiReranker::new(model, endpoint, api_key)) as Box<dyn Reranker>)
}

/// Pure model-dir resolution: explicit dir wins; else `<root>/<subdir>` where
/// root is `$MEMKEEPER_MODELS_DIR`, else `<home>/.memkeeper/models`, else `./...`.
/// This prefers explicit configuration, then checked-in model directories near
/// the running binary/workspace, then the `pull-models` default. Pure (env
/// passed in) so it is unit testable without mutating process environment.
#[cfg(feature = "local")]
fn resolve_model_dir(
    explicit: Option<std::ffi::OsString>,
    models_dir: Option<std::ffi::OsString>,
    home: Option<std::ffi::OsString>,
    candidate_roots: &[std::path::PathBuf],
    subdir: &str,
) -> std::path::PathBuf {
    use std::path::PathBuf;
    if let Some(explicit) = explicit {
        return PathBuf::from(explicit);
    }
    if let Some(models_dir) = models_dir {
        return PathBuf::from(models_dir).join(subdir);
    }
    if let Some(discovered) = candidate_roots
        .iter()
        .map(|root| root.join(subdir))
        .find(|dir| dir.join("model.onnx").is_file())
    {
        return discovered;
    }
    home.map_or_else(|| PathBuf::from("."), PathBuf::from)
        .join(".memkeeper")
        .join("models")
        .join(subdir)
}

#[cfg(feature = "local")]
fn model_root_candidates() -> Vec<std::path::PathBuf> {
    let mut roots = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        push_model_root_candidates(&mut roots, &exe);
    }
    if let Ok(cwd) = std::env::current_dir() {
        push_model_root_candidates(&mut roots, &cwd);
    }
    roots
}

#[cfg(feature = "local")]
fn push_model_root_candidates(roots: &mut Vec<std::path::PathBuf>, anchor: &std::path::Path) {
    for ancestor in anchor.ancestors() {
        push_unique_path(roots, ancestor.join("models"));
        push_unique_path(
            roots,
            ancestor.join("memory").join("memkeeper").join("models"),
        );
    }
}

#[cfg(feature = "local")]
fn push_unique_path(roots: &mut Vec<std::path::PathBuf>, path: std::path::PathBuf) {
    if !roots.iter().any(|existing| existing == &path) {
        roots.push(path);
    }
}

/// Resolve a local model dir from the live environment (explicit `env_var`, else
/// a checked-in nearby model dir, else the `pull-models` default location for
/// `subdir`).
#[cfg(feature = "local")]
fn resolve_local_model_dir(env_var: &str, subdir: &str) -> std::path::PathBuf {
    resolve_model_dir(
        std::env::var_os(env_var),
        std::env::var_os("MEMKEEPER_MODELS_DIR"),
        std::env::var_os("HOME"),
        &model_root_candidates(),
        subdir,
    )
}

/// Where the local embed/rerank models are expected and whether they're present.
/// Lets `doctor` report semantic readiness and point at `pull-models` instead of
/// the user discovering lexical fallback only when a query underperforms.
#[cfg(feature = "local")]
#[derive(Debug, Clone)]
pub struct LocalModelStatus {
    /// Resolved directory where the embed model is expected.
    pub embed_dir: std::path::PathBuf,
    /// Whether the embed `model.onnx` is present in `embed_dir`.
    pub embed_present: bool,
    /// Resolved directory where the rerank model is expected.
    pub rerank_dir: std::path::PathBuf,
    /// Whether the rerank `model.onnx` is present in `rerank_dir`.
    pub rerank_present: bool,
}

/// Report the resolved local model dirs and whether each model file is present.
/// Presence checks the `model.onnx` file (an empty dir is not "present").
#[cfg(feature = "local")]
#[must_use]
pub fn local_model_status() -> LocalModelStatus {
    let embed_dir = resolve_local_model_dir("MEMKEEPER_EMBED_MODEL_DIR", "mxbai-embed-large");
    let rerank_dir = resolve_local_model_dir("MEMKEEPER_RERANK_MODEL_DIR", "mxbai-rerank-base");
    LocalModelStatus {
        embed_present: embed_dir.join("model.onnx").is_file(),
        rerank_present: rerank_dir.join("model.onnx").is_file(),
        embed_dir,
        rerank_dir,
    }
}

#[cfg(feature = "local")]
fn local_embedder_from_env() -> Option<EmbedModel> {
    let dir = resolve_local_model_dir("MEMKEEPER_EMBED_MODEL_DIR", "mxbai-embed-large");
    if !dir.exists() {
        // Loud over silent: a missing model dir is the common first-run state for
        // the semantic release binary. Point at the fix instead of degrading to
        // lexical without a word.
        eprintln!(
            "[memkeeper] semantic embed model not found at {}; run `memkeeper pull-models` \
             to enable semantic search (using lexical BM25 for now)",
            dir.display()
        );
        return None;
    }
    match EmbedModel::load(&dir) {
        Ok(model) => {
            eprintln!("[memkeeper] embed model loaded (dims={})", model.dims());
            Some(model)
        }
        Err(error) => {
            eprintln!("[memkeeper] warning: failed to load embed model: {error}");
            None
        }
    }
}

#[cfg(feature = "local")]
fn local_reranker_from_env() -> Option<RerankerModel> {
    let dir = resolve_local_model_dir("MEMKEEPER_RERANK_MODEL_DIR", "mxbai-rerank-base");
    if !dir.exists() {
        eprintln!(
            "[memkeeper] rerank model not found at {}; run `memkeeper pull-models` \
             to enable reranking (ranking without the cross-encoder for now)",
            dir.display()
        );
        return None;
    }
    match RerankerModel::load(&dir) {
        Ok(model) => {
            eprintln!("[memkeeper] rerank model loaded");
            Some(model)
        }
        Err(error) => {
            eprintln!("[memkeeper] warning: failed to load rerank model: {error}");
            None
        }
    }
}

#[cfg(feature = "api")]
fn api_embedder_from_env() -> Option<Box<dyn Embedder>> {
    let api_key = std::env::var("MEMKEEPER_EMBED_API_KEY").unwrap_or_default();
    if api_key.is_empty() {
        eprintln!("[memkeeper] MEMKEEPER_EMBED_API_KEY not set; API embedding disabled");
        return None;
    }
    let model = std::env::var("MEMKEEPER_EMBED_MODEL")
        .unwrap_or_else(|_| "text-embedding-3-small".to_string());
    let endpoint = std::env::var("MEMKEEPER_EMBED_BASE_URL")
        .unwrap_or_else(|_| "https://api.openai.com/v1/embeddings".to_string());
    let dims = std::env::var("MEMKEEPER_EMBED_DIMS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_EMBED_DIMS);
    Some(Box::new(ApiEmbedder::new(model, dims, endpoint, api_key)) as Box<dyn Embedder>)
}

#[cfg(feature = "local")]
/// Derive a stable model id from a model directory (its final path component).
fn model_id_from_dir(model_dir: &Path) -> String {
    model_dir.file_name().map_or_else(
        || "unknown".to_string(),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// Token-pooling strategy used to collapse the model's last hidden state into a
/// single sentence vector.
///
/// mxbai-embed-large-v1 (the shipping default model) is trained with **CLS**
/// pooling per its model card / sentence-transformers `1_Pooling/config.json`,
/// so [`Pooling::Cls`] is the default. `Mean` (attention-masked mean pooling)
/// remains available for models that expect it.
///
/// NOTE: pooling determines vector geometry, so changing it makes new vectors
/// incompatible with vectors already stored under a different mode. Switching
/// pooling requires re-embedding/reindexing an existing semantic store.
#[cfg(feature = "local")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    /// Use the first ([CLS]) token's hidden state. Correct for mxbai-embed-large-v1.
    Cls,
    /// Attention-masked mean over all token hidden states.
    Mean,
}

#[cfg(feature = "local")]
impl Pooling {
    /// Resolve the pooling mode from `MEMKEEPER_EMBED_POOLING` (`cls`/`mean`),
    /// defaulting to [`Pooling::Cls`]. Unrecognized non-empty values warn and
    /// fall back to the default.
    fn from_env() -> Self {
        match std::env::var("MEMKEEPER_EMBED_POOLING") {
            Ok(value) if value.eq_ignore_ascii_case("mean") => Self::Mean,
            Ok(value) if value.eq_ignore_ascii_case("cls") => Self::Cls,
            Ok(value) if !value.trim().is_empty() => {
                eprintln!(
                    "[memkeeper-embed] warning: unrecognized MEMKEEPER_EMBED_POOLING='{value}'; using cls"
                );
                Self::Cls
            }
            _ => Self::Cls,
        }
    }
}

#[cfg(feature = "local")]
/// ONNX-backed text embedding model (mxbai-embed-large-v1, 1024 dims).
pub struct EmbedModel {
    session: Session,
    tokenizer: Tokenizer,
    dims: usize,
    model_id: String,
    pooling: Pooling,
}

#[cfg(feature = "local")]
impl EmbedModel {
    /// Load model from a directory containing `model.onnx` and `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// Returns an error if the ORT session cannot be created, the model file is missing or
    /// invalid, the tokenizer file is missing or invalid, or the probe embedding fails.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let mut session = Session::builder()
            .context("failed to create ORT session builder")?
            .commit_from_file(model_dir.join("model.onnx"))
            .context("failed to load embedding model.onnx")?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?;
        let pooling = Pooling::from_env();
        let probe = Self::run_session(&mut session, &tokenizer, &["probe"], pooling)?;
        let dims = probe[0].len();
        Ok(Self {
            model_id: model_id_from_dir(model_dir),
            session,
            tokenizer,
            dims,
            pooling,
        })
    }

    /// Return the embedding dimension (1024 for mxbai-embed-large).
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Embed a batch of texts. Returns one unit-normalized vector per input.
    ///
    /// Takes `&mut self` because the underlying ORT session requires exclusive access.
    /// `EmbedModel` cannot be shared across threads via `Arc` without external locking.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or the ONNX inference run fails.
    pub fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Self::run_session(&mut self.session, &self.tokenizer, texts, self.pooling)
    }

    /// Embed a single text.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or the ONNX inference run fails.
    pub fn embed_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let mut result = self.embed(&[text])?;
        result.pop().context("empty embedding result")
    }

    fn run_session(
        session: &mut Session,
        tokenizer: &Tokenizer,
        texts: &[&str],
        pooling: Pooling,
    ) -> Result<Vec<Vec<f32>>> {
        let batch_size = texts.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }

        let encodings: Vec<_> = texts
            .iter()
            .map(|t| {
                tokenizer
                    .encode(*t, true)
                    .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))
            })
            .collect::<Result<_>>()?;

        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(1)
            .min(MAX_SEQ_LEN);

        for enc in &encodings {
            if enc.get_ids().len() > MAX_SEQ_LEN {
                eprintln!(
                    "[memkeeper-embed] warning: input truncated to {MAX_SEQ_LEN} tokens (was {})",
                    enc.get_ids().len()
                );
            }
        }

        let mut input_ids = Array2::<i64>::zeros((batch_size, seq_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, seq_len));
        let token_type_ids = Array2::<i64>::zeros((batch_size, seq_len));

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            for j in 0..seq_len.min(ids.len()) {
                input_ids[[i, j]] = i64::from(ids[j]);
                attention_mask[[i, j]] = i64::from(mask[j]);
            }
        }

        let t_input_ids =
            TensorRef::from_array_view(input_ids.view()).context("input_ids tensor")?;
        let t_attention_mask =
            TensorRef::from_array_view(attention_mask.view()).context("attention_mask tensor")?;
        let t_token_type_ids =
            TensorRef::from_array_view(token_type_ids.view()).context("token_type_ids tensor")?;

        let outputs = session.run(ort::inputs![
            "input_ids" => t_input_ids,
            "attention_mask" => t_attention_mask,
            "token_type_ids" => t_token_type_ids,
        ])?;

        let hidden: ArrayViewD<'_, f32> = outputs[0].try_extract_array::<f32>()?;
        // hidden shape: [batch, seq_len, hidden_dim]
        let hidden_dim = hidden.shape()[2];
        let mut embeddings = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let mut pooled = vec![0f32; hidden_dim];
            match pooling {
                // CLS pooling: the first token's hidden state is the sentence
                // representation (correct for mxbai-embed-large-v1).
                Pooling::Cls => {
                    for k in 0..hidden_dim {
                        pooled[k] = hidden[[i, 0, k]];
                    }
                }
                // Attention-masked mean pooling over all real tokens.
                Pooling::Mean => {
                    let mut mask_sum = 0.0_f32;
                    for j in 0..seq_len {
                        let w = if attention_mask[[i, j]] != 0 {
                            1.0_f32
                        } else {
                            0.0_f32
                        };
                        mask_sum += w;
                        for k in 0..hidden_dim {
                            pooled[k] += hidden[[i, j, k]] * w;
                        }
                    }
                    let denom = mask_sum.max(1e-9_f32);
                    for v in &mut pooled {
                        *v /= denom;
                    }
                }
            }
            let norm: f32 = pooled.iter().map(|x| x * x).sum::<f32>().sqrt();
            let norm = norm.max(1e-9_f32);
            for v in &mut pooled {
                *v /= norm;
            }
            embeddings.push(pooled);
        }
        Ok(embeddings)
    }
}

#[cfg(feature = "local")]
impl Embedder for EmbedModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }
}

#[cfg(feature = "local")]
/// ONNX-backed cross-encoder reranker (mxbai-rerank-base-v1).
pub struct RerankerModel {
    session: Session,
    tokenizer: Tokenizer,
    model_id: String,
}

#[cfg(feature = "local")]
impl RerankerModel {
    /// Load model from a directory containing `model.onnx` and `tokenizer.json`.
    ///
    /// # Errors
    ///
    /// Returns an error if the ORT session cannot be created or the model/tokenizer files are
    /// missing or invalid.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let session = Session::builder()
            .context("failed to create ORT session builder")?
            .commit_from_file(model_dir.join("model.onnx"))
            .context("failed to load reranker model.onnx")?;
        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("failed to load reranker tokenizer: {e}"))?;
        Ok(Self {
            model_id: model_id_from_dir(model_dir),
            session,
            tokenizer,
        })
    }

    /// Score (query, document) pairs. Returns one relevance score \[0,1\] per pair.
    /// Higher score = more relevant. `documents` slice must be non-empty.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or the ONNX inference run fails.
    pub fn rerank(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }
        let batch_size = documents.len();
        let encodings: Vec<_> = documents
            .iter()
            .map(|d| {
                self.tokenizer
                    .encode((query, *d), true)
                    .map_err(|e| anyhow::anyhow!("reranker tokenization failed: {e}"))
            })
            .collect::<Result<_>>()?;

        let seq_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(1)
            .min(MAX_SEQ_LEN);

        for enc in &encodings {
            if enc.get_ids().len() > MAX_SEQ_LEN {
                eprintln!(
                    "[memkeeper-embed] reranker: input truncated to {MAX_SEQ_LEN} tokens (was {})",
                    enc.get_ids().len()
                );
            }
        }

        let mut input_ids = Array2::<i64>::zeros((batch_size, seq_len));
        let mut attention_mask = Array2::<i64>::zeros((batch_size, seq_len));

        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            for j in 0..seq_len.min(ids.len()) {
                input_ids[[i, j]] = i64::from(ids[j]);
                attention_mask[[i, j]] = i64::from(mask[j]);
            }
        }

        let t_input_ids =
            TensorRef::from_array_view(input_ids.view()).context("input_ids tensor")?;
        let t_attention_mask =
            TensorRef::from_array_view(attention_mask.view()).context("attention_mask tensor")?;

        let outputs = self.session.run(ort::inputs![
            "input_ids" => t_input_ids,
            "attention_mask" => t_attention_mask,
        ])?;

        let logits_view: ArrayViewD<'_, f32> = outputs[0].try_extract_array::<f32>()?;
        // Shape may be [batch] (single logit) or [batch, num_labels].
        let scores: Vec<f32> = if logits_view.ndim() == 1 {
            // [batch] — single relevance logit per item
            (0..batch_size)
                .map(|i| {
                    let logit = logits_view[i];
                    1.0_f32 / (1.0_f32 + (-logit).exp())
                })
                .collect()
        } else {
            // [batch, num_labels] — use last column
            let num_labels = logits_view.shape()[1];
            // Assumes positive/relevant class is the last label (index 1 for binary).
            // This matches mxbai-rerank-base-v1's label ordering (0=irrelevant, 1=relevant).
            let label_idx = usize::from(num_labels > 1);
            (0..batch_size)
                .map(|i| {
                    let logit = logits_view[[i, label_idx]];
                    1.0_f32 / (1.0_f32 + (-logit).exp())
                })
                .collect()
        };
        Ok(scores)
    }
}

#[cfg(feature = "local")]
impl Reranker for RerankerModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn rerank(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        self.rerank(query, documents)
    }
}

#[cfg(all(test, feature = "local"))]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::path::PathBuf;

    #[test]
    fn resolve_model_dir_prefers_explicit_then_models_dir_then_discovery_then_home() {
        let temp = tempfile::tempdir().expect("tempdir");
        let discovered_root = temp.path().join("models");
        let discovered_embed = discovered_root.join("mxbai-embed-large");
        std::fs::create_dir_all(&discovered_embed).expect("model dir");
        std::fs::write(discovered_embed.join("model.onnx"), b"test").expect("model marker");
        let candidate_roots = [discovered_root.clone()];

        // Explicit dir wins outright.
        assert_eq!(
            resolve_model_dir(
                Some(OsString::from("/x/embed")),
                Some(OsString::from("/ignored")),
                Some(OsString::from("/ignored")),
                &candidate_roots,
                "mxbai-embed-large",
            ),
            PathBuf::from("/x/embed"),
        );
        // No explicit dir -> $MEMKEEPER_MODELS_DIR/<subdir> (must match pull-models).
        assert_eq!(
            resolve_model_dir(
                None,
                Some(OsString::from("/models")),
                Some(OsString::from("/home/u")),
                &candidate_roots,
                "mxbai-embed-large",
            ),
            PathBuf::from("/models/mxbai-embed-large"),
        );
        // No explicit, no models dir -> nearby checked-in model dir when present.
        assert_eq!(
            resolve_model_dir(
                None,
                None,
                Some(OsString::from("/home/u")),
                &candidate_roots,
                "mxbai-embed-large",
            ),
            discovered_embed,
        );
        // Discovery requires a model file; otherwise fall back to home.
        assert_eq!(
            resolve_model_dir(
                None,
                None,
                Some(OsString::from("/home/u")),
                &candidate_roots,
                "mxbai-rerank-base"
            ),
            PathBuf::from("/home/u/.memkeeper/models/mxbai-rerank-base"),
        );
    }

    fn models_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap() // crates/
            .parent()
            .unwrap() // memory/memkeeper/
            .join("models")
    }

    #[test]
    #[ignore = "requires ONNX model files; run with -- --ignored"]
    fn embed_model_produces_unit_vectors() {
        let mut model =
            EmbedModel::load(&models_dir().join("mxbai-embed-large")).expect("load embed model");
        assert_eq!(model.dims(), 1024);
        let vecs = model
            .embed(&["hello world", "a quick brown fox"])
            .expect("embed");
        assert_eq!(vecs.len(), 2);
        for v in &vecs {
            assert_eq!(v.len(), 1024);
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "vector not unit-normalized, norm={norm}"
            );
        }
    }

    #[test]
    #[ignore = "requires ONNX model files; run with -- --ignored"]
    fn similar_texts_have_higher_cosine_similarity() {
        let mut model =
            EmbedModel::load(&models_dir().join("mxbai-embed-large")).expect("load embed model");
        let vecs = model
            .embed(&[
                "The cat sat on the mat",
                "A cat was sitting on a mat",
                "The stock market crashed today",
            ])
            .expect("embed");
        let cos = |a: &[f32], b: &[f32]| -> f32 { a.iter().zip(b).map(|(x, y)| x * y).sum() };
        let sim_related = cos(&vecs[0], &vecs[1]);
        let sim_unrelated = cos(&vecs[0], &vecs[2]);
        assert!(
            sim_related > sim_unrelated,
            "related texts should be closer: {sim_related:.3} vs {sim_unrelated:.3}"
        );
    }

    #[test]
    #[ignore = "requires ONNX model files; run with -- --ignored"]
    fn reranker_scores_relevant_higher() {
        let mut model = RerankerModel::load(&models_dir().join("mxbai-rerank-base"))
            .expect("load rerank model");
        let query = "What is the capital of France?";
        let docs = ["Paris is the capital of France.", "The sky is blue."];
        let scores = model.rerank(query, &docs).expect("rerank");
        assert_eq!(scores.len(), 2);
        assert!(
            scores[0] > scores[1],
            "relevant doc should score higher: {scores:?}"
        );
    }
}
