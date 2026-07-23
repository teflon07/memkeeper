//! `ColBERT` late-interaction token encoder (answerai-colbert-small-v1 via merged ONNX).
//!
//! Encoding conventions come from `colbert_config.json` written by
//! `scripts/experiments/export_colbert_onnx.py`; the pylate-generated
//! `parity_fixtures.json` in the same directory is the behavioral contract
//! (see tests). The validated recipe: insert the prefix token id at position 1
//! (after `[CLS]`), MASK-pad queries to `query_length` with attention 0 on the
//! expansion tokens, truncate docs at `document_length`, and drop skiplist
//! (punctuation) token positions from doc outputs only.
use anyhow::Result;
use std::path::Path;

/// Per-token embedder for late-interaction retrieval.
pub trait TokenEmbedder: Send {
    /// Stable model identifier stored alongside token embeddings.
    fn model_id(&self) -> &str;
    /// Token-vector dimensionality (96 for answerai-colbert-small-v1).
    fn dims(&self) -> usize;
    /// Encode a query into a fixed-length token matrix (`query_length` x dims).
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or the ONNX inference run fails.
    fn encode_query(&mut self, text: &str) -> Result<Vec<Vec<f32>>>;
    /// Encode documents: one variable-length token matrix per doc.
    ///
    /// # Errors
    ///
    /// Returns an error if tokenization or the ONNX inference run fails.
    fn encode_docs(&mut self, texts: &[&str]) -> Result<Vec<Vec<Vec<f32>>>>;
}

#[cfg(feature = "local")]
mod local {
    use super::TokenEmbedder;
    use anyhow::{Context, Result};
    use ndarray::Array2;
    use ort::session::Session;
    use ort::value::TensorRef;
    use std::collections::BTreeSet;
    use std::path::Path;
    use tokenizers::Tokenizer;

    #[derive(serde::Deserialize)]
    struct ColBertConfig {
        model_id: String,
        dims: usize,
        query_length: usize,
        document_length: usize,
        query_prefix_id: u32,
        document_prefix_id: u32,
        mask_token_id: u32,
        attend_to_expansion_tokens: bool,
        #[serde(default)]
        skiplist_words: Vec<String>,
    }

    /// ONNX-backed `ColBERT` token encoder.
    pub struct ColBertModel {
        session: Session,
        tokenizer: Tokenizer,
        config: ColBertConfig,
        skiplist_ids: BTreeSet<u32>,
    }

    impl ColBertModel {
        /// Load from a dir containing model.onnx, tokenizer.json, `colbert_config.json`.
        ///
        /// # Errors
        ///
        /// Returns an error if the ORT session, tokenizer, or config cannot be loaded.
        pub fn load(model_dir: &Path) -> Result<Self> {
            let session = Session::builder()
                .context("failed to create ORT session builder")?
                .commit_from_file(model_dir.join("model.onnx"))
                .context("failed to load colbert model.onnx")?;
            let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .map_err(|e| anyhow::anyhow!("failed to load colbert tokenizer: {e}"))?;
            let config: ColBertConfig = serde_json::from_str(
                &std::fs::read_to_string(model_dir.join("colbert_config.json"))
                    .context("failed to read colbert_config.json")?,
            )
            .context("failed to parse colbert_config.json")?;
            let skiplist_ids = config
                .skiplist_words
                .iter()
                .filter_map(|w| tokenizer.token_to_id(w))
                .collect();
            Ok(Self {
                session,
                tokenizer,
                config,
                skiplist_ids,
            })
        }

        /// Tokenize with specials, then insert the prefix token id at position 1
        /// (after `[CLS]`), capping total length.
        fn tokenize_with_prefix(&self, text: &str, prefix_id: u32, cap: usize) -> Result<Vec<u32>> {
            let enc = self
                .tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("colbert tokenize failed: {e}"))?;
            let raw = enc.get_ids();
            let mut ids = Vec::with_capacity((raw.len() + 1).min(cap));
            ids.push(*raw.first().unwrap_or(&prefix_id));
            ids.push(prefix_id);
            ids.extend_from_slice(&raw[1.min(raw.len())..]);
            ids.truncate(cap);
            Ok(ids)
        }

        fn run(&mut self, ids: &[Vec<u32>], masks: &[Vec<u32>]) -> Result<Vec<Vec<Vec<f32>>>> {
            let batch = ids.len();
            let seq = ids.iter().map(Vec::len).max().unwrap_or(1);
            let mut input_ids = Array2::<i64>::zeros((batch, seq));
            let mut attention = Array2::<i64>::zeros((batch, seq));
            for (i, (row_ids, row_mask)) in ids.iter().zip(masks).enumerate() {
                for (j, (&id, &mask)) in row_ids.iter().zip(row_mask).enumerate() {
                    input_ids[[i, j]] = i64::from(id);
                    attention[[i, j]] = i64::from(mask);
                }
            }
            let t_ids = TensorRef::from_array_view(input_ids.view()).context("input_ids tensor")?;
            let t_mask =
                TensorRef::from_array_view(attention.view()).context("attention_mask tensor")?;
            let outputs = self.session.run(ort::inputs![
                "input_ids" => t_ids,
                "attention_mask" => t_mask,
            ])?;
            let view = outputs[0].try_extract_array::<f32>()?; // [batch, seq, dims]
            let dims = self.config.dims;
            let mut result = Vec::with_capacity(batch);
            for i in 0..batch {
                let n = ids[i].len();
                let mut doc = Vec::with_capacity(n);
                for j in 0..n {
                    doc.push((0..dims).map(|d| view[[i, j, d]]).collect());
                }
                result.push(doc);
            }
            Ok(result)
        }
    }

    impl TokenEmbedder for ColBertModel {
        fn model_id(&self) -> &str {
            &self.config.model_id
        }

        fn dims(&self) -> usize {
            self.config.dims
        }

        fn encode_query(&mut self, text: &str) -> Result<Vec<Vec<f32>>> {
            let started = std::time::Instant::now();
            let mut ids = self.tokenize_with_prefix(
                text,
                self.config.query_prefix_id,
                self.config.query_length,
            )?;
            let tokens = ids.len();
            let mut mask = vec![1u32; ids.len()];
            let attend = u32::from(self.config.attend_to_expansion_tokens);
            while ids.len() < self.config.query_length {
                ids.push(self.config.mask_token_id);
                mask.push(attend);
            }
            let result = self
                .run(&[ids], &[mask])
                .map(|mut encoded| encoded.pop().unwrap_or_default());
            super::super::metrics::record(
                &self.config.model_id,
                "colbert_query",
                1,
                tokens,
                u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                result.is_ok(),
            );
            result
        }

        fn encode_docs(&mut self, texts: &[&str]) -> Result<Vec<Vec<Vec<f32>>>> {
            if texts.is_empty() {
                return Ok(vec![]);
            }
            let started = std::time::Instant::now();
            let mut all_ids = Vec::with_capacity(texts.len());
            let mut all_masks = Vec::with_capacity(texts.len());
            for text in texts {
                let ids = self.tokenize_with_prefix(
                    text,
                    self.config.document_prefix_id,
                    self.config.document_length,
                )?;
                all_masks.push(vec![1u32; ids.len()]);
                all_ids.push(ids);
            }
            let tokens: usize = all_ids.iter().map(Vec::len).sum();
            let mut encoded = self.run(&all_ids, &all_masks)?;
            // pylate drops skiplist (punctuation) tokens from DOC outputs only.
            for (doc, ids) in encoded.iter_mut().zip(&all_ids) {
                let mut keep = ids.iter().map(|id| !self.skiplist_ids.contains(id));
                doc.retain(|_| keep.next().unwrap_or(true));
            }
            super::super::metrics::record(
                &self.config.model_id,
                "colbert_docs",
                texts.len(),
                tokens,
                u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
                true,
            );
            Ok(encoded)
        }
    }
}

#[cfg(feature = "local")]
pub use local::ColBertModel;

/// Construct the `ColBERT` token embedder from `MEMKEEPER_COLBERT_MODEL_DIR`,
/// or `None` (with a stderr warning on failure) when unset or unavailable
/// in this build.
#[must_use]
pub fn colbert_from_env() -> Option<Box<dyn TokenEmbedder>> {
    let dir = std::env::var("MEMKEEPER_COLBERT_MODEL_DIR").ok()?;
    #[cfg(feature = "local")]
    {
        match ColBertModel::load(Path::new(&dir)) {
            Ok(model) => Some(Box::new(model)),
            Err(e) => {
                eprintln!("[memkeeper] colbert model load failed ({dir}): {e}");
                None
            }
        }
    }
    #[cfg(not(feature = "local"))]
    {
        let _ = Path::new(&dir);
        eprintln!(
            "[memkeeper] MEMKEEPER_COLBERT_MODEL_DIR set but this build lacks the local feature"
        );
        None
    }
}

#[cfg(all(test, feature = "local"))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn model_dir() -> Option<PathBuf> {
        // Parity tests need the exported model; skip cleanly when absent.
        let dir = std::env::var("MEMKEEPER_COLBERT_MODEL_DIR")
            .ok()
            .map(PathBuf::from)
            .or_else(|| {
                let d =
                    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/colbert-small");
                d.exists().then_some(d)
            })?;
        dir.join("parity_fixtures.json").exists().then_some(dir)
    }

    fn fixtures(dir: &Path) -> serde_json::Value {
        serde_json::from_str(
            &std::fs::read_to_string(dir.join("parity_fixtures.json")).expect("read fixtures"),
        )
        .expect("parse fixtures")
    }

    #[test]
    fn query_encoding_matches_pylate_fixtures() {
        let Some(dir) = model_dir() else {
            eprintln!("skipping: no colbert model dir");
            return;
        };
        let mut model = ColBertModel::load(&dir).expect("load");
        for case in fixtures(&dir)["queries"].as_array().expect("queries") {
            let text = case["text"].as_str().expect("text");
            let gold: Vec<Vec<f32>> =
                serde_json::from_value(case["embedding"].clone()).expect("gold");
            let ours = model.encode_query(text).expect("encode_query");
            assert_eq!(ours.len(), gold.len(), "query token count for {text:?}");
            for (i, (a, b)) in ours.iter().zip(&gold).enumerate() {
                let cos: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                assert!(cos > 0.999, "query token {i} cosine {cos} for {text:?}");
            }
        }
    }

    #[test]
    fn doc_encoding_matches_pylate_fixtures() {
        let Some(dir) = model_dir() else {
            eprintln!("skipping: no colbert model dir");
            return;
        };
        let mut model = ColBertModel::load(&dir).expect("load");
        for case in fixtures(&dir)["docs"].as_array().expect("docs") {
            let text = case["text"].as_str().expect("text");
            let gold: Vec<Vec<f32>> =
                serde_json::from_value(case["embedding"].clone()).expect("gold");
            let ours = model
                .encode_docs(&[text])
                .expect("encode_docs")
                .pop()
                .expect("doc");
            assert_eq!(
                ours.len(),
                gold.len(),
                "doc token count for {text:?} (skiplist?)"
            );
            for (i, (a, b)) in ours.iter().zip(&gold).enumerate() {
                let cos: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
                assert!(cos > 0.999, "doc token {i} cosine {cos} for {text:?}");
            }
        }
    }
}
