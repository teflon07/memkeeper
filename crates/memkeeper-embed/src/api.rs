//! Remote HTTP embedding provider (`OpenAI`-compatible).
//!
//! This implements [`super::Embedder`] without the ONNX runtime, so an
//! API-only build carries no local model dependencies. Any `OpenAI`-compatible
//! endpoint (e.g. `OpenRouter`) works via a custom base URL.

use anyhow::{anyhow, Context, Result};
use serde::Deserialize;

use super::{Embedder, Reranker};

/// Remote embedding provider over HTTP. One model, one endpoint.
pub struct ApiEmbedder {
    model_id: String,
    model: String,
    dims: usize,
    endpoint: String,
    api_key: String,
    agent: ureq::Agent,
}

impl ApiEmbedder {
    /// Build a remote embedder.
    ///
    /// `endpoint` is the full embeddings URL (e.g.
    /// `https://api.openai.com/v1/embeddings`). `dims` is the expected output
    /// dimension; for `OpenAI` it is also requested via the `dimensions`
    /// parameter (Matryoshka truncation).
    #[must_use]
    pub fn new(model: String, dims: usize, endpoint: String, api_key: String) -> Self {
        let model_id = format!("openai:{model}@{dims}");
        // Bounded timeouts so a stuck embeddings API cannot hang a long-lived
        // host (the serve daemon processes requests on one loop).
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(30))
            .build();
        Self {
            model_id,
            model,
            dims,
            endpoint,
            api_key,
            agent,
        }
    }

    fn request_body(&self, texts: &[&str]) -> String {
        serde_json::json!({
            "model": self.model,
            "input": texts,
            "dimensions": self.dims,
        })
        .to_string()
    }
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingItem>,
}

#[derive(Deserialize)]
struct EmbeddingItem {
    embedding: Vec<f32>,
    #[serde(default)]
    index: usize,
}

impl Embedder for ApiEmbedder {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dims(&self) -> usize {
        self.dims
    }

    fn embed(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let body = self.request_body(texts);
        let response = self
            .agent
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_string(&body)
            .map_err(|error| anyhow!("embedding API request failed: {error}"))?;
        let text = response
            .into_string()
            .context("reading embedding API response body")?;
        let parsed: EmbeddingResponse =
            serde_json::from_str(&text).context("parsing embedding API response")?;
        if parsed.data.len() != texts.len() {
            return Err(anyhow!(
                "embedding API returned {} vectors for {} inputs",
                parsed.data.len(),
                texts.len()
            ));
        }
        let mut items = parsed.data;
        items.sort_by_key(|item| item.index);
        let mut out = Vec::with_capacity(items.len());
        for item in items {
            if item.embedding.len() != self.dims {
                return Err(anyhow!(
                    "embedding API returned dimension {} but {} was configured",
                    item.embedding.len(),
                    self.dims
                ));
            }
            out.push(item.embedding);
        }
        Ok(out)
    }
}

/// Remote cross-encoder reranker over HTTP, speaking the Cohere `/rerank`
/// shape (`{model, query, documents}` -> `{results:[{index, relevance_score}]}`).
/// Covers Cohere's native API and any compatible proxy (e.g. `OpenRouter`) via a
/// custom base URL. No ONNX runtime, so it works in an API-only build.
pub struct ApiReranker {
    model_id: String,
    model: String,
    endpoint: String,
    api_key: String,
    agent: ureq::Agent,
}

impl ApiReranker {
    /// Build a remote reranker. `endpoint` is the full rerank URL (e.g.
    /// `https://openrouter.ai/api/v1/rerank`).
    #[must_use]
    pub fn new(model: String, endpoint: String, api_key: String) -> Self {
        let model_id = format!("rerank-api:{model}");
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(30))
            .build();
        Self {
            model_id,
            model,
            endpoint,
            api_key,
            agent,
        }
    }

    fn request_body(&self, query: &str, documents: &[&str]) -> String {
        serde_json::json!({
            "model": self.model,
            "query": query,
            "documents": documents,
        })
        .to_string()
    }
}

#[derive(Deserialize)]
struct RerankResponse {
    results: Vec<RerankResult>,
}

#[derive(Deserialize)]
struct RerankResult {
    index: usize,
    relevance_score: f32,
}

impl Reranker for ApiReranker {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn rerank(&mut self, query: &str, documents: &[&str]) -> Result<Vec<f32>> {
        if documents.is_empty() {
            return Ok(Vec::new());
        }
        let body = self.request_body(query, documents);
        let response = self
            .agent
            .post(&self.endpoint)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .set("Content-Type", "application/json")
            .send_string(&body)
            .map_err(|error| anyhow!("rerank API request failed: {error}"))?;
        let text = response
            .into_string()
            .context("reading rerank API response body")?;
        let parsed: RerankResponse =
            serde_json::from_str(&text).context("parsing rerank API response")?;
        // The API returns results in relevance order; map each score back onto
        // its input position so the caller gets exactly `documents.len()` scores
        // aligned to input order.
        Self::scores_in_input_order(parsed.results, documents.len())
    }
}

impl ApiReranker {
    /// Reconstruct input-order relevance scores from the (relevance-sorted) API
    /// results. Positions absent from `results` keep `0.0`. Errors on an index
    /// outside the input range.
    fn scores_in_input_order(results: Vec<RerankResult>, n: usize) -> Result<Vec<f32>> {
        let mut scores = vec![0.0_f32; n];
        for result in results {
            if result.index >= n {
                return Err(anyhow!(
                    "rerank API returned out-of-range index {} for {n} documents",
                    result.index
                ));
            }
            scores[result.index] = result.relevance_score;
        }
        Ok(scores)
    }
}

#[cfg(test)]
mod tests {
    use super::ApiEmbedder;

    #[test]
    fn model_id_encodes_provider_model_and_dims() {
        let embedder = ApiEmbedder::new(
            "text-embedding-3-small".to_string(),
            1024,
            "https://api.openai.com/v1/embeddings".to_string(),
            "sk-test".to_string(),
        );
        assert_eq!(embedder.model_id, "openai:text-embedding-3-small@1024");
    }

    #[test]
    fn openai_request_body_includes_dimensions() {
        let embedder = ApiEmbedder::new(
            "text-embedding-3-small".to_string(),
            512,
            "https://api.openai.com/v1/embeddings".to_string(),
            "sk-test".to_string(),
        );
        let body = embedder.request_body(&["hello"]);
        let value: serde_json::Value = serde_json::from_str(&body).expect("valid json");
        assert_eq!(value["dimensions"], 512);
        assert_eq!(value["model"], "text-embedding-3-small");
        assert_eq!(value["input"][0], "hello");
    }

    #[test]
    fn rerank_request_body_shape() {
        let reranker = super::ApiReranker::new(
            "cohere/rerank-v3.5".to_string(),
            "https://openrouter.ai/api/v1/rerank".to_string(),
            "key".to_string(),
        );
        let body = reranker.request_body("rollback policy", &["doc a", "doc b"]);
        let value: serde_json::Value = serde_json::from_str(&body).expect("valid json");
        assert_eq!(value["model"], "cohere/rerank-v3.5");
        assert_eq!(value["query"], "rollback policy");
        assert_eq!(value["documents"][1], "doc b");
    }

    #[test]
    fn rerank_scores_map_back_to_input_order() {
        // Results come back relevance-sorted (input #2 is best); the scores must
        // be returned in input order, not relevance order.
        let json = r#"{"results":[{"index":2,"relevance_score":0.9},{"index":0,"relevance_score":0.5},{"index":1,"relevance_score":0.1}]}"#;
        let parsed: super::RerankResponse = serde_json::from_str(json).expect("valid json");
        let scores = super::ApiReranker::scores_in_input_order(parsed.results, 3).expect("ok");
        assert_eq!(scores, vec![0.5_f32, 0.1, 0.9]);
    }

    #[test]
    fn rerank_rejects_out_of_range_index() {
        let json = r#"{"results":[{"index":5,"relevance_score":0.9}]}"#;
        let parsed: super::RerankResponse = serde_json::from_str(json).expect("valid json");
        assert!(super::ApiReranker::scores_in_input_order(parsed.results, 2).is_err());
    }
}
