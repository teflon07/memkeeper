use serde::{Deserialize, Serialize};
use serde_json::Value;
#[cfg(not(test))]
use std::sync::{mpsc, OnceLock};
use std::{fs, path::PathBuf};
#[cfg(not(test))]
use std::{
    thread,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct Event {
    ts: u64,
    model: String,
    operation: String,
    items: usize,
    tokens: usize,
    latency_ms: u64,
    ok: bool,
}

#[cfg(not(test))]
static SENDER: OnceLock<mpsc::Sender<Event>> = OnceLock::new();

#[cfg(not(test))]
pub(crate) fn record(
    model: &str,
    operation: &str,
    items: usize,
    tokens: usize,
    latency_ms: u64,
    ok: bool,
) {
    let sender = SENDER.get_or_init(|| {
        let (sender, receiver) = mpsc::channel::<Event>();
        thread::spawn(move || write_events(receiver));
        sender
    });
    let _ = sender.send(Event {
        ts: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model: model.to_string(),
        operation: operation.to_string(),
        items,
        tokens,
        latency_ms,
        ok,
    });
}

#[cfg(test)]
pub(crate) fn record(
    _model: &str,
    _operation: &str,
    _items: usize,
    _tokens: usize,
    _latency_ms: u64,
    _ok: bool,
) {
}

fn metrics_dir() -> PathBuf {
    std::env::var_os("MEMKEEPER_LOCAL_USAGE_DIR").map_or_else(
        || {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
                .join(".local/share/memkeeper/local-inference")
        },
        PathBuf::from,
    )
}

#[cfg(not(test))]
fn write_events(receiver: mpsc::Receiver<Event>) {
    let dir = metrics_dir();
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = dir.join(format!("{}.jsonl", std::process::id()));
    let Ok(mut file) = fs::OpenOptions::new().create(true).append(true).open(path) else {
        return;
    };
    for event in receiver {
        if serde_json::to_writer(&mut file, &event).is_err()
            || std::io::Write::write_all(&mut file, b"\n").is_err()
        {
            return;
        }
        let _ = std::io::Write::flush(&mut file);
    }
}

/// Read and aggregate the local model usage ledgers.
#[must_use]
pub fn report() -> Value {
    let dir = metrics_dir();
    let mut events = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|ext| ext == "jsonl") {
                if let Ok(text) = fs::read_to_string(entry.path()) {
                    events.extend(
                        text.lines()
                            .filter_map(|line| serde_json::from_str::<Event>(line).ok()),
                    );
                }
            }
        }
    }
    serde_json::json!({"events": events.len(), "by_model": aggregate(events.iter()), "path": metrics_dir()})
}

fn aggregate<'a>(events: impl IntoIterator<Item = &'a Event>) -> Vec<Value> {
    use std::collections::BTreeMap;
    let mut totals: BTreeMap<(String, String), [u64; 6]> = BTreeMap::new();
    for event in events {
        let entry = totals
            .entry((event.model.clone(), event.operation.clone()))
            .or_default();
        entry[0] += 1;
        entry[1] += event.items as u64;
        entry[2] += event.tokens as u64;
        entry[3] += event.latency_ms;
        entry[4] += u64::from(!event.ok);
        entry[5] += u64::from(event.ok);
    }
    totals
        .into_iter()
        .map(
            |((model, operation), [calls, items, tokens, latency_ms, failures, successes])| {
                serde_json::json!({
                    "model": model,
                    "operation": operation,
                    "calls": calls,
                    "items": items,
                    "tokens": tokens,
                    "latency_ms": latency_ms,
                    "failures": failures,
                    "successes": successes,
                })
            },
        )
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_local_usage_events_by_model_and_operation() {
        let events = [
            Event {
                ts: 0,
                model: "mxbai-embed-large".into(),
                operation: "embed".into(),
                items: 2,
                tokens: 18,
                latency_ms: 4,
                ok: true,
            },
            Event {
                ts: 0,
                model: "mxbai-embed-large".into(),
                operation: "embed".into(),
                items: 1,
                tokens: 7,
                latency_ms: 3,
                ok: false,
            },
        ];
        let report = aggregate(events.iter());
        assert_eq!(report[0]["calls"], 2);
        assert_eq!(report[0]["items"], 3);
        assert_eq!(report[0]["tokens"], 25);
        assert_eq!(report[0]["failures"], 1);
    }
}
