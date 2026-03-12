use std::collections::HashMap;
use std::io::Read;
use std::sync::Arc;
use std::sync::Mutex;

use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(crate) struct ClientUsageStats {
    pub requests: u64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    pub reasoning_output_tokens: i64,
    pub total_tokens: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub(crate) struct UsageStatsSnapshot {
    pub clients: HashMap<String, ClientUsageStats>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct UsageStatsStore {
    inner: Arc<Mutex<HashMap<String, ClientUsageStats>>>,
}

impl UsageStatsStore {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    pub(crate) fn record_request(&self, client_id: &str) {
        if let Ok(mut stats) = self.inner.lock() {
            let stats = stats.entry(client_id.to_string()).or_default();
            stats.requests += 1;
        }
    }

    pub(crate) fn record_usage(&self, client_id: &str, usage: TokenUsage) {
        if let Ok(mut stats) = self.inner.lock() {
            let stats = stats.entry(client_id.to_string()).or_default();
            stats.input_tokens += usage.input_tokens;
            stats.output_tokens += usage.output_tokens;
            stats.cached_input_tokens += usage.cached_input_tokens;
            stats.reasoning_output_tokens += usage.reasoning_output_tokens;
            stats.total_tokens += usage.total_tokens;
        }
    }

    pub(crate) fn snapshot(&self) -> UsageStatsSnapshot {
        match self.inner.lock() {
            Ok(stats) => UsageStatsSnapshot {
                clients: stats.clone(),
            },
            Err(_) => UsageStatsSnapshot::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct TokenUsage {
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cached_input_tokens: i64,
    pub reasoning_output_tokens: i64,
    pub total_tokens: i64,
}

#[derive(Debug)]
pub(crate) struct TrackingBodyReader<R: Read> {
    inner: R,
    body: Vec<u8>,
    client_id: Option<String>,
    stats_store: UsageStatsStore,
    finished: bool,
}

impl<R: Read> TrackingBodyReader<R> {
    pub(crate) fn new(inner: R, client_id: Option<String>, stats_store: UsageStatsStore) -> Self {
        Self {
            inner,
            body: Vec::new(),
            client_id,
            stats_store,
            finished: false,
        }
    }

    fn finalize_usage_if_needed(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;

        let Some(client_id) = self.client_id.as_deref() else {
            return;
        };
        let Ok(text) = std::str::from_utf8(&self.body) else {
            return;
        };
        if let Some(usage) = extract_usage(text) {
            self.stats_store.record_usage(client_id, usage);
        }
    }
}

impl<R: Read> Read for TrackingBodyReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let read = self.inner.read(buf)?;
        if read == 0 {
            self.finalize_usage_if_needed();
            return Ok(0);
        }

        self.body.extend_from_slice(&buf[..read]);
        Ok(read)
    }
}

impl<R: Read> Drop for TrackingBodyReader<R> {
    fn drop(&mut self) {
        self.finalize_usage_if_needed();
    }
}

fn extract_usage(payload: &str) -> Option<TokenUsage> {
    extract_usage_from_json(payload).or_else(|| extract_usage_from_sse(payload))
}

fn extract_usage_from_json(payload: &str) -> Option<TokenUsage> {
    let value: Value = serde_json::from_str(payload).ok()?;
    extract_usage_from_json_value(&value)
}

fn extract_usage_from_sse(payload: &str) -> Option<TokenUsage> {
    let mut usage = None;
    for event in payload.split("\n\n") {
        for data in event
            .lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(str::trim)
        {
            let Ok(value) = serde_json::from_str::<Value>(data) else {
                continue;
            };
            if value.get("type").and_then(Value::as_str) != Some("response.completed") {
                continue;
            }
            if let Some(event_usage) = value
                .get("response")
                .and_then(extract_usage_from_json_value)
                .or_else(|| extract_usage_from_json_value(&value))
            {
                usage = Some(event_usage);
            }
        }
    }
    usage
}

fn extract_usage_from_json_value(value: &Value) -> Option<TokenUsage> {
    let usage = value.get("usage")?;
    let input_tokens = usage.get("input_tokens").and_then(Value::as_i64)?;
    let output_tokens = usage.get("output_tokens").and_then(Value::as_i64)?;
    let total_tokens = usage.get("total_tokens").and_then(Value::as_i64)?;
    let cached_input_tokens = usage
        .get("input_tokens_details")
        .and_then(|details| details.get("cached_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let reasoning_output_tokens = usage
        .get("output_tokens_details")
        .and_then(|details| details.get("reasoning_tokens"))
        .and_then(Value::as_i64)
        .unwrap_or(0);

    Some(TokenUsage {
        input_tokens,
        output_tokens,
        cached_input_tokens,
        reasoning_output_tokens,
        total_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_usage_from_json() {
        let usage = extract_usage(
            r#"{"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15,"input_tokens_details":{"cached_tokens":2},"output_tokens_details":{"reasoning_tokens":1}}}"#,
        );

        assert_eq!(
            usage,
            Some(TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
                cached_input_tokens: 2,
                reasoning_output_tokens: 1,
                total_tokens: 15,
            })
        );
    }

    #[test]
    fn extracts_usage_from_sse() {
        let usage = extract_usage(
            "event: response.completed\n\
             data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":8,\"output_tokens\":3,\"total_tokens\":11}}}\n\n",
        );

        assert_eq!(
            usage,
            Some(TokenUsage {
                input_tokens: 8,
                output_tokens: 3,
                cached_input_tokens: 0,
                reasoning_output_tokens: 0,
                total_tokens: 11,
            })
        );
    }

    #[test]
    fn tracks_usage_for_multiple_clients() {
        let store = UsageStatsStore::new();
        store.record_request("client-a");
        store.record_request("client-b");

        let mut sink = Vec::new();
        let mut reader_a = TrackingBodyReader::new(
            std::io::Cursor::new(
                "{\"usage\":{\"input_tokens\":5,\"output_tokens\":2,\"total_tokens\":7}}",
            ),
            Some("client-a".to_string()),
            store.clone(),
        );
        std::io::copy(&mut reader_a, &mut sink).expect("copy A");

        sink.clear();
        let mut reader_b = TrackingBodyReader::new(
            std::io::Cursor::new(
                "{\"usage\":{\"input_tokens\":11,\"output_tokens\":5,\"total_tokens\":16}}",
            ),
            Some("client-b".to_string()),
            store.clone(),
        );
        std::io::copy(&mut reader_b, &mut sink).expect("copy B");

        let snapshot = store.snapshot();
        assert_eq!(
            snapshot.clients.get("client-a").map(|value| value.requests),
            Some(1)
        );
        assert_eq!(
            snapshot
                .clients
                .get("client-a")
                .map(|value| value.input_tokens),
            Some(5)
        );
        assert_eq!(
            snapshot.clients.get("client-b").map(|value| value.requests),
            Some(1)
        );
        assert_eq!(
            snapshot
                .clients
                .get("client-b")
                .map(|value| value.input_tokens),
            Some(11)
        );
    }
}
