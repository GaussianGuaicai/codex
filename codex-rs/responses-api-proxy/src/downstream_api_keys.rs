use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use serde::Deserialize;

#[derive(Debug, Clone)]
pub(crate) struct DownstreamApiKeyRegistry {
    client_id_by_key: HashMap<String, String>,
}

impl DownstreamApiKeyRegistry {
    pub(crate) fn from_file(path: &Path) -> Result<Self> {
        let raw = fs::read_to_string(path).with_context(|| {
            format!("reading downstream API key config from {}", path.display())
        })?;
        let entries: DownstreamApiKeyConfig = serde_json::from_str(&raw).with_context(|| {
            format!("parsing downstream API key config from {}", path.display())
        })?;

        let mut client_id_by_key = HashMap::new();
        for entry in entries.keys {
            if entry.client_id.trim().is_empty() {
                return Err(anyhow!("downstream key entry has empty client_id"));
            }
            if entry.api_key.trim().is_empty() {
                return Err(anyhow!("downstream key entry has empty api_key"));
            }
            if let Some(previous_client_id) =
                client_id_by_key.insert(entry.api_key, entry.client_id)
            {
                return Err(anyhow!(
                    "duplicate downstream api_key found for client {previous_client_id}"
                ));
            }
        }

        Ok(Self { client_id_by_key })
    }

    pub(crate) fn lookup_client_id(&self, api_key: &str) -> Option<&str> {
        self.client_id_by_key.get(api_key).map(String::as_str)
    }
}

#[derive(Debug, Clone, Deserialize)]
struct DownstreamApiKeyConfig {
    keys: Vec<DownstreamApiKeyEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct DownstreamApiKeyEntry {
    client_id: String,
    api_key: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_registry_from_json() {
        let parsed: DownstreamApiKeyConfig = serde_json::from_str(
            r#"{
            "keys": [
                {"client_id": "client-a", "api_key": "dsk_a"},
                {"client_id": "client-b", "api_key": "dsk_b"}
            ]
        }"#,
        )
        .expect("should parse");

        assert_eq!(
            parsed
                .keys
                .into_iter()
                .map(|entry| (entry.client_id, entry.api_key))
                .collect::<Vec<_>>(),
            vec![
                ("client-a".to_string(), "dsk_a".to_string()),
                ("client-b".to_string(), "dsk_b".to_string())
            ]
        );
    }

    #[test]
    fn lookup_client_id_returns_match() {
        let registry = DownstreamApiKeyRegistry {
            client_id_by_key: [("dsk_abc".to_string(), "client-123".to_string())]
                .into_iter()
                .collect(),
        };

        assert_eq!(registry.lookup_client_id("dsk_abc"), Some("client-123"));
        assert_eq!(registry.lookup_client_id("missing"), None);
    }

    #[test]
    fn rejects_duplicate_api_keys() {
        let path = std::env::temp_dir().join(format!(
            "codex-responses-api-proxy-duplicate-keys-{}.json",
            std::process::id()
        ));
        fs::write(
            &path,
            r#"{"keys":[{"client_id":"client-a","api_key":"dsk_same"},{"client_id":"client-b","api_key":"dsk_same"}]}"#,
        )
        .expect("should write downstream key file");

        let error =
            DownstreamApiKeyRegistry::from_file(&path).expect_err("duplicate keys should fail");
        let _ = fs::remove_file(path);

        assert_eq!(
            error.to_string().contains("duplicate downstream api_key"),
            true
        );
    }
}
