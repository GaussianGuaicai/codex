use std::fs::File;
use std::fs::{self};
use std::io::Write;
use std::net::SocketAddr;
use std::net::TcpListener;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use clap::Parser;
use rand::TryRngCore;
use rand::rngs::OsRng;
use reqwest::Url;
use reqwest::blocking::Client;
use reqwest::header::AUTHORIZATION;
use reqwest::header::HOST;
use reqwest::header::HeaderMap;
use reqwest::header::HeaderName;
use reqwest::header::HeaderValue;
use serde::Serialize;
use tiny_http::Header;
use tiny_http::Method;
use tiny_http::Request;
use tiny_http::Response;
use tiny_http::Server;
use tiny_http::StatusCode;

mod downstream_api_keys;
mod read_api_key;
mod usage_stats;
use downstream_api_keys::DownstreamApiKeyRegistry;
use read_api_key::read_auth_header_from_stdin;
use usage_stats::TrackingBodyReader;
use usage_stats::UsageStatsStore;

/// CLI arguments for the proxy.
#[derive(Debug, Clone, Parser)]
#[command(name = "responses-api-proxy", about = "Minimal OpenAI responses proxy")]
pub struct Args {
    /// Port to listen on. If not set, an ephemeral port is used.
    #[arg(long)]
    pub port: Option<u16>,

    /// Path to a JSON file to write startup info (single line). Includes {"port": <u16>}.
    #[arg(long, value_name = "FILE")]
    pub server_info: Option<PathBuf>,

    /// Enable HTTP shutdown endpoint at GET /shutdown
    #[arg(long)]
    pub http_shutdown: bool,

    /// Absolute URL the proxy should forward requests to (defaults to OpenAI).
    #[arg(long, default_value = "https://api.openai.com/v1/responses")]
    pub upstream_url: String,

    /// Path to JSON config that contains per-client downstream API keys.
    #[arg(long, value_name = "FILE")]
    pub downstream_api_keys: Option<PathBuf>,

    /// Prints a newly generated downstream API key and exits.
    #[arg(long)]
    pub generate_downstream_api_key: bool,
}

#[derive(Serialize)]
struct ServerInfo {
    port: u16,
    pid: u32,
}

struct ForwardConfig {
    upstream_url: Url,
    host_header: HeaderValue,
}

/// Entry point for the library main, for parity with other crates.
pub fn run_main(args: Args) -> Result<()> {
    if args.generate_downstream_api_key {
        println!("{}", generate_downstream_api_key()?);
        return Ok(());
    }

    let auth_header = read_auth_header_from_stdin()?;
    let downstream_api_keys = match args.downstream_api_keys.as_ref() {
        Some(path) => Some(Arc::new(DownstreamApiKeyRegistry::from_file(path)?)),
        None => None,
    };
    let usage_stats_store = Arc::new(UsageStatsStore::new());

    let upstream_url = Url::parse(&args.upstream_url).context("parsing --upstream-url")?;
    let host = match (upstream_url.host_str(), upstream_url.port()) {
        (Some(host), Some(port)) => format!("{host}:{port}"),
        (Some(host), None) => host.to_string(),
        _ => return Err(anyhow!("upstream URL must include a host")),
    };
    let host_header =
        HeaderValue::from_str(&host).context("constructing Host header from upstream URL")?;

    let forward_config = Arc::new(ForwardConfig {
        upstream_url,
        host_header,
    });

    let (listener, bound_addr) = bind_listener(args.port)?;
    if let Some(path) = args.server_info.as_ref() {
        write_server_info(path, bound_addr.port())?;
    }
    let server = Server::from_listener(listener, None)
        .map_err(|err| anyhow!("creating HTTP server: {err}"))?;
    let client = Arc::new(
        Client::builder()
            // Disable reqwest's 30s default so long-lived response streams keep flowing.
            .timeout(None::<Duration>)
            .build()
            .context("building reqwest client")?,
    );

    eprintln!("responses-api-proxy listening on {bound_addr}");

    let http_shutdown = args.http_shutdown;
    for request in server.incoming_requests() {
        let client = client.clone();
        let forward_config = forward_config.clone();
        let downstream_api_keys = downstream_api_keys.clone();
        let usage_stats_store = usage_stats_store.clone();
        std::thread::spawn(move || {
            if http_shutdown && request.method() == &Method::Get && request.url() == "/shutdown" {
                let _ = request.respond(Response::new_empty(StatusCode(200)));
                std::process::exit(0);
            }

            if request.method() == &Method::Get && request.url() == "/metrics" {
                let snapshot = usage_stats_store.snapshot();
                if let Err(err) = respond_json(request, StatusCode(200), &snapshot) {
                    eprintln!("failed to respond with usage metrics: {err}");
                }
                return;
            }

            if let Err(e) = forward_request(
                &client,
                auth_header,
                &forward_config,
                downstream_api_keys.as_deref(),
                usage_stats_store.as_ref(),
                request,
            ) {
                eprintln!("forwarding error: {e}");
            }
        });
    }

    Err(anyhow!("server stopped unexpectedly"))
}

fn bind_listener(port: Option<u16>) -> Result<(TcpListener, SocketAddr)> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port.unwrap_or(0)));
    let listener = TcpListener::bind(addr).with_context(|| format!("failed to bind {addr}"))?;
    let bound = listener.local_addr().context("failed to read local_addr")?;
    Ok((listener, bound))
}

fn write_server_info(path: &Path, port: u16) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let info = ServerInfo {
        port,
        pid: std::process::id(),
    };
    let mut data = serde_json::to_string(&info)?;
    data.push('\n');
    let mut f = File::create(path)?;
    f.write_all(data.as_bytes())?;
    Ok(())
}

fn forward_request(
    client: &Client,
    auth_header: &'static str,
    config: &ForwardConfig,
    downstream_api_keys: Option<&DownstreamApiKeyRegistry>,
    usage_stats_store: &UsageStatsStore,
    mut req: Request,
) -> Result<()> {
    // Only allow POST /v1/responses exactly, no query string.
    let method = req.method().clone();
    let url_path = normalize_incoming_url_path(req.url());
    let allow = method == Method::Post && url_path == "/v1/responses";

    if !allow {
        let resp = Response::new_empty(StatusCode(403));
        let _ = req.respond(resp);
        return Ok(());
    }

    let client_id = if let Some(registry) = downstream_api_keys {
        let Some(key) = extract_bearer_token(&req) else {
            let resp = Response::new_empty(StatusCode(401));
            let _ = req.respond(resp);
            return Ok(());
        };

        let Some(client_id) = registry.lookup_client_id(key) else {
            let resp = Response::new_empty(StatusCode(401));
            let _ = req.respond(resp);
            return Ok(());
        };

        Some(client_id.to_string())
    } else {
        None
    };

    if let Some(client_id) = client_id.as_deref() {
        usage_stats_store.record_request(client_id);
    }

    // Read request body
    let mut body = Vec::new();
    let mut reader = req.as_reader();
    std::io::Read::read_to_end(&mut reader, &mut body)?;

    // Build headers for upstream, forwarding everything from the incoming
    // request except Authorization (we replace it below).
    let mut headers = HeaderMap::new();
    for header in req.headers() {
        let name_ascii = header.field.as_str();
        let lower = name_ascii.to_ascii_lowercase();
        if lower.as_str() == "authorization" || lower.as_str() == "host" {
            continue;
        }

        let header_name = match HeaderName::from_bytes(lower.as_bytes()) {
            Ok(name) => name,
            Err(_) => continue,
        };
        if let Ok(value) = HeaderValue::from_bytes(header.value.as_bytes()) {
            headers.append(header_name, value);
        }
    }

    // As part of our effort to to keep `auth_header` secret, we use a
    // combination of `from_static()` and `set_sensitive(true)`.
    let mut auth_header_value = HeaderValue::from_static(auth_header);
    auth_header_value.set_sensitive(true);
    headers.insert(AUTHORIZATION, auth_header_value);

    headers.insert(HOST, config.host_header.clone());

    let upstream_resp = client
        .post(config.upstream_url.clone())
        .headers(headers)
        .body(body)
        .send()
        .context("forwarding request to upstream")?;

    // We have to create an adapter between a `reqwest::blocking::Response`
    // and a `tiny_http::Response`. Fortunately, `reqwest::blocking::Response`
    // implements `Read`, so we can use it directly as the body of the
    // `tiny_http::Response`.
    let status = upstream_resp.status();
    let mut response_headers = Vec::new();
    for (name, value) in upstream_resp.headers().iter() {
        // Skip headers that tiny_http manages itself.
        if matches!(
            name.as_str(),
            "content-length" | "transfer-encoding" | "connection" | "trailer" | "upgrade"
        ) {
            continue;
        }

        if let Ok(header) = Header::from_bytes(name.as_str().as_bytes(), value.as_bytes()) {
            response_headers.push(header);
        }
    }

    let content_length = upstream_resp.content_length().and_then(|len| {
        if len <= usize::MAX as u64 {
            Some(len as usize)
        } else {
            None
        }
    });

    let tracking_reader =
        TrackingBodyReader::new(upstream_resp, client_id, usage_stats_store.clone());

    let response = Response::new(
        StatusCode(status.as_u16()),
        response_headers,
        tracking_reader,
        content_length,
        None,
    );

    let _ = req.respond(response);
    Ok(())
}

fn extract_bearer_token(req: &Request) -> Option<&str> {
    req.headers().iter().find_map(|header| {
        if !header.field.equiv("Authorization") {
            return None;
        }

        parse_bearer_token(header.value.as_str())
    })
}

fn parse_bearer_token(value: &str) -> Option<&str> {
    value.strip_prefix("Bearer ")
}

fn normalize_incoming_url_path(raw_url: &str) -> String {
    if let Ok(parsed) = Url::parse(raw_url) {
        if parsed.query().is_some() {
            return format!("{}?{}", parsed.path(), parsed.query().unwrap_or_default());
        }
        return parsed.path().to_string();
    }

    raw_url.to_string()
}

fn respond_json<T: Serialize>(request: Request, status: StatusCode, value: &T) -> Result<()> {
    let content_type = Header::from_bytes("Content-Type", "application/json")
        .map_err(|_| anyhow!("failed to build content-type header"))?;
    let body = serde_json::to_string(value)?;
    request
        .respond(
            Response::from_string(body)
                .with_status_code(status)
                .with_header(content_type),
        )
        .context("writing JSON response")?;
    Ok(())
}

fn generate_downstream_api_key() -> Result<String> {
    let mut random = [0u8; 24];
    let mut rng = OsRng;
    rng.try_fill_bytes(&mut random)
        .context("generating random bytes for downstream API key")?;

    let mut key = String::from("dsk_");
    for byte in random {
        key.push(hex_digit((byte >> 4) & 0x0f));
        key.push(hex_digit(byte & 0x0f));
    }

    Ok(key)
}

fn hex_digit(nibble: u8) -> char {
    match nibble {
        0..=9 => (b'0' + nibble) as char,
        10..=15 => (b'a' + nibble - 10) as char,
        _ => unreachable!("nibble should be in 0..=15"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bearer_token_parses_authorization_header() {
        assert_eq!(parse_bearer_token("Bearer dsk_123"), Some("dsk_123"));
        assert_eq!(parse_bearer_token("Basic abc"), None);
    }

    #[test]
    fn normalize_incoming_url_path_supports_origin_and_absolute_form() {
        assert_eq!(
            normalize_incoming_url_path("/v1/responses"),
            "/v1/responses"
        );
        assert_eq!(
            normalize_incoming_url_path("http://127.0.0.1:18081/v1/responses"),
            "/v1/responses"
        );
        assert_eq!(
            normalize_incoming_url_path("http://127.0.0.1:18081/v1/responses?a=1"),
            "/v1/responses?a=1"
        );
    }

    #[test]
    fn generated_key_has_expected_prefix_and_length() {
        let key = generate_downstream_api_key().expect("key should be generated");
        assert_eq!(key.starts_with("dsk_"), true);
        assert_eq!(key.len(), 52);
    }
}
