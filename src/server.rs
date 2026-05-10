use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::api::{self, ChatCompletionRequest, CompletionRequest};
use crate::inference::{InferenceBackend, NativeMlxBackend};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    pub model: String,
    pub host: String,
    pub port: u16,
    pub model_id: String,
}

pub fn serve(config: ServerConfig) -> Result<()> {
    let listener = TcpListener::bind((&*config.host, config.port))
        .with_context(|| format!("bind {}:{}", config.host, config.port))?;
    println!(
        "Ferrite server listening on http://{}:{}",
        config.host, config.port
    );
    println!("Python dependency: no");
    println!("Native backend: {}", NativeMlxBackend::status());
    let backend = NativeMlxBackend::new();
    let preload_s = backend.preload(&config.model)?;
    println!("Preloaded model in {:.3}s", preload_s);
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                if let Err(err) = handle_connection(stream, &config, &backend) {
                    eprintln!("request error: {err:#}");
                }
            }
            Err(err) => eprintln!("accept error: {err}"),
        }
    }
    Ok(())
}

fn handle_connection(
    mut stream: TcpStream,
    config: &ServerConfig,
    backend: &NativeMlxBackend,
) -> Result<()> {
    let mut buffer = Vec::new();
    let mut temp = [0_u8; 8192];
    loop {
        let n = stream.read(&mut temp)?;
        if n == 0 {
            break;
        }
        buffer.extend_from_slice(&temp[..n]);
        if request_complete(&buffer) {
            break;
        }
        if buffer.len() > 16 * 1024 * 1024 {
            write_json(
                &mut stream,
                413,
                &api::error("request too large", "invalid_request_error"),
            )?;
            return Ok(());
        }
    }

    let request = String::from_utf8_lossy(&buffer);
    let (head, body) = split_request(&request);
    let mut lines = head.lines();
    let Some(request_line) = lines.next() else {
        write_json(
            &mut stream,
            400,
            &api::error("empty request", "invalid_request_error"),
        )?;
        return Ok(());
    };
    let parts = request_line.split_whitespace().collect::<Vec<_>>();
    if parts.len() < 2 {
        write_json(
            &mut stream,
            400,
            &api::error("bad request line", "invalid_request_error"),
        )?;
        return Ok(());
    }
    let method = parts[0];
    let path = parts[1].split('?').next().unwrap_or(parts[1]);

    match (method, path) {
        ("GET", "/health") => write_json(
            &mut stream,
            200,
            &json!({"ok": true, "backend": NativeMlxBackend::status()}),
        )?,
        ("GET", "/v1/models") => write_json(
            &mut stream,
            200,
            &json!({
                "object": "list",
                "data": [{
                    "id": config.model_id,
                    "object": "model",
                    "created": unix_time(),
                    "owned_by": "ferrite"
                }]
            }),
        )?,
        ("POST", "/v1/chat/completions") => {
            let parsed: ChatCompletionRequest = serde_json::from_str(body)?;
            let wants_stream = parsed.stream.unwrap_or(false);
            let response_model = parsed
                .model
                .clone()
                .unwrap_or_else(|| config.model_id.clone());
            let request = parsed.into_inference(&config.model, &config.model_id);
            if wants_stream && request.stop.is_empty() {
                write_chat_sse_live(&mut stream, backend, &request, &response_model)?;
                return Ok(());
            }
            match backend.infer(&request) {
                Ok(response) if wants_stream => write_chat_sse(
                    &mut stream,
                    &response_model,
                    &response.text,
                    &response.finish_reason,
                )?,
                Ok(response) => write_json(
                    &mut stream,
                    200,
                    &json!({
                    "id": format!("chatcmpl-{}", unix_time()),
                    "object": "chat.completion",
                    "created": unix_time(),
                    "model": response_model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response.text},
                        "finish_reason": response.finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_tokens": response.prompt_tokens + response.completion_tokens
                    },
                    "ferrite_metrics": response.timings
                    }),
                )?,
                Err(err) => write_json(
                    &mut stream,
                    501,
                    &api::error(err.to_string(), "backend_not_implemented"),
                )?,
            }
        }
        ("POST", "/v1/completions") => {
            let parsed: CompletionRequest = serde_json::from_str(body)?;
            let wants_stream = parsed.stream.unwrap_or(false);
            let response_model = parsed
                .model
                .clone()
                .unwrap_or_else(|| config.model_id.clone());
            let request = parsed.into_inference(&config.model, &config.model_id);
            if wants_stream && request.stop.is_empty() {
                write_completion_sse_live(&mut stream, backend, &request, &response_model)?;
                return Ok(());
            }
            match backend.infer(&request) {
                Ok(response) if wants_stream => write_completion_sse(
                    &mut stream,
                    &response_model,
                    &response.text,
                    &response.finish_reason,
                )?,
                Ok(response) => write_json(
                    &mut stream,
                    200,
                    &json!({
                    "id": format!("cmpl-{}", unix_time()),
                    "object": "text_completion",
                    "created": unix_time(),
                    "model": response_model,
                    "choices": [{
                        "index": 0,
                        "text": response.text,
                        "finish_reason": response.finish_reason
                    }],
                    "usage": {
                        "prompt_tokens": response.prompt_tokens,
                        "completion_tokens": response.completion_tokens,
                        "total_tokens": response.prompt_tokens + response.completion_tokens
                    },
                    "ferrite_metrics": response.timings
                    }),
                )?,
                Err(err) => write_json(
                    &mut stream,
                    501,
                    &api::error(err.to_string(), "backend_not_implemented"),
                )?,
            }
        }
        _ => write_json(
            &mut stream,
            404,
            &api::error("not found", "invalid_request_error"),
        )?,
    }
    Ok(())
}

fn write_chat_sse_live(
    stream: &mut TcpStream,
    backend: &NativeMlxBackend,
    request: &crate::inference::InferenceRequest,
    model: &str,
) -> Result<()> {
    write_sse_headers(stream)?;
    let id = format!("chatcmpl-{}", unix_time());
    write_sse_event(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": unix_time(),
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        }),
    )?;
    let result = backend.infer_stream(request, |delta| {
        write_sse_event(
            stream,
            &json!({
                "id": id,
                "object": "chat.completion.chunk",
                "created": unix_time(),
                "model": model,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": null}]
            }),
        )
    });
    match result {
        Ok(response) => {
            write_sse_event(
                stream,
                &json!({
                    "id": id,
                    "object": "chat.completion.chunk",
                    "created": unix_time(),
                    "model": model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": response.finish_reason}]
                }),
            )?;
        }
        Err(err) => {
            write_sse_event(
                stream,
                &json!({
                    "error": {
                        "message": err.to_string(),
                        "type": "backend_error"
                    }
                }),
            )?;
        }
    }
    stream.write_all(b"data: [DONE]\n\n")?;
    Ok(())
}

fn write_completion_sse_live(
    stream: &mut TcpStream,
    backend: &NativeMlxBackend,
    request: &crate::inference::InferenceRequest,
    model: &str,
) -> Result<()> {
    write_sse_headers(stream)?;
    let id = format!("cmpl-{}", unix_time());
    let result = backend.infer_stream(request, |delta| {
        write_sse_event(
            stream,
            &json!({
                "id": id,
                "object": "text_completion",
                "created": unix_time(),
                "model": model,
                "choices": [{"index": 0, "text": delta, "finish_reason": null}]
            }),
        )
    });
    match result {
        Ok(response) => {
            write_sse_event(
                stream,
                &json!({
                    "id": id,
                    "object": "text_completion",
                    "created": unix_time(),
                    "model": model,
                    "choices": [{"index": 0, "text": "", "finish_reason": response.finish_reason}]
                }),
            )?;
        }
        Err(err) => {
            write_sse_event(
                stream,
                &json!({
                    "error": {
                        "message": err.to_string(),
                        "type": "backend_error"
                    }
                }),
            )?;
        }
    }
    stream.write_all(b"data: [DONE]\n\n")?;
    Ok(())
}

fn write_chat_sse(
    stream: &mut TcpStream,
    model: &str,
    text: &str,
    finish_reason: &crate::inference::FinishReason,
) -> Result<()> {
    write_sse_headers(stream)?;
    let id = format!("chatcmpl-{}", unix_time());
    write_sse_event(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": unix_time(),
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        }),
    )?;
    write_sse_event(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": unix_time(),
            "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}]
        }),
    )?;
    write_sse_event(
        stream,
        &json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": unix_time(),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]
        }),
    )?;
    stream.write_all(b"data: [DONE]\n\n")?;
    Ok(())
}

fn write_completion_sse(
    stream: &mut TcpStream,
    model: &str,
    text: &str,
    finish_reason: &crate::inference::FinishReason,
) -> Result<()> {
    write_sse_headers(stream)?;
    let id = format!("cmpl-{}", unix_time());
    write_sse_event(
        stream,
        &json!({
            "id": id,
            "object": "text_completion",
            "created": unix_time(),
            "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": null}]
        }),
    )?;
    write_sse_event(
        stream,
        &json!({
            "id": id,
            "object": "text_completion",
            "created": unix_time(),
            "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": finish_reason}]
        }),
    )?;
    stream.write_all(b"data: [DONE]\n\n")?;
    Ok(())
}

fn write_sse_headers(stream: &mut TcpStream) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n"
    )?;
    Ok(())
}

fn write_sse_event<T: Serialize>(stream: &mut TcpStream, value: &T) -> Result<()> {
    let body = serde_json::to_string(value)?;
    write!(stream, "data: {body}\n\n")?;
    Ok(())
}

fn request_complete(buffer: &[u8]) -> bool {
    let Some(header_end) = find_subsequence(buffer, b"\r\n\r\n") else {
        return false;
    };
    let head = String::from_utf8_lossy(&buffer[..header_end]);
    let content_length = head
        .lines()
        .find_map(|line| line.strip_prefix("Content-Length:"))
        .or_else(|| {
            head.lines()
                .find_map(|line| line.strip_prefix("content-length:"))
        })
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0);
    buffer.len() >= header_end + 4 + content_length
}

fn split_request(request: &str) -> (&str, &str) {
    request.split_once("\r\n\r\n").unwrap_or((request, ""))
}

fn write_json<T: Serialize>(stream: &mut TcpStream, status: u16, value: &T) -> Result<()> {
    let body = serde_json::to_vec(value)?;
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        501 => "Not Implemented",
        _ => "OK",
    };
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(&body)?;
    Ok(())
}

fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}
