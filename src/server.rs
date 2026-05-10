use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::api::{self, ChatCompletionRequest, CompletionRequest};
use crate::inference::{InferenceBackend, NativeMlxBackend};

const OLLAMA_TOOL_MAX_TOKENS: u32 = 1024;
const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const MAX_BUFFERED_TOOL_CALL_CHARS: usize = 64 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ServerConfig {
    pub model: String,
    pub host: String,
    pub port: u16,
    pub model_id: String,
}

#[derive(Clone, Debug, Deserialize)]
struct OllamaChatRequest {
    model: Option<String>,
    messages: Vec<OllamaMessage>,
    tools: Option<Vec<OllamaTool>>,
    stream: Option<bool>,
    think: Option<bool>,
    options: Option<OllamaOptions>,
}

#[derive(Clone, Debug, Deserialize)]
struct OllamaMessage {
    role: String,
    content: serde_json::Value,
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Clone, Debug, Deserialize)]
struct OllamaOptions {
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    num_predict: Option<u32>,
    num_ctx: Option<u32>,
    stop: Option<serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OllamaTool {
    #[serde(rename = "type")]
    kind: String,
    function: OllamaToolDefinition,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OllamaToolDefinition {
    name: String,
    description: Option<String>,
    parameters: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OllamaToolCall {
    function: OllamaFunctionCall,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OllamaFunctionCall {
    name: String,
    #[serde(default)]
    arguments: serde_json::Value,
}

impl OllamaChatRequest {
    fn into_inference(
        self,
        default_model: &str,
        public_model_id: &str,
    ) -> crate::inference::InferenceRequest {
        let options = self.options.unwrap_or_default();
        let _requested_num_ctx = options.num_ctx;
        let _think = self.think.unwrap_or(false);
        let tools_prompt = ollama_tools_prompt(self.tools.as_deref().unwrap_or(&[]));
        let messages = self
            .messages
            .into_iter()
            .map(|message| api::ChatMessage {
                role: ollama_message_role(&message.role).to_string(),
                content: serde_json::Value::String(ollama_message_content(message)),
            })
            .collect::<Vec<_>>();
        let mut request = ChatCompletionRequest {
            model: self.model,
            messages,
            max_tokens: options.num_predict,
            max_completion_tokens: None,
            temperature: options.temperature,
            top_p: options.top_p,
            top_k: options.top_k,
            stop: options.stop,
            stream: self.stream,
            generation_mode: None,
        }
        .into_inference(default_model, public_model_id);
        if let Some(tools_prompt) = tools_prompt {
            request.system = Some(match request.system {
                Some(system) if !system.is_empty() => format!("{system}\n\n{tools_prompt}"),
                _ => tools_prompt,
            });
        }
        request
    }

    fn has_tools(&self) -> bool {
        self.tools.as_ref().is_some_and(|tools| !tools.is_empty())
    }
}

impl Default for OllamaOptions {
    fn default() -> Self {
        Self {
            temperature: None,
            top_p: None,
            top_k: None,
            num_predict: None,
            num_ctx: None,
            stop: None,
        }
    }
}

pub fn serve(config: ServerConfig) -> Result<()> {
    let listener = TcpListener::bind((&*config.host, config.port))
        .with_context(|| format!("bind {}:{}", config.host, config.port))?;
    println!(
        "Ferrite server listening on http://{}:{}",
        config.host, config.port
    );
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
        ("GET", "/api/tags") => write_json(
            &mut stream,
            200,
            &json!({
                "models": [{
                    "name": config.model_id,
                    "model": config.model_id,
                    "modified_at": iso_time(),
                    "size": 0,
                    "digest": "ferrite-local",
                    "details": {
                        "parent_model": "",
                        "format": "safetensors",
                        "family": "qwen3.6",
                        "families": ["qwen3.6"],
                        "parameter_size": "27B",
                        "quantization_level": "4-bit"
                    }
                }]
            }),
        )?,
        ("GET", "/api/version") => write_json(
            &mut stream,
            200,
            &json!({"version": env!("CARGO_PKG_VERSION")}),
        )?,
        ("POST", "/api/show") => write_json(
            &mut stream,
            200,
            &json!({
                "modelfile": "",
                "parameters": "num_ctx 65536",
                "model_info": {
                    "general.architecture": "qwen3.6",
                    "general.context_length": 262144,
                    "qwen3.6.context_length": 262144
                }
            }),
        )?,
        ("POST", "/api/chat") => {
            let parsed: OllamaChatRequest = serde_json::from_str(body)?;
            let response_model = parsed
                .model
                .clone()
                .unwrap_or_else(|| config.model_id.clone());
            let wants_stream = parsed.stream.unwrap_or(true);
            let has_tools = parsed.has_tools();
            let mut request = parsed.into_inference(&config.model, &config.model_id);
            if has_tools {
                bound_ollama_tool_request(&mut request);
            }
            if wants_stream && request.stop.is_empty() && !has_tools {
                write_ollama_chat_stream(&mut stream, backend, &request, &response_model)?;
                return Ok(());
            }
            if wants_stream && has_tools {
                write_ollama_tool_chat_stream(&mut stream, backend, &request, &response_model)?;
                return Ok(());
            }
            match backend.infer(&request) {
                Ok(response) if wants_stream => {
                    let (content, tool_calls) = extract_ollama_tool_calls(&response.text);
                    write_ollama_chat_compat_stream(
                        &mut stream,
                        &response_model,
                        &content,
                        &tool_calls,
                    )?
                }
                Ok(response) => {
                    let (content, tool_calls) = extract_ollama_tool_calls(&response.text);
                    let mut message = json!({"role": "assistant", "content": content});
                    if !tool_calls.is_empty() {
                        message["tool_calls"] = json!(tool_calls);
                    }
                    write_json(
                        &mut stream,
                        200,
                        &json!({
                            "model": response_model,
                            "created_at": iso_time(),
                            "message": message,
                            "done_reason": ollama_done_reason(&response.finish_reason),
                            "done": true,
                            "prompt_eval_count": response.prompt_tokens,
                            "eval_count": response.completion_tokens
                        }),
                    )?
                }
                Err(err) => write_json(
                    &mut stream,
                    500,
                    &api::error(err.to_string(), "backend_error"),
                )?,
            }
        }
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

fn write_ollama_chat_stream(
    stream: &mut TcpStream,
    backend: &NativeMlxBackend,
    request: &crate::inference::InferenceRequest,
    model: &str,
) -> Result<()> {
    write_ndjson_headers(stream)?;
    let result = backend.infer_stream(request, |delta| {
        write_json_line(
            stream,
            &json!({
                "model": model,
                "created_at": iso_time(),
                "message": {"role": "assistant", "content": delta},
                "done": false
            }),
        )
    });
    match result {
        Ok(response) => write_json_line(
            stream,
            &json!({
                "model": model,
                "created_at": iso_time(),
                "message": {"role": "assistant", "content": ""},
                "done_reason": ollama_done_reason(&response.finish_reason),
                "done": true,
                "prompt_eval_count": response.prompt_tokens,
                "eval_count": response.completion_tokens
            }),
        )?,
        Err(err) => write_json_line(
            stream,
            &json!({
                "error": err.to_string(),
                "done": true
            }),
        )?,
    }
    Ok(())
}

fn write_ollama_tool_chat_stream(
    stream: &mut TcpStream,
    backend: &NativeMlxBackend,
    request: &crate::inference::InferenceRequest,
    model: &str,
) -> Result<()> {
    write_ndjson_headers(stream)?;
    let mut filter = OllamaToolStreamFilter::new();
    let result = backend.infer_stream(request, |delta| {
        for event in filter.feed(delta) {
            write_ollama_stream_event(stream, model, event)?;
        }
        Ok(())
    });
    for event in filter.flush() {
        write_ollama_stream_event(stream, model, event)?;
    }
    match result {
        Ok(response) => write_json_line(
            stream,
            &json!({
                "model": model,
                "created_at": iso_time(),
                "message": {"role": "assistant", "content": ""},
                "done_reason": ollama_done_reason(&response.finish_reason),
                "done": true,
                "prompt_eval_count": response.prompt_tokens,
                "eval_count": response.completion_tokens
            }),
        )?,
        Err(err) => write_json_line(
            stream,
            &json!({
                "error": err.to_string(),
                "done": true
            }),
        )?,
    }
    Ok(())
}

fn write_ollama_stream_event(
    stream: &mut TcpStream,
    model: &str,
    event: OllamaStreamEvent,
) -> Result<()> {
    match event {
        OllamaStreamEvent::Text(text) if text.is_empty() => Ok(()),
        OllamaStreamEvent::Text(text) => write_json_line(
            stream,
            &json!({
                "model": model,
                "created_at": iso_time(),
                "message": {"role": "assistant", "content": text},
                "done": false
            }),
        ),
        OllamaStreamEvent::ToolCall(call) => write_json_line(
            stream,
            &json!({
                "model": model,
                "created_at": iso_time(),
                "message": {"role": "assistant", "content": "", "tool_calls": [call]},
                "done": false
            }),
        ),
    }
}

fn write_ollama_chat_compat_stream(
    stream: &mut TcpStream,
    model: &str,
    text: &str,
    tool_calls: &[OllamaToolCall],
) -> Result<()> {
    write_ndjson_headers(stream)?;
    if !text.is_empty() || !tool_calls.is_empty() {
        let mut message = json!({"role": "assistant", "content": text});
        if !tool_calls.is_empty() {
            message["tool_calls"] = json!(tool_calls);
        }
        write_json_line(
            stream,
            &json!({
                "model": model,
                "created_at": iso_time(),
                "message": message,
                "done": false
            }),
        )?;
    }
    write_json_line(
        stream,
        &json!({
            "model": model,
            "created_at": iso_time(),
            "message": {"role": "assistant", "content": ""},
            "done": true
        }),
    )
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

fn write_ndjson_headers(stream: &mut TcpStream) -> Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nConnection: close\r\n\r\n"
    )?;
    Ok(())
}

fn write_json_line<T: Serialize>(stream: &mut TcpStream, value: &T) -> Result<()> {
    let body = serde_json::to_string(value)?;
    writeln!(stream, "{body}")?;
    Ok(())
}

fn bound_ollama_tool_request(request: &mut crate::inference::InferenceRequest) {
    request.max_tokens = Some(
        request
            .max_tokens
            .unwrap_or(OLLAMA_TOOL_MAX_TOKENS)
            .min(OLLAMA_TOOL_MAX_TOKENS),
    );
    if !request.stop.iter().any(|stop| stop == TOOL_CALL_CLOSE) {
        request.stop.push(TOOL_CALL_CLOSE.to_string());
    }
}

fn ollama_message_role(role: &str) -> &str {
    if role == "tool" { "user" } else { role }
}

fn ollama_message_content(message: OllamaMessage) -> String {
    let mut content = api::content_to_text(&message.content);
    if message.role == "tool" {
        return format!("<tool_response>\n{content}\n</tool_response>");
    }
    if let Some(tool_calls) = message.tool_calls {
        for call in tool_calls {
            if let Ok(body) = serde_json::to_string(&json!({
                "name": call.function.name,
                "arguments": call.function.arguments
            })) {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str("<tool_call>\n");
                content.push_str(&body);
                content.push_str("\n</tool_call>");
            }
        }
    }
    content
}

fn ollama_tools_prompt(tools: &[OllamaTool]) -> Option<String> {
    if tools.is_empty() {
        return None;
    }
    let mut out = String::from(
        "# Tools\n\nYou may call one or more functions to help with the user request.\n\n\
         You are provided with function signatures inside <tools></tools> XML tags:\n<tools>\n",
    );
    for tool in tools {
        if let Ok(line) = serde_json::to_string(tool) {
            out.push_str(&line);
            out.push('\n');
        }
    }
    out.push_str(
        "</tools>\n\nFor each function call, return a JSON object with function name and arguments \
         inside <tool_call></tool_call> XML tags:\n<tool_call>\n\
         {\"name\":\"function_name\",\"arguments\":{\"arg\":\"value\"}}\n</tool_call>",
    );
    Some(out)
}

enum OllamaStreamEvent {
    Text(String),
    ToolCall(OllamaToolCall),
}

struct OllamaToolStreamFilter {
    buffered: String,
    in_tool_call: bool,
}

impl OllamaToolStreamFilter {
    fn new() -> Self {
        Self {
            buffered: String::new(),
            in_tool_call: false,
        }
    }

    fn feed(&mut self, delta: &str) -> Vec<OllamaStreamEvent> {
        self.buffered.push_str(delta);
        let mut events = Vec::new();
        loop {
            if self.in_tool_call {
                if let Some(close) = self.buffered.find(TOOL_CALL_CLOSE) {
                    let body = self.buffered[..close].trim().to_string();
                    self.buffered.drain(..close + TOOL_CALL_CLOSE.len());
                    self.in_tool_call = false;
                    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) {
                        if let Some(call) = ollama_tool_call_from_value(value) {
                            events.push(OllamaStreamEvent::ToolCall(call));
                            continue;
                        }
                    }
                    events.push(OllamaStreamEvent::Text(format!(
                        "{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}"
                    )));
                    continue;
                }
                if self.buffered.len() > MAX_BUFFERED_TOOL_CALL_CHARS {
                    events.push(OllamaStreamEvent::Text(format!(
                        "{TOOL_CALL_OPEN}{}",
                        self.buffered
                    )));
                    self.buffered.clear();
                    self.in_tool_call = false;
                }
                break;
            }

            if let Some(open) = self.buffered.find(TOOL_CALL_OPEN) {
                if open > 0 {
                    events.push(OllamaStreamEvent::Text(self.buffered[..open].to_string()));
                }
                self.buffered.drain(..open + TOOL_CALL_OPEN.len());
                self.in_tool_call = true;
                continue;
            }

            let emit_len = safe_tool_prefix_emit_len(&self.buffered);
            if emit_len > 0 {
                events.push(OllamaStreamEvent::Text(
                    self.buffered.drain(..emit_len).collect(),
                ));
                continue;
            }
            break;
        }
        events
    }

    fn flush(&mut self) -> Vec<OllamaStreamEvent> {
        if self.buffered.is_empty() {
            return Vec::new();
        }
        let text = if self.in_tool_call {
            format!("{TOOL_CALL_OPEN}{}", std::mem::take(&mut self.buffered))
        } else {
            std::mem::take(&mut self.buffered)
        };
        self.in_tool_call = false;
        vec![OllamaStreamEvent::Text(text)]
    }
}

fn safe_tool_prefix_emit_len(text: &str) -> usize {
    let max_suffix = text.len().min(TOOL_CALL_OPEN.len() - 1);
    for keep in (1..=max_suffix).rev() {
        let boundary = text.len() - keep;
        if text.is_char_boundary(boundary) && TOOL_CALL_OPEN.starts_with(&text[boundary..]) {
            return boundary;
        }
    }
    text.len()
}

fn extract_ollama_tool_calls(text: &str) -> (String, Vec<OllamaToolCall>) {
    let mut remaining = text;
    let mut visible = String::new();
    let mut calls = Vec::new();
    while let Some(start) = remaining.find(TOOL_CALL_OPEN) {
        visible.push_str(&remaining[..start]);
        let after_start = &remaining[start + TOOL_CALL_OPEN.len()..];
        let Some(end) = after_start.find(TOOL_CALL_CLOSE) else {
            visible.push_str(&remaining[start..]);
            return (visible.trim().to_string(), calls);
        };
        let body = after_start[..end].trim();
        match serde_json::from_str::<serde_json::Value>(body)
            .ok()
            .and_then(ollama_tool_call_from_value)
        {
            Some(call) => calls.push(call),
            None => {
                visible.push_str(TOOL_CALL_OPEN);
                visible.push_str(body);
                visible.push_str(TOOL_CALL_CLOSE);
            }
        }
        remaining = &after_start[end + TOOL_CALL_CLOSE.len()..];
    }
    visible.push_str(remaining);
    (visible.trim().to_string(), calls)
}

fn ollama_tool_call_from_value(value: serde_json::Value) -> Option<OllamaToolCall> {
    if let Some(array) = value.as_array() {
        return array.first().cloned().and_then(ollama_tool_call_from_value);
    }
    if let Some(function) = value.get("function") {
        let name = function.get("name")?.as_str()?.to_string();
        let arguments = function
            .get("arguments")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        return Some(OllamaToolCall {
            function: OllamaFunctionCall { name, arguments },
        });
    }
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value
        .get("arguments")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    Some(OllamaToolCall {
        function: OllamaFunctionCall { name, arguments },
    })
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
        500 => "Internal Server Error",
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

fn iso_time() -> &'static str {
    "1970-01-01T00:00:00Z"
}

fn ollama_done_reason(reason: &crate::inference::FinishReason) -> &'static str {
    match reason {
        crate::inference::FinishReason::Stop => "stop",
        crate::inference::FinishReason::Length => "length",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ollama_tool_prompt_contains_serialized_tools() {
        let prompt = ollama_tools_prompt(&[OllamaTool {
            kind: "function".to_string(),
            function: OllamaToolDefinition {
                name: "read_file".to_string(),
                description: Some("Read a file".to_string()),
                parameters: json!({"type": "object"}),
            },
        }])
        .unwrap();

        assert!(prompt.contains("<tools>"));
        assert!(prompt.contains("\"name\":\"read_file\""));
        assert!(prompt.contains("<tool_call>"));
    }

    #[test]
    fn ollama_message_content_renders_tool_history() {
        let content = ollama_message_content(OllamaMessage {
            role: "assistant".to_string(),
            content: serde_json::Value::String(String::new()),
            tool_calls: Some(vec![OllamaToolCall {
                function: OllamaFunctionCall {
                    name: "list_dir".to_string(),
                    arguments: json!({"path": "."}),
                },
            }]),
        });

        assert!(content.contains("<tool_call>"));
        assert!(content.contains("\"name\":\"list_dir\""));
        assert!(content.contains("\"path\":\".\""));
    }

    #[test]
    fn ollama_message_content_maps_tool_result_to_user_payload() {
        let content = ollama_message_content(OllamaMessage {
            role: "tool".to_string(),
            content: serde_json::Value::String("file contents".to_string()),
            tool_calls: None,
        });

        assert_eq!(content, "<tool_response>\nfile contents\n</tool_response>");
        assert_eq!(ollama_message_role("tool"), "user");
    }

    #[test]
    fn extract_ollama_tool_calls_removes_blocks_from_visible_text() {
        let (text, calls) = extract_ollama_tool_calls(
            "I will inspect.\n<tool_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"Cargo.toml\"}}\n</tool_call>",
        );

        assert_eq!(text, "I will inspect.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments["path"], "Cargo.toml");
    }

    #[test]
    fn bound_ollama_tool_request_caps_tokens_and_adds_stop() {
        let mut request = crate::inference::InferenceRequest {
            model: "model".to_string(),
            prompt: "hi".to_string(),
            system: None,
            messages: Vec::new(),
            stop: Vec::new(),
            max_tokens: Some(4096),
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
            depth: 2,
            mtp: true,
            profile_timings: false,
        };

        bound_ollama_tool_request(&mut request);

        assert_eq!(request.max_tokens, Some(OLLAMA_TOOL_MAX_TOKENS));
        assert_eq!(request.stop, vec![TOOL_CALL_CLOSE.to_string()]);
    }

    #[test]
    fn tool_stream_filter_extracts_split_tool_call() {
        let mut filter = OllamaToolStreamFilter::new();
        let mut events = filter.feed("before <tool");
        events.extend(
            filter.feed("_call>\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"Cargo.toml\"}}"),
        );
        events.extend(filter.feed("\n</tool_call> after"));
        events.extend(filter.flush());

        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], OllamaStreamEvent::Text(text) if text == "before "));
        assert!(matches!(
            &events[1],
            OllamaStreamEvent::ToolCall(call)
                if call.function.name == "read_file"
                    && call.function.arguments["path"] == "Cargo.toml"
        ));
        assert!(matches!(&events[2], OllamaStreamEvent::Text(text) if text == " after"));
    }
}
