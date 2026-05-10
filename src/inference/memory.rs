use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{
    Arc, Once,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::Result;

use super::{InferenceRequest, env_flag, ferrite_env_var};

const DEFAULT_MLX_CACHE_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const DEFAULT_MLX_MEMORY_LIMIT_BYTES: usize = 8 * 1024 * 1024 * 1024;
const DEFAULT_RSS_KILL_PERCENT: u64 = 50;
const DEFAULT_MLX_ACTIVE_KILL_PERCENT: u64 = 80;

#[derive(Clone, Copy, Debug)]
struct MemorySnapshot {
    rss_bytes: Option<u64>,
    mlx_active_bytes: Option<usize>,
    mlx_cache_bytes: Option<usize>,
    mlx_peak_bytes: Option<usize>,
}

pub(super) struct RequestMemoryGuard {
    enabled: bool,
    request_label: &'static str,
    watchdog: Option<RequestWatchdog>,
}

struct RequestWatchdog {
    alive: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl RequestMemoryGuard {
    pub(super) fn new(request_label: &'static str) -> Self {
        configure_mlx_memory_once();
        let enabled = memory_trace_enabled();
        if enabled {
            log_memory_sample(request_label, "start", "");
        }
        Self {
            enabled,
            request_label,
            watchdog: RequestWatchdog::start(request_label),
        }
    }
}

impl Drop for RequestMemoryGuard {
    fn drop(&mut self) {
        self.watchdog.take();
        clear_mlx_cache();
        if self.enabled {
            log_memory_sample(self.request_label, "drop_after_clear_cache", "");
        }
    }
}

impl RequestWatchdog {
    fn start(request_label: &'static str) -> Option<Self> {
        if !env_flag("MTPLX_MEMORY_WATCHDOG", true) {
            return None;
        }
        let alive = Arc::new(AtomicBool::new(true));
        let thread_alive = Arc::clone(&alive);
        let interval = memory_watchdog_interval();
        let handle = thread::Builder::new()
            .name("ferrite-memory-watchdog".to_string())
            .spawn(move || memory_watchdog_loop(request_label, thread_alive, interval))
            .ok()?;
        Some(Self {
            alive,
            handle: Some(handle),
        })
    }
}

impl Drop for RequestWatchdog {
    fn drop(&mut self) {
        self.alive.store(false, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub(super) fn memory_trace_enabled() -> bool {
    env_flag("MTPLX_MEMORY_TRACE", true)
}

fn configure_mlx_memory_once() {
    static CONFIGURE: Once = Once::new();
    CONFIGURE.call_once(|| {
        let memory_limit = ferrite_env_var("MTPLX_MLX_MEMORY_LIMIT_BYTES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_MLX_MEMORY_LIMIT_BYTES);
        if memory_limit > 0 {
            let mut previous = 0_usize;
            let status = unsafe { mlx_sys::mlx_set_memory_limit(&mut previous, memory_limit) };
            if status != 0 {
                let _ = append_memory_log_line(
                    "ferrite_memory phase=configure_mlx_memory_limit status=error",
                );
            } else if memory_trace_enabled() {
                let line = format!(
                    "ferrite_memory phase=configure_mlx_memory_limit limit_bytes={memory_limit} previous_bytes={previous}"
                );
                let _ = append_memory_log_line(&line);
                if env_flag("MTPLX_MEMORY_TRACE_STDERR", false) {
                    eprintln!("{line}");
                }
            }
        }

        let cache_limit = ferrite_env_var("MTPLX_MLX_CACHE_LIMIT_BYTES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_MLX_CACHE_LIMIT_BYTES);
        if cache_limit > 0 {
            let mut previous = 0_usize;
            let status = unsafe { mlx_sys::mlx_set_cache_limit(&mut previous, cache_limit) };
            if status != 0 {
                let _ = append_memory_log_line(
                    "ferrite_memory phase=configure_mlx_cache_limit status=error",
                );
            } else if memory_trace_enabled() {
                let line = format!(
                    "ferrite_memory phase=configure_mlx_cache_limit limit_bytes={cache_limit} previous_bytes={previous}"
                );
                let _ = append_memory_log_line(&line);
                if env_flag("MTPLX_MEMORY_TRACE_STDERR", false) {
                    eprintln!("{line}");
                }
            }
        }
    });
}

pub(super) fn clear_mlx_cache() {
    let _ = unsafe { mlx_sys::mlx_clear_cache() };
    mlx_rs::transforms::compile::clear_cache();
}

pub(super) fn log_memory_sample(request_label: &str, phase: &str, details: &str) {
    let snapshot = memory_snapshot();
    let detail_prefix = if details.is_empty() { "" } else { " " };
    let line = format!(
        "ferrite_memory request={request_label} phase={phase} rss_bytes={} mlx_active_bytes={} mlx_cache_bytes={} mlx_peak_bytes={}",
        optional_u64(snapshot.rss_bytes),
        optional_usize(snapshot.mlx_active_bytes),
        optional_usize(snapshot.mlx_cache_bytes),
        optional_usize(snapshot.mlx_peak_bytes),
    );
    let line = format!("{line}{detail_prefix}{details}");
    if let Err(err) = append_memory_log_line(&line) {
        eprintln!("ferrite_memory_log_error error={err}");
    }
    if env_flag("MTPLX_MEMORY_TRACE_STDERR", false) {
        eprintln!("{line}");
    }
}

fn append_memory_log_line(line: &str) -> std::io::Result<()> {
    append_memory_log_line_to(&memory_log_path(), line)
}

pub(super) fn append_memory_log_line_to(path: &Path, line: &str) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{line}")
}

fn memory_log_path() -> PathBuf {
    ferrite_env_var("MTPLX_MEMORY_LOG")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ferrite-memory.log"))
}

pub(super) fn request_memory_details(
    request: &InferenceRequest,
    prompt_tokens: Option<usize>,
    max_tokens: Option<u32>,
) -> String {
    format!(
        "model={} requested_num_ctx={} prompt_tokens={} max_tokens={} request_max_tokens={} prefix_cache={}",
        request.model,
        request
            .requested_context_tokens
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        prompt_tokens
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        max_tokens
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        request
            .max_tokens
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string()),
        if env_flag("MTPLX_PREFIX_CACHE", false) {
            "enabled"
        } else {
            "disabled"
        }
    )
}

pub(super) fn check_kill_switch(phase: &str) -> Result<()> {
    if kill_switch_path().exists() {
        log_memory_sample("kill_switch", phase, "reason=file");
        anyhow::bail!(
            "Ferrite kill switch is active; remove {} to resume",
            kill_switch_path().display()
        );
    }
    if let Some(limit) = rss_kill_limit_bytes()
        && let Some(rss) = process_rss_bytes()
        && rss > limit
    {
        log_memory_sample(
            "kill_switch",
            phase,
            &format!("reason=rss rss_bytes={rss} limit_bytes={limit}"),
        );
        clear_mlx_cache();
        anyhow::bail!("Ferrite RSS kill switch tripped: rss={rss} limit={limit}");
    }
    if let Some(limit) = mlx_active_kill_limit_bytes() {
        let active = mlx_memory_value(mlx_sys::mlx_get_active_memory);
        if let Some(active) = active
            && active > limit
        {
            log_memory_sample(
                "kill_switch",
                phase,
                &format!("reason=mlx_active mlx_active_bytes={active} limit_bytes={limit}"),
            );
            clear_mlx_cache();
            anyhow::bail!(
                "Ferrite MLX active-memory kill switch tripped: active={active} limit={limit}"
            );
        }
    }
    Ok(())
}

fn memory_watchdog_loop(request_label: &'static str, alive: Arc<AtomicBool>, interval: Duration) {
    while alive.load(Ordering::Relaxed) {
        if kill_switch_path().exists() {
            hard_exit_from_watchdog(request_label, "file", None, None, None, None);
        }
        if let Some(limit) = rss_kill_limit_bytes()
            && let Some(rss) = process_rss_bytes()
            && rss > limit
        {
            hard_exit_from_watchdog(request_label, "rss", Some(rss), Some(limit), None, None);
        }
        if let Some(limit) = mlx_active_kill_limit_bytes()
            && let Some(active) = mlx_memory_value(mlx_sys::mlx_get_active_memory)
            && active > limit
        {
            hard_exit_from_watchdog(
                request_label,
                "mlx_active",
                None,
                None,
                Some(active),
                Some(limit),
            );
        }
        thread::sleep(interval);
    }
}

fn hard_exit_from_watchdog(
    request_label: &'static str,
    reason: &str,
    rss_bytes: Option<u64>,
    rss_limit_bytes: Option<u64>,
    mlx_active_bytes: Option<usize>,
    mlx_active_limit_bytes: Option<usize>,
) -> ! {
    let snapshot = memory_snapshot();
    let line = format!(
        "ferrite_memory request={request_label} phase=watchdog_hard_exit reason={reason} rss_bytes={} rss_limit_bytes={} mlx_active_bytes={} mlx_active_limit_bytes={} mlx_cache_bytes={} mlx_peak_bytes={}",
        rss_bytes
            .or(snapshot.rss_bytes)
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        rss_limit_bytes
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        mlx_active_bytes
            .or(snapshot.mlx_active_bytes)
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        mlx_active_limit_bytes
            .map(|value| value.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        optional_usize(snapshot.mlx_cache_bytes),
        optional_usize(snapshot.mlx_peak_bytes),
    );
    let _ = append_memory_log_line(&line);
    unsafe { libc::_exit(137) }
}

fn memory_watchdog_interval() -> Duration {
    let millis = ferrite_env_var("MTPLX_MEMORY_WATCHDOG_INTERVAL_MS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(250);
    Duration::from_millis(millis)
}

fn kill_switch_path() -> PathBuf {
    ferrite_env_var("MTPLX_KILL_SWITCH_FILE")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("ferrite-kill-switch"))
}

fn rss_kill_limit_bytes() -> Option<u64> {
    match ferrite_env_var("MTPLX_RSS_KILL_BYTES") {
        Ok(value) => value.trim().parse::<u64>().ok().filter(|value| *value > 0),
        Err(_) => physical_memory_bytes()
            .map(|bytes| bytes.saturating_mul(DEFAULT_RSS_KILL_PERCENT) / 100),
    }
}

fn mlx_active_kill_limit_bytes() -> Option<usize> {
    match ferrite_env_var("MTPLX_MLX_ACTIVE_KILL_BYTES") {
        Ok(value) => value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0),
        Err(_) => physical_memory_bytes()
            .and_then(|bytes| bytes.checked_mul(DEFAULT_MLX_ACTIVE_KILL_PERCENT))
            .and_then(|bytes| usize::try_from(bytes / 100).ok()),
    }
}

fn physical_memory_bytes() -> Option<u64> {
    let mut value = 0_u64;
    let mut size = std::mem::size_of::<u64>();
    let name = c"hw.memsize";
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&mut value as *mut u64).cast::<libc::c_void>(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (status == 0 && value > 0).then_some(value)
}

fn memory_snapshot() -> MemorySnapshot {
    MemorySnapshot {
        rss_bytes: process_rss_bytes(),
        mlx_active_bytes: mlx_memory_value(mlx_sys::mlx_get_active_memory),
        mlx_cache_bytes: mlx_memory_value(mlx_sys::mlx_get_cache_memory),
        mlx_peak_bytes: mlx_memory_value(mlx_sys::mlx_get_peak_memory),
    }
}

fn mlx_memory_value(call: unsafe extern "C" fn(*mut usize) -> i32) -> Option<usize> {
    let mut value = 0_usize;
    let status = unsafe { call(&mut value) };
    (status == 0).then_some(value)
}

fn process_rss_bytes() -> Option<u64> {
    const PROC_PIDTASKINFO: libc::c_int = 4;
    #[repr(C)]
    #[derive(Default)]
    struct ProcTaskInfo {
        virtual_size: u64,
        resident_size: u64,
        total_user: u64,
        total_system: u64,
        threads_user: u64,
        threads_system: u64,
        policy: i32,
        faults: i32,
        pageins: i32,
        cow_faults: i32,
        messages_sent: i32,
        messages_received: i32,
        syscalls_mach: i32,
        syscalls_unix: i32,
        csw: i32,
        threadnum: i32,
        numrunning: i32,
        priority: i32,
    }

    unsafe extern "C" {
        fn proc_pidinfo(
            pid: libc::c_int,
            flavor: libc::c_int,
            arg: u64,
            buffer: *mut libc::c_void,
            buffersize: libc::c_int,
        ) -> libc::c_int;
    }

    let mut info = ProcTaskInfo::default();
    let size = std::mem::size_of::<ProcTaskInfo>();
    let written = unsafe {
        proc_pidinfo(
            std::process::id() as libc::c_int,
            PROC_PIDTASKINFO,
            0,
            (&mut info as *mut ProcTaskInfo).cast::<libc::c_void>(),
            size as libc::c_int,
        )
    };
    (written as usize == size).then_some(info.resident_size)
}

fn optional_usize(value: Option<usize>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
