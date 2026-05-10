use std::sync::OnceLock;

use super::super::{env_flag, env_i32};

const DEFAULT_SMALL_M_QMV4_M_VALUES: [bool; 7] = [false, true, false, false, false, true, true];

pub fn small_m_qmm4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("MTPLX_SMALL_M_QMM4", false))
}

pub fn small_m_qmv4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("FERRITE_SMALL_M_QMV4", true))
}

pub(super) fn small_m_qmv4_m_enabled(m: i32) -> bool {
    static VALUES: OnceLock<[bool; 7]> = OnceLock::new();
    let values = VALUES.get_or_init(|| {
        if let Ok(value) = std::env::var("FERRITE_SMALL_M_QMV4_M_VALUES") {
            return small_m_qmv4_m_values_from_str(&value).unwrap_or(DEFAULT_SMALL_M_QMV4_M_VALUES);
        }
        if std::env::var_os("FERRITE_SMALL_M_QMV4_MAX_M").is_some() {
            return small_m_qmv4_m_values_to_max(small_m_qmv4_max_m());
        }
        DEFAULT_SMALL_M_QMV4_M_VALUES
    });
    usize::try_from(m)
        .ok()
        .and_then(|index| values.get(index))
        .copied()
        .unwrap_or(false)
}

pub fn small_m_qmv4_m_values_from_str(value: &str) -> Option<[bool; 7]> {
    let normalized = value.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "all" | "true" | "yes" | "on") {
        return Some(small_m_qmv4_m_values_to_max(6));
    }
    if matches!(normalized.as_str(), "none" | "0" | "false" | "no" | "off") {
        return Some([false; 7]);
    }

    let mut values = [false; 7];
    let mut any = false;
    for part in normalized.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let m = part.parse::<usize>().ok()?;
        if !(1..=6).contains(&m) {
            return None;
        }
        values[m] = true;
        any = true;
    }
    any.then_some(values)
}

pub(super) fn small_m_qmv4_simdgroups() -> i32 {
    static SIMDGROUPS: OnceLock<i32> = OnceLock::new();
    *SIMDGROUPS
        .get_or_init(|| qmv4_simdgroups_or_default(env_i32("FERRITE_SMALL_M_QMV4_SIMDGROUPS", 8)))
}

pub(super) fn small_m_qmv4_packs_per_thread() -> i32 {
    static PACKS: OnceLock<i32> = OnceLock::new();
    *PACKS.get_or_init(|| {
        qmv4_packs_per_thread_or_default(env_i32("FERRITE_SMALL_M_QMV4_PACKS_PER_THREAD", 2))
    })
}

pub fn small_m_qmv4_strict() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("FERRITE_SMALL_M_QMV4_STRICT", false))
}

pub fn gate_up_swiglu_qmv4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("MTPLX_GATE_UP_SWIGLU_QMV4", false))
}

pub(super) fn multi3_qmv4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("MTPLX_MULTI3_QMV4", false))
}

fn small_m_qmv4_max_m() -> i32 {
    static MAX_M: OnceLock<i32> = OnceLock::new();
    *MAX_M.get_or_init(|| env_i32("FERRITE_SMALL_M_QMV4_MAX_M", 1).clamp(1, 6))
}

fn small_m_qmv4_m_values_to_max(max_m: i32) -> [bool; 7] {
    let mut values = [false; 7];
    for m in 1..=max_m.clamp(1, 6) {
        values[m as usize] = true;
    }
    values
}

pub(super) fn qmv4_simdgroups_or_default(value: i32) -> i32 {
    if matches!(value, 1 | 2 | 4 | 8) {
        value
    } else {
        8
    }
}

pub(super) fn qmv4_packs_per_thread_or_default(value: i32) -> i32 {
    if value == 2 { value } else { 2 }
}
