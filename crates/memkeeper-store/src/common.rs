//! `common` helpers extracted from `lib.rs` (pure code movement).
//! Re-exported from the crate root so the public API is unchanged.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use memkeeper_core::{kind, scope, status};
use rusqlite::Connection;

use crate::{Error, Result, MAX_TAGS, MAX_TAG_CHARS};

pub(crate) static ID_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) fn sha256_text(value: &str) -> String {
    sha256_hex(value.as_bytes())
}

pub(crate) fn is_supported_status(value: &str) -> bool {
    matches!(
        value,
        status::ACTIVE
            | status::SUPERSEDED
            | status::CONFLICTED
            | status::TOMBSTONED
            | status::EXPIRED
    )
}

pub(crate) fn limit_i64(limit: usize) -> Result<i64> {
    i64::try_from(limit).map_err(|_| Error::InvalidRequest {
        message: "limit overflowed i64".to_string(),
    })
}

pub(crate) fn collect_rows<T, F>(rows: rusqlite::MappedRows<'_, F>) -> Result<Vec<T>>
where
    F: FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
{
    let mut values = Vec::new();
    for row in rows {
        values.push(row?);
    }
    Ok(values)
}

pub(crate) fn now_timestamp(connection: &Connection) -> Result<String> {
    connection
        .query_row("SELECT strftime('%Y-%m-%dT%H:%M:%fZ', 'now')", [], |row| {
            row.get(0)
        })
        .map_err(Into::into)
}

pub(crate) fn is_supported_scope(value: &str) -> bool {
    matches!(
        value,
        scope::GLOBAL | scope::WORKSPACE | scope::PROJECT | scope::SESSION | scope::CUSTOM
    )
}

pub(crate) fn is_supported_kind(value: &str) -> bool {
    matches!(
        value,
        kind::FACT
            | kind::DECISION
            | kind::PREFERENCE
            | kind::LESSON
            | kind::TASK
            | kind::ACTION
            | kind::CONTINUITY
            | kind::SUMMARY
            | kind::REFERENCE
            | kind::ENTITY
    )
}

pub(crate) fn normalized_tags(tags: &[String]) -> Result<Vec<String>> {
    if tags.len() > MAX_TAGS {
        return Err(Error::InvalidRequest {
            message: format!("at most {MAX_TAGS} tags are allowed"),
        });
    }
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::with_capacity(tags.len());
    for tag in tags {
        let trimmed = tag.trim();
        if trimmed.is_empty() || trimmed.chars().count() > MAX_TAG_CHARS {
            return Err(Error::InvalidRequest {
                message: format!("tags must be non-empty and at most {MAX_TAG_CHARS} characters"),
            });
        }
        if !seen.insert(trimmed.to_string()) {
            return Err(Error::InvalidRequest {
                message: format!("duplicate tag: {trimmed}"),
            });
        }
        normalized.push(trimmed.to_string());
    }
    Ok(normalized)
}

pub(crate) fn json_string_for_store(value: &str) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "\"\"".to_string())
}

pub(crate) fn string_array_json(values: &[String]) -> String {
    let mut output = String::from("[");
    for (index, value) in values.iter().enumerate() {
        if index > 0 {
            output.push(',');
        }
        output.push('"');
        for character in value.chars() {
            match character {
                '"' => output.push_str("\\\""),
                '\\' => output.push_str("\\\\"),
                '\n' => output.push_str("\\n"),
                '\r' => output.push_str("\\r"),
                '\t' => output.push_str("\\t"),
                character if character.is_control() => {
                    let _ = write!(output, "\\u{:04x}", character as u32);
                }
                character => output.push(character),
            }
        }
        output.push('"');
    }
    output.push(']');
    output
}

pub(crate) fn next_id(prefix: &str) -> String {
    let counter = ID_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!(
        "{prefix}_{:032x}_{:x}_{counter:x}",
        unique_nanos(),
        process::id()
    )
}

pub(crate) const SHA256_K: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

#[derive(Clone)]
pub(crate) struct Sha256 {
    h: [u32; 8],
    buffer: [u8; 64],
    buffer_len: usize,
    total_len: u128,
}

impl Sha256 {
    pub(crate) const fn new() -> Self {
        Self {
            h: [
                0x6a09_e667,
                0xbb67_ae85,
                0x3c6e_f372,
                0xa54f_f53a,
                0x510e_527f,
                0x9b05_688c,
                0x1f83_d9ab,
                0x5be0_cd19,
            ],
            buffer: [0; 64],
            buffer_len: 0,
            total_len: 0,
        }
    }

    pub(crate) fn update(&mut self, mut input: &[u8]) {
        self.total_len = self.total_len.saturating_add(input.len() as u128);
        if self.buffer_len > 0 {
            let needed = 64 - self.buffer_len;
            let take = needed.min(input.len());
            self.buffer[self.buffer_len..self.buffer_len + take].copy_from_slice(&input[..take]);
            self.buffer_len += take;
            input = &input[take..];
            if self.buffer_len == 64 {
                sha256_compress(&mut self.h, &self.buffer);
                self.buffer_len = 0;
            }
        }
        for chunk in input.chunks_exact(64) {
            sha256_compress(&mut self.h, chunk);
        }
        let remainder = input.len() % 64;
        if remainder > 0 {
            let start = input.len() - remainder;
            self.buffer[..remainder].copy_from_slice(&input[start..]);
            self.buffer_len = remainder;
        }
    }

    pub(crate) fn finish_hex(mut self) -> String {
        let masked_bit_len = self.total_len.wrapping_mul(8) & u128::from(u64::MAX);
        let bit_len = u64::try_from(masked_bit_len).unwrap_or(0);
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;
        if self.buffer_len > 56 {
            self.buffer[self.buffer_len..].fill(0);
            sha256_compress(&mut self.h, &self.buffer);
            self.buffer_len = 0;
        }
        self.buffer[self.buffer_len..56].fill(0);
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());
        sha256_compress(&mut self.h, &self.buffer);

        let mut output = String::with_capacity(64);
        for word in self.h {
            let _ = write!(output, "{word:08x}");
        }
        output
    }
}

#[allow(clippy::many_single_char_names)]
pub(crate) fn sha256_compress(h: &mut [u32; 8], chunk: &[u8]) {
    debug_assert_eq!(chunk.len(), 64);
    let mut w = [0u32; 64];
    for (index, word) in w.iter_mut().take(16).enumerate() {
        let offset = index * 4;
        *word = u32::from_be_bytes([
            chunk[offset],
            chunk[offset + 1],
            chunk[offset + 2],
            chunk[offset + 3],
        ]);
    }
    for index in 16..64 {
        let s0 =
            w[index - 15].rotate_right(7) ^ w[index - 15].rotate_right(18) ^ (w[index - 15] >> 3);
        let s1 =
            w[index - 2].rotate_right(17) ^ w[index - 2].rotate_right(19) ^ (w[index - 2] >> 10);
        w[index] = w[index - 16]
            .wrapping_add(s0)
            .wrapping_add(w[index - 7])
            .wrapping_add(s1);
    }

    let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
    for index in 0..64 {
        let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
        let ch = (e & f) ^ ((!e) & g);
        let temp1 = hh
            .wrapping_add(s1)
            .wrapping_add(ch)
            .wrapping_add(SHA256_K[index])
            .wrapping_add(w[index]);
        let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
        let maj = (a & b) ^ (a & c) ^ (b & c);
        let temp2 = s0.wrapping_add(maj);
        hh = g;
        g = f;
        f = e;
        e = d.wrapping_add(temp1);
        d = c;
        c = b;
        b = a;
        a = temp1.wrapping_add(temp2);
    }

    h[0] = h[0].wrapping_add(a);
    h[1] = h[1].wrapping_add(b);
    h[2] = h[2].wrapping_add(c);
    h[3] = h[3].wrapping_add(d);
    h[4] = h[4].wrapping_add(e);
    h[5] = h[5].wrapping_add(f);
    h[6] = h[6].wrapping_add(g);
    h[7] = h[7].wrapping_add(hh);
}

pub(crate) fn sha256_hex(input: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input);
    hasher.finish_hex()
}

pub(crate) fn sha256_path(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher.finish_hex())
}

pub(crate) fn unique_nanos() -> u128 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |duration| duration.as_nanos())
}
