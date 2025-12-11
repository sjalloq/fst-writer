// Copyright 2024 Cornell University
// Copyright 2025 Shareef Jalloq
// released under BSD 3-Clause License
// Raw FST writer for creating filtered copies with minimal recompression.
//
// This module provides low-level writing capabilities that allow creating
// FST files by copying compressed value change data directly from source files.

use crate::io::{write_u8, write_u64, write_variant_u64};
use crate::{FstInfo, FstScopeType, FstVarDirection, FstVarType, Result};
use miniz_oxide::deflate::compress_to_vec_zlib;
use rustc_hash::FxHashMap;
use std::io::{Seek, SeekFrom, Write};

// ============================================================================
// Block type constants
// ============================================================================

const FST_BL_HEADER: u8 = 0;
const FST_BL_VCDATA: u8 = 1;
const FST_BL_GEOMETRY: u8 = 3;
const FST_BL_HIERARCHY_LZ4: u8 = 6;
#[allow(dead_code)]
const FST_BL_VCDATA_DYNAMICALIAS2: u8 = 8;

const ZLIB_LEVEL: u8 = 4;
const HEADER_LENGTH: u64 = 329;
const HEADER_VERSION_MAX_LEN: usize = 128;
const HEADER_DATE_MAX_LEN: usize = 119;
const DOUBLE_ENDIAN_TEST: f64 = std::f64::consts::E;

/// Compression type for value change data in VC blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VcPackType {
    Lz4,
    FastLz,
    Zlib,
}

// ============================================================================
// Low-level raw FST file writer
// ============================================================================

/// Low-level writer for creating FST files with raw block access.
///
/// This writer is designed for filtered copying where you want to:
/// - Write header/geometry/hierarchy blocks from scratch
/// - Write VC blocks with pre-compressed signal data (from another FST file)
///
/// For normal FST writing (from simulation), use `open_fst()` instead.
pub struct FstRawWriter<W: Write + Seek> {
    output: W,
    #[allow(dead_code)]
    start_time: u64,
    end_time: u64,
    num_vc_sections: u64,
    scope_count: u64,
    var_count: u64,
    num_signals: u64,
}

impl<W: Write + Seek> FstRawWriter<W> {
    /// Create a new raw FST writer.
    ///
    /// Writes the header block immediately. Call `finish()` when done to
    /// update the header with final values.
    pub fn new(mut output: W, info: &FstInfo) -> Result<Self> {
        // Write header block
        write_header(&mut output, info)?;

        Ok(Self {
            output,
            start_time: info.start_time,
            end_time: 0,
            num_vc_sections: 0,
            scope_count: 0,
            var_count: 0,
            num_signals: 0,
        })
    }

    /// Get mutable access to the underlying writer.
    pub fn inner_mut(&mut self) -> &mut W {
        &mut self.output
    }

    /// Get the current position in the output stream.
    pub fn position(&mut self) -> Result<u64> {
        Ok(self.output.stream_position()?)
    }

    /// Start writing a new VC block.
    ///
    /// Returns a `VcBlockWriter` that must be used to write the block contents.
    pub fn begin_vc_block(
        &mut self,
        start_time: u64,
        end_time: u64,
    ) -> Result<VcBlockWriter<'_, W>> {
        self.end_time = end_time;
        VcBlockWriter::new(&mut self.output, start_time, end_time)
    }

    /// Record that a VC block was written (call after VcBlockWriter::finish).
    pub fn vc_block_written(&mut self) {
        self.num_vc_sections += 1;
    }

    /// Write the geometry block.
    ///
    /// `signal_lengths` contains the bit width for each signal (0 for real signals).
    pub fn write_geometry(&mut self, signal_lengths: &[u32]) -> Result<()> {
        self.num_signals = signal_lengths.len() as u64;
        write_geometry_block(&mut self.output, signal_lengths)
    }

    /// Write the hierarchy block.
    ///
    /// `hierarchy_bytes` should be the uncompressed hierarchy data.
    pub fn write_hierarchy(&mut self, hierarchy_bytes: &[u8]) -> Result<()> {
        write_hierarchy_block(&mut self.output, hierarchy_bytes)
    }

    /// Set the scope and var counts (for header update).
    pub fn set_counts(&mut self, scope_count: u64, var_count: u64) {
        self.scope_count = scope_count;
        self.var_count = var_count;
    }

    /// Finish writing and update the header with final values.
    pub fn finish(mut self) -> Result<W> {
        update_header(
            &mut self.output,
            self.end_time,
            self.scope_count,
            self.var_count,
            self.num_signals,
            self.num_vc_sections,
        )?;
        Ok(self.output)
    }
}

// ============================================================================
// VC Block Writer
// ============================================================================

/// Writer for a single VC (Value Change) block.
///
/// Use this to write filtered VC blocks with pre-compressed signal data.
pub struct VcBlockWriter<'a, W: Write + Seek> {
    output: &'a mut W,
    block_start: u64,
}

impl<'a, W: Write + Seek> VcBlockWriter<'a, W> {
    /// Start writing a new VC block.
    pub fn new(output: &'a mut W, start_time: u64, end_time: u64) -> Result<Self> {
        // Write block type
        write_u8(output, FST_BL_VCDATA)?;

        // Remember position for section length (to be fixed later)
        let block_start = output.stream_position()?;

        // Write placeholder section length
        write_u64(output, 0)?;
        // Write start/end time
        write_u64(output, start_time)?;
        write_u64(output, end_time)?;
        // Write placeholder memory_required
        write_u64(output, 0)?;

        Ok(Self {
            output,
            block_start,
        })
    }

    /// Write the frame (initial values) for filtered signals.
    ///
    /// `filtered_frame` should contain the concatenated initial values for
    /// only the kept signals, in their new order.
    pub fn write_frame(&mut self, filtered_frame: &[u8], num_signals: usize) -> Result<()> {
        // Compress the frame with zlib
        let compressed = compress_to_vec_zlib(filtered_frame, ZLIB_LEVEL);

        // Use compressed if smaller, otherwise uncompressed
        let (data, uncomp_len, comp_len) = if compressed.len() < filtered_frame.len() {
            (&compressed[..], filtered_frame.len(), compressed.len())
        } else {
            (filtered_frame, filtered_frame.len(), 0)
        };

        // Write frame header: uncompressed_len, compressed_len, num_signals
        write_variant_u64(&mut self.output, uncomp_len as u64)?;
        write_variant_u64(&mut self.output, comp_len as u64)?;
        write_variant_u64(&mut self.output, num_signals as u64)?;

        // Write frame data
        self.output.write_all(data)?;

        Ok(())
    }

    /// Write the value change section header and return position for wave data.
    pub fn begin_waves(&mut self, num_signals: usize, pack_type: VcPackType) -> Result<u64> {
        // Write waves header
        write_variant_u64(&mut self.output, num_signals as u64)?;

        let pack_byte = match pack_type {
            VcPackType::Lz4 => b'4',
            VcPackType::FastLz => b'F',
            VcPackType::Zlib => b'Z',
        };
        write_u8(&mut self.output, pack_byte)?;

        Ok(self.output.stream_position()?)
    }

    /// Write raw wave data bytes directly (already compressed per-signal data).
    pub fn write_raw_wave_data(&mut self, data: &[u8]) -> Result<()> {
        self.output.write_all(data)?;
        Ok(())
    }

    /// Write the position table (offset table) for the kept signals.
    ///
    /// `signal_offsets` contains (has_data, offset_from_waves_start) for each kept signal.
    /// Signals without data should have has_data=false.
    /// Signals may share offsets (aliases) - these will be encoded as explicit aliases.
    ///
    /// Standard format position table encoding:
    /// - (zero_count << 1) | 0: run of signals with no data
    /// - (increment << 1) | 1: new offset = prev_offset + increment (increment > 0)
    /// - 0 followed by (alias_idx + 1): explicit alias to signal at alias_idx
    pub fn write_position_table(&mut self, signal_offsets: &[(bool, u64)]) -> Result<()> {
        let mut offset_bytes = Vec::new();
        let mut prev_offset: u64 = 0;
        let mut zero_count: u64 = 0;

        // Track which signal index first used each offset
        let mut offset_to_first_signal: FxHashMap<u64, usize> = FxHashMap::default();

        for (sig_idx, &(has_data, offset)) in signal_offsets.iter().enumerate() {
            if !has_data {
                zero_count += 1;
            } else {
                // Flush any pending zeros
                if zero_count > 0 {
                    write_variant_u64(&mut offset_bytes, zero_count << 1)?;
                    zero_count = 0;
                }

                if let Some(&first_sig) = offset_to_first_signal.get(&offset) {
                    // This offset was already used - write an explicit alias
                    // Format: 0 followed by (alias_idx + 1)
                    write_variant_u64(&mut offset_bytes, 0)?;
                    write_variant_u64(&mut offset_bytes, (first_sig + 1) as u64)?;
                } else {
                    // New offset - write increment
                    let increment = offset - prev_offset;
                    write_variant_u64(&mut offset_bytes, (increment << 1) | 1)?;
                    prev_offset = offset;
                    offset_to_first_signal.insert(offset, sig_idx);
                }
            }
        }

        // Flush final zeros
        if zero_count > 0 {
            write_variant_u64(&mut offset_bytes, zero_count << 1)?;
        }

        // Write position table data (raw, not compressed)
        self.output.write_all(&offset_bytes)?;

        // Write position table length AFTER the data
        write_u64(&mut self.output, offset_bytes.len() as u64)?;

        Ok(())
    }

    /// Write the time table by copying raw bytes from source.
    pub fn write_time_table_raw(
        &mut self,
        time_data: &[u8],
        uncompressed_len: u64,
        compressed_len: u64,
        time_count: u64,
    ) -> Result<()> {
        // Write compressed time data
        self.output.write_all(time_data)?;

        // Write footer: uncompressed_len, compressed_len, time_count
        write_u64(&mut self.output, uncompressed_len)?;
        write_u64(&mut self.output, compressed_len)?;
        write_u64(&mut self.output, time_count)?;

        Ok(())
    }

    /// Finish the block by fixing up the section length and memory_required.
    pub fn finish(self, memory_required: u64) -> Result<()> {
        let end_pos = self.output.stream_position()?;
        // section_length INCLUDES the 8-byte section_length field itself
        // This is how the reader works: it seeks (section_length - 8) to skip content
        let section_length = end_pos - self.block_start;

        // Fix section length
        self.output.seek(SeekFrom::Start(self.block_start))?;
        write_u64(self.output, section_length)?;

        // Fix memory_required (skip start_time and end_time)
        self.output.seek(SeekFrom::Current(2 * 8))?;
        write_u64(self.output, memory_required)?;

        // Seek back to end
        self.output.seek(SeekFrom::Start(end_pos))?;

        Ok(())
    }
}

// ============================================================================
// Frame extraction helper
// ============================================================================

/// Information about a signal needed for frame filtering.
#[derive(Debug, Clone, Copy)]
pub struct SignalGeometry {
    /// Byte length in frame (1 byte per bit for digital, 8 for real)
    pub frame_bytes: u32,
    /// Whether this is a real (floating point) signal
    pub is_real: bool,
}

/// Extract initial values for kept signals from a decompressed frame.
///
/// `geometries` is the geometry for ALL signals in the source file.
/// `kept_signals` is a list of source signal indices to keep.
/// `frame_data` is the decompressed frame from the source.
///
/// Returns the filtered frame containing only kept signals' initial values.
///
/// # Performance
///
/// This function pre-computes offset positions and pre-allocates the result
/// buffer based on estimated size. This optimization reduces Vec reallocations
/// when processing large files with many signals.
pub fn extract_filtered_frame(
    geometries: &[SignalGeometry],
    kept_signals: &[usize],
    frame_data: &[u8],
) -> Vec<u8> {
    // Pre-compute offset map for O(1) lookup of each signal's position.
    // This avoids re-computing cumulative offsets for each kept signal.
    let mut offsets = Vec::with_capacity(geometries.len());
    let mut offset = 0usize;
    for g in geometries {
        offsets.push(offset);
        offset += g.frame_bytes as usize;
    }

    // Pre-allocate result buffer based on average signal size to minimize
    // Vec reallocations during extraction.
    let estimated_size = if !kept_signals.is_empty() && !geometries.is_empty() {
        let avg_size = offset / geometries.len();
        kept_signals.len() * avg_size
    } else {
        0
    };
    let mut result = Vec::with_capacity(estimated_size);

    // Extract kept signals using pre-computed offsets
    for &sig_idx in kept_signals {
        if sig_idx < geometries.len() {
            let start = offsets[sig_idx];
            let len = geometries[sig_idx].frame_bytes as usize;
            if start + len <= frame_data.len() {
                result.extend_from_slice(&frame_data[start..start + len]);
            }
        }
    }

    result
}

// ============================================================================
// Internal helper functions
// ============================================================================

fn write_header<W: Write + Seek>(output: &mut W, info: &FstInfo) -> Result<()> {
    write_u8(output, FST_BL_HEADER)?;
    write_u64(output, HEADER_LENGTH)?;
    write_u64(output, 0)?; // start time is always zero
    write_u64(output, 0)?; // dummy end time
    output.write_all(&DOUBLE_ENDIAN_TEST.to_le_bytes())?;
    write_u64(output, 0)?; // memory used by writer
    write_u64(output, 0)?; // dummy scope count
    write_u64(output, 0)?; // dummy var count
    write_u64(output, 0)?; // dummy num signals
    write_u64(output, 0)?; // dummy num vc sections
    output.write_all(&[info.timescale_exponent as u8])?;
    write_c_str_fixed_length(output, &info.version, HEADER_VERSION_MAX_LEN)?;
    write_c_str_fixed_length(output, &info.date, HEADER_DATE_MAX_LEN)?;
    write_u8(output, info.file_type as u8)?;
    write_u64(output, info.start_time)?;
    Ok(())
}

fn update_header<W: Write + Seek>(
    output: &mut W,
    end_time: u64,
    scope_count: u64,
    var_count: u64,
    num_signals: u64,
    num_vc_sections: u64,
) -> Result<()> {
    // go to start of header + skip block type, length and start time
    output.seek(SeekFrom::Start(1 + 2 * 8))?;
    write_u64(output, end_time)?;
    // skip endian test + writer memory
    output.seek(SeekFrom::Current(2 * 8))?;
    write_u64(output, scope_count)?;
    write_u64(output, var_count)?;
    write_u64(output, num_signals)?;
    write_u64(output, num_vc_sections)?;
    Ok(())
}

fn write_geometry_block<W: Write + Seek>(output: &mut W, signal_lengths: &[u32]) -> Result<()> {
    write_u8(output, FST_BL_GEOMETRY)?;

    let start = output.stream_position()?;
    write_u64(output, 0)?; // dummy section length
    write_u64(output, 0)?; // dummy uncompressed section length
    write_u64(output, signal_lengths.len() as u64)?; // max_handle

    for &len in signal_lengths {
        // Convert to file format: 0 = real, otherwise bit width
        let file_val = if len == 0 { 0u64 } else { len as u64 };
        write_variant_u64(output, file_val)?;
    }

    let end = output.stream_position()?;
    let section_len = end - start;
    output.seek(SeekFrom::Start(start))?;
    write_u64(output, section_len)?;
    write_u64(output, section_len - 3 * 8)?; // uncompressed content length
    output.seek(SeekFrom::Start(end))?;

    Ok(())
}

fn write_hierarchy_block<W: Write + Seek>(output: &mut W, hierarchy_bytes: &[u8]) -> Result<()> {
    write_u8(output, FST_BL_HIERARCHY_LZ4)?;

    let start = output.stream_position()?;
    write_u64(output, 0)?; // dummy section length
    write_u64(output, hierarchy_bytes.len() as u64)?; // uncompressed length

    // Compress with LZ4
    let compressed = lz4_flex::compress(hierarchy_bytes);
    output.write_all(&compressed)?;

    let end = output.stream_position()?;
    output.seek(SeekFrom::Start(start))?;
    write_u64(output, end - start)?;
    output.seek(SeekFrom::Start(end))?;

    Ok(())
}

fn write_c_str_fixed_length<W: Write>(output: &mut W, value: &str, max_len: usize) -> Result<()> {
    let bytes = value.as_bytes();
    let write_len = bytes.len().min(max_len - 1);
    output.write_all(&bytes[..write_len])?;
    let zeros = vec![0u8; max_len - write_len];
    output.write_all(&zeros)?;
    Ok(())
}

// ============================================================================
// Hierarchy building helpers
// ============================================================================

const HIERARCHY_TPE_VCD_SCOPE: u8 = 254;
const HIERARCHY_TPE_VCD_UP_SCOPE: u8 = 255;

/// Builder for constructing hierarchy bytes.
pub struct HierarchyBuilder {
    data: Vec<u8>,
    scope_count: u64,
    var_count: u64,
}

impl HierarchyBuilder {
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            scope_count: 0,
            var_count: 0,
        }
    }

    pub fn scope(&mut self, name: &str, component: &str, tpe: FstScopeType) {
        self.data.push(HIERARCHY_TPE_VCD_SCOPE);
        self.data.push(tpe as u8);
        self.write_c_str(name);
        self.write_c_str(component);
        self.scope_count += 1;
    }

    pub fn up_scope(&mut self) {
        self.data.push(HIERARCHY_TPE_VCD_UP_SCOPE);
    }

    pub fn var(
        &mut self,
        name: &str,
        tpe: FstVarType,
        dir: FstVarDirection,
        length: u32,
        signal_id: u32,
    ) {
        self.data.push(tpe as u8);
        self.data.push(dir as u8);
        self.write_c_str(name);
        self.write_variant_u64(length as u64);
        self.write_variant_u64(signal_id as u64);
        self.var_count += 1;
    }

    pub fn finish(self) -> (Vec<u8>, u64, u64) {
        (self.data, self.scope_count, self.var_count)
    }

    fn write_c_str(&mut self, s: &str) {
        self.data.extend_from_slice(s.as_bytes());
        self.data.push(0);
    }

    fn write_variant_u64(&mut self, mut value: u64) {
        if value <= 0x7f {
            self.data.push(value as u8);
            return;
        }
        while value != 0 {
            let next_value = value >> 7;
            let mask: u8 = if next_value == 0 { 0 } else { 0x80 };
            self.data.push((value & 0x7f) as u8 | mask);
            value = next_value;
        }
    }
}

impl Default for HierarchyBuilder {
    fn default() -> Self {
        Self::new()
    }
}
