package com.splats.app.telemetry

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import com.google.gson.GsonBuilder

// ─── PoseStampSequence Emitter ────────────────────────────────────────────────

/**
 * Stage 9 — Serialisation.
 *
 * Writes two output files:
 *  1. `<sessionId>.posestamps`      — binary, little-endian (spec §5.4)
 *  2. `<sessionId>.posestamps.json` — human-readable JSON sidecar
 *
 * Binary layout:
 * ┌──────────────────────────────────────────┐
 * │ Offset  Size  Field                       │
 * │ 0–3      4    Magic: 0x504F5354 ("POST")  │
 * │ 4–7      4    Version: Int32 = 1          │
 * │ 8–15     8    Record count: Int64         │
 * │ 16–87   72    EnuOrigin struct            │
 * │ 88–95    8    timeOffsetUs: Int64         │
 * │ 96+     96    PoseStamp records (packed)  │
 * └──────────────────────────────────────────┘
 *
 * Each PoseStamp record is 96 bytes (see [PoseStamp.toBytes]).
 */
internal object PoseStampEmitter {

    private const val MAGIC   = 0x504F5354   // "POST"
    private const val VERSION = 1

    // Header size: 4 (magic) + 4 (version) + 8 (count) + 72 (origin) + 8 (offset) = 96 bytes
    private const val HEADER_SIZE = 96
    private const val RECORD_SIZE = 96

    fun emit(
        sequence:  PoseStampSequence,
        outputDir: File,
        sessionId: String
    ): Pair<File, File> {
        outputDir.mkdirs()

        val binaryFile = File(outputDir, "$sessionId.posestamps")
        val jsonFile   = File(outputDir, "$sessionId.posestamps.json")

        writeBinary(sequence, binaryFile)
        writeJson(sequence, jsonFile)

        return Pair(binaryFile, jsonFile)
    }

    // ── Binary writer ─────────────────────────────────────────────────────────

    private fun writeBinary(seq: PoseStampSequence, file: File) {
        val totalSize = HEADER_SIZE + seq.records.size * RECORD_SIZE
        val buf = ByteBuffer.allocate(totalSize).order(ByteOrder.LITTLE_ENDIAN)

        // Header
        buf.putInt(MAGIC)
        buf.putInt(VERSION)
        buf.putLong(seq.records.size.toLong())
        writeOrigin(buf, seq.origin)
        buf.putLong(seq.timeOffsetUs)

        // Records
        seq.records.forEach { buf.put(it.toBytes()) }

        file.writeBytes(buf.array())
    }

    /** Writes the 72-byte EnuOrigin block. */
    private fun writeOrigin(buf: ByteBuffer, o: EnuOrigin) {
        // 72 bytes: 4 × Double (32) + 1 × Long (8) + padding (32)
        buf.putDouble(o.lat0)
        buf.putDouble(o.lon0)
        buf.putDouble(o.alt0)
        buf.putDouble(o.cosLat0)
        buf.putLong(o.timestampUs)
        // 32 bytes padding to reach 72 bytes total
        repeat(4) { buf.putLong(0L) }
    }

    // ── JSON sidecar ──────────────────────────────────────────────────────────

    private fun writeJson(seq: PoseStampSequence, file: File) {
        val gson = GsonBuilder().setPrettyPrinting().create()
        file.writeText(gson.toJson(seq))
    }
}

// ─── PoseStamp binary serialisation extension ─────────────────────────────────

/**
 * Serialises a single [PoseStamp] to its 96-byte little-endian binary representation.
 *
 * Layout:
 * ┌────────┬──────┬────────────────────────┐
 * │ Offset │ Size │ Field                  │
 * ├────────┼──────┼────────────────────────┤
 * │  0     │  4   │ frameIndex   (Int32)   │
 * │  4     │  8   │ ptsUs        (Int64)   │
 * │ 12     │  8   │ enuE         (Double)  │
 * │ 20     │  8   │ enuN         (Double)  │
 * │ 28     │  8   │ enuU         (Double)  │
 * │ 36     │  8   │ headingDeg   (Double)  │
 * │ 44     │  8   │ gimbalPitch  (Double)  │
 * │ 52     │  8   │ velE         (Double)  │
 * │ 60     │  8   │ velN         (Double)  │
 * │ 68     │  8   │ velU         (Double)  │
 * │ 76     │  8   │ hdop         (Double)  │
 * │ 84     │  8   │ covPosition  (Double)  │
 * │ 92     │  2   │ flags        (Int16)   │
 * │ 94     │  2   │ padding      (0x0000)  │
 * └────────┴──────┴────────────────────────┘
 */
fun PoseStamp.toBytes(): ByteArray {
    val buf = ByteBuffer.allocate(96).order(ByteOrder.LITTLE_ENDIAN)
    buf.putInt(frameIndex)
    buf.putLong(ptsUs)
    buf.putDouble(enuE)
    buf.putDouble(enuN)
    buf.putDouble(enuU)
    buf.putDouble(headingDeg)
    buf.putDouble(gimbalPitch)
    buf.putDouble(velE)
    buf.putDouble(velN)
    buf.putDouble(velU)
    buf.putDouble(hdop)
    buf.putDouble(covPosition)
    buf.putShort(flags.toShort())
    buf.putShort(0x0000)
    return buf.array()
}
