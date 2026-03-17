package com.splats.app.telemetry

import java.io.File

// ─── Raw Telemetry Row (internal, pre-ENU conversion) ────────────────────────

data class TelRow(
    val timestampUs: Long,      // microseconds since epoch (UTC)
    val lat:         Double,    // WGS84 latitude, degrees
    val lon:         Double,    // WGS84 longitude, degrees
    val altM:        Double,    // barometric altitude, metres AGL
    val headingDeg:  Double,    // compass heading, degrees CW from North [0, 360)
    val gimbalPitch: Double,    // degrees; nadir = -90
    val velN:        Double,    // velocity North, m/s
    val velE:        Double,    // velocity East, m/s
    val velD:        Double,    // velocity Down, m/s (positive = descending)
    val hdop:        Double,    // horizontal dilution of precision
    val fixType:     Int,       // 0=no fix, 2=2D, 3=3D, 4=3D+DGPS
    val satellites:  Int
)

// ─── ENU-converted, validated record ─────────────────────────────────────────

data class TelRecord(
    val timestampUs: Long,      // before time sync
    val tsAligned:   Long,      // after time sync offset applied
    val enuE:        Double,    // East metres from origin
    val enuN:        Double,    // North metres from origin
    val enuU:        Double,    // Up metres from origin
    val headingDeg:  Double,    // unwrapped, degrees CW from North
    val gimbalPitch: Double,    // degrees; nadir = -90
    val velE:        Double,    // m/s
    val velN:        Double,    // m/s
    val velU:        Double,    // m/s (positive = ascending)
    val hdop:        Double,
    val fixType:     Int
)

// ─── Per-keyframe output record ───────────────────────────────────────────────

data class PoseStamp(
    val frameIndex:  Int,       // index into the keyframe sequence
    val ptsUs:       Long,      // video PTS in microseconds
    val enuE:        Double,    // East metres
    val enuN:        Double,    // North metres
    val enuU:        Double,    // Up metres
    val headingDeg:  Double,    // degrees CW from North
    val gimbalPitch: Double,    // degrees
    val velE:        Double,    // m/s
    val velN:        Double,    // m/s
    val velU:        Double,    // m/s
    val hdop:        Double,
    val covPosition: Double,    // position uncertainty radius, metres
    val covHeading:  Double,    // heading uncertainty, degrees (2.0° fixed for DJI)
    val flags:       Int        // bitmask — see QualityFlag
)

// ─── Quality flags as bitmask constants ──────────────────────────────────────

object QualityFlag {
    const val CLEAN                = 0x0000
    const val POOR_GPS             = 0x0001
    const val NO_FIX               = 0x0002  // hard exclusion
    const val IMU_GAP              = 0x0004
    const val IMPLAUSIBLE_VELOCITY = 0x0008  // hard exclusion
    const val IMPLAUSIBLE_GIMBAL   = 0x0010
    const val NEGATIVE_ALTITUDE    = 0x0020
    const val EXTRAPOLATED         = 0x0040
    const val LARGE_SCENE_COORD    = 0x0080
    const val HEADING_INTERPOLATED = 0x0100
}

// ─── ENU origin anchor ───────────────────────────────────────────────────────

data class EnuOrigin(
    val lat0:        Double,    // WGS84 latitude of origin
    val lon0:        Double,    // WGS84 longitude of origin
    val alt0:        Double,    // barometric altitude of origin, metres
    val cosLat0:     Double,    // precomputed cos(lat0 × π/180)
    val timestampUs: Long       // timestamp of the anchor record
)

// ─── Final sequence output ────────────────────────────────────────────────────

data class PoseStampSequence(
    val origin:       EnuOrigin,
    val timeOffsetUs: Long,             // Δt from time sync, microseconds
    val records:      List<PoseStamp>,
    val sourceMode:   TelemetryMode,    // always MODE_C in this module
    val logPath:      File,
    val videoPath:    File,
    val createdAt:    Long              // System.currentTimeMillis()
)

enum class TelemetryMode { MODE_A, MODE_B, MODE_C }

// ─── Processing summary ───────────────────────────────────────────────────────

data class TelemetryProcessingReport(
    val totalCsvRows:     Int,
    val rejectedRows:     Int,
    val totalKeyframes:   Int,
    val outputFrames:     Int,          // after hard gate exclusions
    val excludedFrames:   Int,
    val flaggedFrames:    Int,          // soft warnings, still in output
    val syncOffsetMs:     Double,
    val syncCorrelation:  Double,       // 0.0–1.0; > 0.6 required
    val origin:           EnuOrigin,
    val processingTimeMs: Long,
    val warnings:         List<String>
)

// ─── Processing stages ────────────────────────────────────────────────────────

enum class ProcessingStage(val label: String) {
    PARSING("Parsing CSV"),
    CONVERTING("Converting coordinates"),
    SYNCING("Synchronising timelines"),
    INTERPOLATING("Interpolating poses"),
    FLAGGING("Applying quality gates"),
    EMITTING("Writing output")
}
