package com.splats.app.telemetry

// ─── Structured Error Hierarchy ───────────────────────────────────────────────

sealed class TelemetryError(message: String) : Exception(message) {

    /** CSV file could not be found at the supplied path. */
    data class CsvNotFound(val path: String)
        : TelemetryError("CSV file not found: $path")

    /** Telemetry logs must be CSV; other formats are not supported. */
    data class UnsupportedFormat(val extension: String)
        : TelemetryError("Unsupported telemetry format: $extension. Please use CSV.")

    /** MP4 video file could not be found at the supplied path. */
    data class VideoNotFound(val path: String)
        : TelemetryError("Video file not found: $path")

    /**
     * CSV was present but contained fewer than 10 valid rows after parsing
     * and row-level validation. Also raised if both DJI and Litchi column
     * maps fail (vendor-mismatch fallback exhausted).
     */
    data class InsufficientRecords(val count: Int)
        : TelemetryError("Only $count valid rows found — minimum 10 required")

    /** Parsed rows exist but none carry a 3D GPS fix (fixType ≥ 3). */
    object NoGpsFix
        : TelemetryError("No rows with a 3D GPS fix found in telemetry log")

    /** No row with HDOP ≤ 3.0 exists; ENU origin cannot be established. */
    object NoValidOrigin
        : TelemetryError("No row with HDOP ≤ 3.0 found — cannot establish ENU origin")

    /**
     * More than 20 % of rows were rejected by the row validator.
     * [percent] is the rejection fraction (0–100).
     */
    data class ExcessiveRejections(val percent: Double)
        : TelemetryError("${percent.format(1)}% of rows rejected — exceeds 20% threshold")

    /**
     * Cross-correlation between the telemetry altitude curve and video motion
     * proxy fell below 0.6. [offsetSeconds] is the offset at peak correlation.
     */
    data class SyncFailure(val offsetSeconds: Double)
        : TelemetryError("Time sync correlation too low at offset ${offsetSeconds}s")

    /**
     * Module wall-clock budget expired before completion.
     * [elapsedSeconds] is how long processing ran before cancellation.
     */
    data class Timeout(val elapsedSeconds: Double)
        : TelemetryError("Processing timed out after ${elapsedSeconds}s")

    /** Unexpected internal failure not covered by the above categories. */
    data class InternalError(val reason: String)
        : TelemetryError("Internal error: $reason")
}

// ─── Private helpers ──────────────────────────────────────────────────────────

private fun Double.format(decimals: Int) = "%.${decimals}f".format(this)
