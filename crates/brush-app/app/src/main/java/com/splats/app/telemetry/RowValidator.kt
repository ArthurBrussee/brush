package com.splats.app.telemetry

// ─── Row Validator ────────────────────────────────────────────────────────────

/**
 * Stage 3 — Per-row validation.
 *
 * Rejects individual rows that violate range or ordering invariants.
 * If > 20 % of rows are rejected, raises [TelemetryError.ExcessiveRejections].
 * If < 10 valid rows remain, raises [TelemetryError.InsufficientRecords].
 * If no row has a 3D GPS fix (fixType ≥ 3), raises [TelemetryError.NoGpsFix].
 */
internal object RowValidator {

    fun validate(rows: List<TelRow>): List<TelRow> {
        if (rows.isEmpty()) throw TelemetryError.InsufficientRecords(0)

        val valid = mutableListOf<TelRow>()
        var prevTimestamp = Long.MIN_VALUE

        for (row in rows) {
            if (!isRowValid(row, prevTimestamp)) continue
            valid += row
            prevTimestamp = row.timestampUs
        }

        val rejectedCount = rows.size - valid.size
        val rejectedPct   = rejectedCount.toDouble() / rows.size * 100.0

        if (rejectedPct > 20.0) {
            throw TelemetryError.ExcessiveRejections(rejectedPct)
        }
        if (valid.size < 10) {
            throw TelemetryError.InsufficientRecords(valid.size)
        }
        if (valid.none { it.fixType >= 3 }) {
            throw TelemetryError.NoGpsFix
        }

        return valid
    }

    private fun isRowValid(row: TelRow, prevTimestamp: Long): Boolean {
        // Latitude and longitude range checks
        if (!row.lat.isFinite() || !row.lon.isFinite()) return false
        if (row.lat  < -90.0  || row.lat  > 90.0)  return false
        if (row.lon  < -180.0 || row.lon  > 180.0)  return false
        // Altitude plausibility (barometric AGL)
        if (!row.altM.isFinite()) return false
        if (row.altM < -50.0  || row.altM > 6000.0) return false
        // Timestamp must be positive and monotonically increasing
        if (row.timestampUs <= 0L)                   return false
        if (row.timestampUs <= prevTimestamp)         return false
        return true
    }
}
