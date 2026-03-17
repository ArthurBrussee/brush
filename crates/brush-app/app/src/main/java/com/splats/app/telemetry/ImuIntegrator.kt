package com.splats.app.telemetry

import kotlin.math.abs
import kotlin.math.pow

// ─── IMU Integrator ───────────────────────────────────────────────────────────

/**
 * Stage 5 — Post-processing of velocity and heading.
 *
 * Operations (in order):
 *  1. Heading unwrapping — resolves 358° → 2° discontinuities before
 *     interpolation then re-wraps to [0°, 360°) afterwards.
 *  2. Velocity smoothing — 5-sample causal median filter on velE/velN/velU
 *     to suppress GPS jitter.
 *  3. IMU gap detection — consecutive records with timestamp gap > 200 ms
 *     are flagged with [QualityFlag.IMU_GAP].
 *
 * Returns a new list of [TelRecord] (immutable; originals untouched) and a
 * parallel [IntArray] of quality-flag bitmasks (one entry per record).
 */
internal object ImuIntegrator {

    private const val IMU_GAP_THRESHOLD_US        = 200_000L  // 200 ms in microseconds
    private const val MEDIAN_WINDOW               = 5
    private const val COMPASS_SPIKE_DEG           = 45.0      // §6.2
    private const val COMPASS_SPEED_THRESHOLD_MS  = 2.0       // m/s

    data class Result(
        val records:        List<TelRecord>,
        val flags:          IntArray,           // same length as records
        val compassWarning: Boolean             // true if > 10% of headings repaired
    )

    fun process(records: List<TelRecord>): Result {
        if (records.isEmpty()) return Result(emptyList(), IntArray(0), false)

        // 1. Unwrap headings to a continuous (unbounded) sequence.
        val unwrapped = unwrapHeadings(records.map { it.headingDeg })

        // 2. Smooth velocities with a 5-sample causal median filter.
        val smoothE = medianFilter(records.map { it.velE })
        val smoothN = medianFilter(records.map { it.velN })
        val smoothU = medianFilter(records.map { it.velU })

        // 3. Build output records, detect IMU gaps, and repair compass interference.
        val flags    = IntArray(records.size) { QualityFlag.CLEAN }
        val updated  = records.mapIndexed { i, rec ->
            if (i > 0) {
                val gapUs = rec.timestampUs - records[i - 1].timestampUs
                if (gapUs > IMU_GAP_THRESHOLD_US) {
                    flags[i] = flags[i] or QualityFlag.IMU_GAP
                }
            }
            rec.copy(
                headingDeg = wrapTo360(unwrapped[i]),
                velE       = smoothE[i],
                velN       = smoothN[i],
                velU       = smoothU[i]
            )
        }.toMutableList()

        // 4. Compass interference repair (spec §6.2).
        //    Symptom: heading change > 45° between consecutive records at speed < 2 m/s.
        //    Response: replace with linear interpolation from nearest clean records.
        //              Set HEADING_INTERPOLATED silently; warn if > 10% affected.
        val corruptIndices = mutableSetOf<Int>()
        for (i in 1 until updated.size) {
            val speed = Math.sqrt(
                updated[i].velE.pow(2) + updated[i].velN.pow(2) + updated[i].velU.pow(2)
            )
            val headingChange = yawDiffDeg(updated[i - 1].headingDeg, updated[i].headingDeg)
            if (headingChange > COMPASS_SPIKE_DEG && speed < COMPASS_SPEED_THRESHOLD_MS) {
                corruptIndices += i
                flags[i] = flags[i] or QualityFlag.HEADING_INTERPOLATED
            }
        }

        if (corruptIndices.isNotEmpty()) {
            repairCompassSpikes(updated, corruptIndices)
        }

        val compassWarning = corruptIndices.size.toDouble() / updated.size > 0.10

        return Result(updated, flags, compassWarning)
    }

    // ── Heading unwrap ────────────────────────────────────────────────────────

    /**
     * Convert a sequence of compass headings in [0°, 360°) to a continuous
     * unwrapped sequence (may exceed 360° or go negative).
     */
    private fun unwrapHeadings(headings: List<Double>): DoubleArray {
        val out = DoubleArray(headings.size)
        if (headings.isEmpty()) return out
        out[0] = headings[0]
        for (i in 1 until headings.size) {
            var diff = headings[i] - headings[i - 1]
            // Shortest-path correction
            diff = ((diff + 180.0) % 360.0 + 360.0) % 360.0 - 180.0
            out[i] = out[i - 1] + diff
        }
        return out
    }

    /** Wrap an unbounded angle back into [0°, 360°). */
    private fun wrapTo360(deg: Double): Double = ((deg % 360.0) + 360.0) % 360.0

    // ── Causal median filter ──────────────────────────────────────────────────

    /**
     * 5-sample causal median filter.  For indices < window−1, the available
     * prefix is used (minimum 1 sample).
     */
    private fun medianFilter(values: List<Double>): DoubleArray {
        val out = DoubleArray(values.size)
        for (i in values.indices) {
            val start  = maxOf(0, i - MEDIAN_WINDOW + 1)
            val window = values.subList(start, i + 1).sorted()
            out[i] = window[window.size / 2]
        }
        return out
    }

    // ── Compass spike repair ──────────────────────────────────────────────────

    /**
     * Replaces corrupt heading values (identified by [corruptIndices]) with
     * linear interpolation between the nearest clean records on either side.
     * Operates in-place on [records].
     */
    private fun repairCompassSpikes(
        records: MutableList<TelRecord>,
        corruptIndices: Set<Int>
    ) {
        // Walk through corrupt runs and replace each with interpolated heading.
        var i = 0
        while (i < records.size) {
            if (i !in corruptIndices) { i++; continue }
            // Find the run boundaries.
            val runStart = i
            var runEnd   = i
            while (runEnd + 1 < records.size && (runEnd + 1) in corruptIndices) runEnd++

            // Find nearest clean anchor on the left and right.
            val leftIdx  = (runStart - 1).coerceAtLeast(0)
            val rightIdx = (runEnd   + 1).coerceAtMost(records.size - 1)
            val h0       = records[leftIdx].headingDeg
            val h1       = records[rightIdx].headingDeg
            val span     = (runEnd - runStart + 2).toDouble()   // includes anchors

            for (j in runStart..runEnd) {
                val t           = (j - runStart + 1) / span
                val interpolated = slerpHeading(h0, h1, t)
                records[j]      = records[j].copy(headingDeg = interpolated)
            }
            i = runEnd + 1
        }
    }

    /** Shortest-path absolute yaw difference in degrees (0–180). */
    private fun yawDiffDeg(yaw1: Double, yaw2: Double): Double {
        var diff = (yaw2 - yaw1 + 360.0) % 360.0
        if (diff > 180.0) diff -= 360.0
        return abs(diff)
    }

    /** Shortest-path heading SLERP; result in [0°, 360°). */
    private fun slerpHeading(h1: Double, h2: Double, t: Double): Double {
        var diff = (h2 - h1 + 360.0) % 360.0
        if (diff > 180.0) diff -= 360.0
        return ((h1 + diff * t) + 360.0) % 360.0
    }
}
