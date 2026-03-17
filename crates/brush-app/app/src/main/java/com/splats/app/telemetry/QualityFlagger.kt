package com.splats.app.telemetry

// ─── Quality Flagger ──────────────────────────────────────────────────────────

/**
 * Stage 8 — Apply quality gates to each interpolated [PoseStamp].
 *
 * Gates (from spec Section 2.3):
 *
 * | Gate                  | Condition                             | Flag                | Action         |
 * |-----------------------|---------------------------------------|---------------------|----------------|
 * | GPS horizontal acc.   | HDOP > 5.0                            | POOR_GPS            | retain, warn   |
 * | GPS fix type          | fixType < 3D  (checked upstream)      | NO_FIX              | hard exclude   |
 * | IMU data gap          | gap > 200 ms  (set by ImuIntegrator)  | IMU_GAP             | retain, warn   |
 * | Velocity magnitude    | speed > 25 m/s                        | IMPLAUSIBLE_VELOCITY| hard exclude   |
 * | Gimbal pitch          | outside −95° to +30°                  | IMPLAUSIBLE_GIMBAL  | retain, warn   |
 * | Altitude              | enuU < 0 (negative barometric alt)    | NEGATIVE_ALTITUDE   | retain, warn   |
 *
 * Returns a [Result] that splits frames into [retained] (possibly with
 * warning flags) and [excluded] (hard-gate failures).
 */
internal object QualityFlagger {

    private const val HDOP_THRESHOLD         = 5.0
    private const val MAX_SPEED_MS           = 25.0
    private const val GIMBAL_PITCH_MIN       = -95.0
    private const val GIMBAL_PITCH_MAX       = 30.0
    private const val LARGE_SCENE_THRESHOLD  = 50_000.0  // 50 km in metres

    data class Result(
        val retained:  List<PoseStamp>,
        val excluded:  List<PoseStamp>,   // hard-gate failures removed from output
        val warnings:  List<String>
    )

    fun apply(poses: List<PoseStamp>, largeSceneWarning: Boolean = false): Result {
        val retained  = mutableListOf<PoseStamp>()
        val excluded  = mutableListOf<PoseStamp>()
        val warnings  = mutableListOf<String>()

        for (pose in poses) {
            var flags    = pose.flags
            var hardFail = false

            // ── Hard exclusions ───────────────────────────────────────────────

            // NO_FIX already set upstream; treat as hard exclusion here.
            if (flags and QualityFlag.NO_FIX != 0) {
                hardFail = true
            }

            // Implausible velocity
            val speed = speed(pose)
            if (speed > MAX_SPEED_MS) {
                flags    = flags or QualityFlag.IMPLAUSIBLE_VELOCITY
                hardFail = true
            }

            if (hardFail) {
                excluded += pose.copy(flags = flags)
                continue
            }

            // ── Soft warnings (retained with flag) ───────────────────────────

            if (pose.hdop > HDOP_THRESHOLD) {
                flags = flags or QualityFlag.POOR_GPS
            }
            if (pose.gimbalPitch < GIMBAL_PITCH_MIN || pose.gimbalPitch > GIMBAL_PITCH_MAX) {
                flags = flags or QualityFlag.IMPLAUSIBLE_GIMBAL
                warnings += "Frame ${pose.frameIndex}: gimbal pitch ${pose.gimbalPitch}° out of range."
            }
            if (pose.enuU < 0.0) {
                flags = flags or QualityFlag.NEGATIVE_ALTITUDE
                warnings += "Frame ${pose.frameIndex}: negative barometric altitude (${pose.enuU} m)."
            }

            if (largeSceneWarning) {
                flags = flags or QualityFlag.LARGE_SCENE_COORD
            }

            retained += pose.copy(flags = flags)
        }

        if (excluded.count { it.flags and QualityFlag.NO_FIX != 0 } > 0) {
            warnings += "${excluded.count { it.flags and QualityFlag.NO_FIX != 0 }} frame(s) excluded: no GPS fix."
        }
        if (excluded.count { it.flags and QualityFlag.IMPLAUSIBLE_VELOCITY != 0 } > 0) {
            warnings += "${excluded.count { it.flags and QualityFlag.IMPLAUSIBLE_VELOCITY != 0 }} frame(s) excluded: implausible velocity."
        }
        if (largeSceneWarning) {
            warnings += "Scene extent exceeds 50 km — consider UTM for survey-grade accuracy."
        }

        return Result(retained, excluded, warnings)
    }

    private fun speed(pose: PoseStamp): Double =
        Math.sqrt(pose.velE * pose.velE + pose.velN * pose.velN + pose.velU * pose.velU)
}
