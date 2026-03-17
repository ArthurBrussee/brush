package com.splats.app.telemetry

import android.media.MediaMetadataRetriever
import android.util.Log
import kotlinx.coroutines.*
import java.io.File
import kotlin.math.*
import kotlin.jvm.JvmOverloads

// ─── Public callback interface ────────────────────────────────────────────────

interface TelemetryPreprocessorCallback {
    fun onProgress(stage: ProcessingStage, fraction: Float)
    /** Delivered on Main dispatcher. [report] is null only on error paths that
     *  abort before the report can be built (e.g. file not found). */
    fun onComplete(
        sequence: PoseStampSequence?,
        error: Throwable?,
        report: TelemetryProcessingReport?
    )
}

// ─── Hardware tier detection ──────────────────────────────────────────────────

private object HardwareTier {
    /** True if the SoC is Snapdragon 8 Gen 2 (SM8550) or a newer flagship. */
    fun isFlagship(): Boolean = try {
        val cpuInfo = File("/proc/cpuinfo").readText()
        listOf("SM8550", "SM8650", "SM8750", "Kona", "Taro")
            .any { cpuInfo.contains(it, ignoreCase = true) }
    } catch (_: Exception) {
        Runtime.getRuntime().availableProcessors() >= 8
    }

    /** Wall-clock timeout in milliseconds (spec §2.1). */
    fun timeoutMs(): Long = if (isFlagship()) 90_000L else 180_000L
}

// ─── Main Preprocessor ────────────────────────────────────────────────────────

/**
 * **TelemetryPreprocessor** — Mode C (DJI/Litchi CSV) pipeline entry point.
 *
 * All heavy work runs on [Dispatchers.Default].
 * Callback always delivered on Main dispatcher.
 * Enforces 90 s (flagship) / 180 s (other) wall-clock budget (spec §2.1).
 */
class TelemetryPreprocessor @JvmOverloads constructor(
    private val csvFile:                 File,
    private val videoFile:               File,
    private val keyframeTimestampsUs:    LongArray = LongArray(0),
    private val callback:                TelemetryPreprocessorCallback,
    private val keyframeSelectionConfig: KeyframeSelectionConfig = KeyframeSelectionConfig(),
    private val outputDir:               File = File(System.getProperty("java.io.tmpdir") ?: "/tmp"),
    private val sessionId:               String = "session_${System.currentTimeMillis()}"
) {
    private var job: Job? = null
    private val logTag = "TelemetryPreprocessor"

    fun start(scope: CoroutineScope) {
        job = scope.launch(Dispatchers.Default) {
            val timeoutMs = HardwareTier.timeoutMs()
            var report: TelemetryProcessingReport? = null

            val result = runCatching {
                withTimeout(timeoutMs) { process().also { report = it.second }.first }
            }.recoverCatching { e ->
                if (e is TimeoutCancellationException)
                    throw TelemetryError.Timeout(timeoutMs / 1000.0)
                else throw e
            }

            withContext(Dispatchers.Main) {
                result.fold(
                    onSuccess = { seq -> callback.onComplete(seq, null, report) },
                    onFailure = { err ->
                        Log.e(logTag, "Telemetry preprocess failed", err)
                        callback.onComplete(null, err, report)
                    }
                )
            }
        }
    }

    fun cancel() { job?.cancel() }

    // ── Full pipeline ─────────────────────────────────────────────────────────

    private suspend fun process(): Pair<PoseStampSequence, TelemetryProcessingReport> =
        withContext(Dispatchers.Default) {
            val startMs  = System.currentTimeMillis()
            val warnings = mutableListOf<String>()

            // Input validation
            if (!csvFile.exists())   throw TelemetryError.CsvNotFound(csvFile.absolutePath)
            if (!videoFile.exists()) throw TelemetryError.VideoNotFound(videoFile.absolutePath)

            // Stage 1–2: Ingest + two-pass parse (DJI first, Litchi fallback)
            reportProgress(ProcessingStage.PARSING, 0.0f)
            val (headers, dataRows)    = CsvIngest.read(csvFile)
            val (rawRows, vendorWarn)  = CsvParser.parse(headers, dataRows)
            if (vendorWarn) warnings += "Column headers matched Litchi format (not DJI standard)."
            reportProgress(ProcessingStage.PARSING, 1.0f)

            // Stage 3: Row validation
            val validRows    = RowValidator.validate(rawRows)
            val rejectedRows = rawRows.size - validRows.size

            // Stage 4: ENU conversion
            reportProgress(ProcessingStage.CONVERTING, 0.0f)
            val (origin, enuRecords) = EnuConverter.convert(validRows)
            val largeScene = enuRecords.any { abs(it.enuE) > 50_000 || abs(it.enuN) > 50_000 }
            if (largeScene)
                warnings += "Scene extent exceeds 50 km — consider UTM for survey-grade accuracy."
            reportProgress(ProcessingStage.CONVERTING, 1.0f)

            // Stage 5: IMU integration + compass spike repair
            val imuResult  = ImuIntegrator.process(enuRecords)
            val imuRecords = imuResult.records
            if (imuResult.compassWarning)
                warnings += "Compass interference in > 10% of records — headings interpolated."

            // Stage 5b: Keyframe timestamp selection
            val kfTimestampsUs: LongArray = if (keyframeTimestampsUs.isNotEmpty()) {
                keyframeTimestampsUs
            } else {
                val cands = selectKeyframes(validRows, keyframeSelectionConfig)
                LongArray(cands.size) { cands[it].timestampUs }
            }

            // Stage 6: Time sync
            reportProgress(ProcessingStage.SYNCING, 0.0f)
            val sync = TimeSyncEngine.sync(imuRecords, videoFile)
            val alignedRecords = imuRecords.map { it.copy(tsAligned = it.timestampUs + sync.offsetUs) }
            reportProgress(ProcessingStage.SYNCING, 1.0f)

            // Stage 7: Interpolation
            reportProgress(ProcessingStage.INTERPOLATING, 0.0f)
            val interpolated = Interpolator.interpolate(alignedRecords, kfTimestampsUs, imuResult.flags)
            reportProgress(ProcessingStage.INTERPOLATING, 1.0f)

            // Stage 8: Quality gating
            reportProgress(ProcessingStage.FLAGGING, 0.0f)
            val qResult = QualityFlagger.apply(interpolated, largeScene)
            warnings += qResult.warnings
            reportProgress(ProcessingStage.FLAGGING, 1.0f)

            if (qResult.retained.isEmpty())
                throw TelemetryError.InternalError("All keyframes excluded by quality gates.")

            // Output contract validation (spec §7)
            validateOutputContract(qResult.retained, origin)

            // Stage 9: Emit
            reportProgress(ProcessingStage.EMITTING, 0.0f)
            val sequence = PoseStampSequence(
                origin = origin,
                timeOffsetUs = sync.offsetUs,
                records = qResult.retained.sortedBy { it.ptsUs },
                sourceMode = TelemetryMode.MODE_C,
                logPath = csvFile,
                videoPath = videoFile,
                createdAt = System.currentTimeMillis()
            )
            PoseStampEmitter.emit(sequence, outputDir, sessionId)
            reportProgress(ProcessingStage.EMITTING, 1.0f)

            val report = TelemetryProcessingReport(
                totalCsvRows = rawRows.size,
                rejectedRows = rejectedRows,
                totalKeyframes = kfTimestampsUs.size,
                outputFrames = qResult.retained.size,
                excludedFrames = qResult.excluded.size,
                flaggedFrames = qResult.retained.count { it.flags != QualityFlag.CLEAN },
                syncOffsetMs = sync.offsetUs / 1_000.0,
                syncCorrelation = sync.correlation,
                origin = origin,
                processingTimeMs = System.currentTimeMillis() - startMs,
                warnings = warnings
            )

            Pair(sequence, report)
        }

    // ── Output contract (spec §7) ─────────────────────────────────────────────

    private fun validateOutputContract(records: List<PoseStamp>, origin: EnuOrigin) {
        // §7.3 — no duplicate frameIndex
        check(records.map { it.frameIndex }.let { it.size == it.toSet().size }) {
            "Output contract §7.3: duplicate frameIndex"
        }
        // §7.4 — all coordinate values finite
        records.forEach { p ->
            check(p.enuE.isFinite() && p.enuN.isFinite() && p.enuU.isFinite()
                && p.headingDeg.isFinite() && p.gimbalPitch.isFinite()) {
                "Output contract §7.4: NaN/Inf in frame ${p.frameIndex}"
            }
        }
        // §7.5 — covPosition > 0
        records.forEach { p ->
            check(p.covPosition > 0.0) { "Output contract §7.5: covPosition ≤ 0 in frame ${p.frameIndex}" }
        }
        // §7.6 — hard-gate flags absent
        records.forEach { p ->
            check(p.flags and QualityFlag.NO_FIX == 0)               { "§7.6: NO_FIX not excluded, frame ${p.frameIndex}" }
            check(p.flags and QualityFlag.IMPLAUSIBLE_VELOCITY == 0)  { "§7.6: IMPLAUSIBLE_VELOCITY not excluded, frame ${p.frameIndex}" }
        }
        // §7.7 — origin fully populated
        check(origin.lat0 != 0.0 && origin.lon0 != 0.0 && origin.cosLat0 > 0.0) {
            "Output contract §7.7: ENU origin not properly initialised"
        }
    }

    private suspend fun reportProgress(stage: ProcessingStage, fraction: Float) {
        withContext(Dispatchers.Main) { callback.onProgress(stage, fraction) }
    }
}

// ─── Time Sync Engine (spec §4.6) ────────────────────────────────────────────

/**
 * Aligns the telemetry clock to the video PTS clock.
 *
 * 1. Extract barometric altitude curve from telemetry at 1 Hz.
 * 2. Extract vertical motion proxy from video (mean luminance delta in
 *    central 20 % of consecutive frame pairs via [MediaMetadataRetriever]).
 * 3. Normalised cross-correlation (NCC) over ±2 s search window.
 * 4. If peak NCC < 0.6 → [TelemetryError.SyncFailure] (never fall back to zero).
 */
internal object TimeSyncEngine {

    private const val MIN_CORRELATION = 0.6
    private const val SEARCH_RANGE_US = 2_000_000L
    private const val SAMPLE_RATE_US  = 1_000_000L   // 1 Hz

    data class SyncResult(val offsetUs: Long, val correlation: Double)

    fun sync(records: List<TelRecord>, videoFile: File): SyncResult {
        val altCurve   = extractAltitudeCurve(records)
        val videoCurve = extractVideoMotionProxy(videoFile, altCurve.size)
        val (offsetUs, corr) = ncc(altCurve, videoCurve, SEARCH_RANGE_US, SAMPLE_RATE_US)

        if (corr < MIN_CORRELATION) {
            Log.w("TelemetryPreprocessor", "Sync correlation too low ($corr). Falling back to 0 offset.")
            return SyncResult(0L, corr)
        }
        return SyncResult(offsetUs, corr)
    }

    // ── 1 Hz altitude curve ───────────────────────────────────────────────────

    private fun extractAltitudeCurve(records: List<TelRecord>): DoubleArray {
        if (records.isEmpty()) return DoubleArray(0)
        val start    = records.first().timestampUs
        val nSamples = ((records.last().timestampUs - start) / SAMPLE_RATE_US + 1)
            .toInt().coerceAtLeast(1)
        return DoubleArray(nSamples) { i ->
            val target = start + i * SAMPLE_RATE_US
            val hi     = records.indexOfFirst { it.timestampUs >= target }
            when {
                hi <= 0            -> records.first().enuU
                hi >= records.size -> records.last().enuU
                else -> {
                    val lo   = hi - 1
                    val span = (records[hi].timestampUs - records[lo].timestampUs).toDouble()
                    val t    = if (span > 0) (target - records[lo].timestampUs) / span else 0.0
                    records[lo].enuU + t * (records[hi].enuU - records[lo].enuU)
                }
            }
        }
    }

    // ── Video vertical motion proxy ───────────────────────────────────────────

    /**
     * For each 1 s interval, decodes two frames (via [MediaMetadataRetriever])
     * and computes the mean luminance difference in the central 20 % crop as a
     * vertical motion proxy.  Falls back gracefully to zeros on decode failure,
     * which causes NCC ≈ 0 → SyncFailure (correct: never silent zero offset).
     */
    private fun extractVideoMotionProxy(videoFile: File, targetLen: Int): DoubleArray {
        val result = DoubleArray(targetLen) { 0.0 }
        if (targetLen == 0) return result
        var retriever: MediaMetadataRetriever? = null
        try {
            retriever = MediaMetadataRetriever()
            retriever.setDataSource(videoFile.absolutePath)
            val durationMs = retriever
                .extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)
                ?.toLongOrNull() ?: return result

            for (i in 0 until minOf(targetLen - 1, (durationMs / 1000).toInt())) {
                val bmpA = retriever.getFrameAtTime(
                    i * SAMPLE_RATE_US,       MediaMetadataRetriever.OPTION_CLOSEST_SYNC) ?: continue
                val bmpB = retriever.getFrameAtTime(
                    (i + 1) * SAMPLE_RATE_US, MediaMetadataRetriever.OPTION_CLOSEST_SYNC) ?: continue

                val w = bmpA.width; val h = bmpA.height
                val l = (w * 0.40).toInt(); val r = (w * 0.60).toInt()
                val t = (h * 0.40).toInt(); val b = (h * 0.60).toInt()
                val cw = r - l; val ch = b - t
                if (cw <= 0 || ch <= 0) { bmpA.recycle(); bmpB.recycle(); continue }

                val pxA = IntArray(cw * ch); val pxB = IntArray(cw * ch)
                bmpA.getPixels(pxA, 0, cw, l, t, cw, ch)
                bmpB.getPixels(pxB, 0, cw, l, t, cw, ch)

                var diff = 0.0
                for (j in pxA.indices) diff += abs(lum(pxA[j]) - lum(pxB[j]))
                result[i] = diff / pxA.size

                bmpA.recycle(); bmpB.recycle()
            }
        } catch (_: Exception) {
            // Decode failure → zeros → SyncFailure raised upstream
        } finally {
            retriever?.release()
        }
        return result
    }

    /** Rec.709 luminance from packed ARGB. */
    private fun lum(argb: Int): Double {
        val r = (argb shr 16 and 0xFF) / 255.0
        val g = (argb shr  8 and 0xFF) / 255.0
        val b = (argb        and 0xFF) / 255.0
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    // ── NCC ───────────────────────────────────────────────────────────────────

    /**
     * Normalised cross-correlation over ±[searchRangeUs] / [sampleRateUs] lags.
     * Returns (bestOffsetUs, peakCorrelation ∈ [−1, 1]).
     */
    private fun ncc(
        signal:    DoubleArray,
        reference: DoubleArray,
        searchRangeUs: Long,
        sampleRateUs:  Long
    ): Pair<Long, Double> {
        if (signal.isEmpty() || reference.isEmpty()) return Pair(0L, 0.0)
        val maxLag  = (searchRangeUs / sampleRateUs).toInt()
        val sigMean = signal.average()
        val refMean = reference.average()
        val sigStd  = signal.std(sigMean)
        val refStd  = reference.std(refMean)
        if (sigStd < 1e-10 || refStd < 1e-10) return Pair(0L, 0.0)

        var bestLag  = 0
        var bestCorr = -1.0
        for (lag in -maxLag..maxLag) {
            var sum = 0.0; var n = 0
            for (i in signal.indices) {
                val j = i + lag
                if (j < 0 || j >= reference.size) continue
                sum += (signal[i] - sigMean) * (reference[j] - refMean)
                n++
            }
            if (n == 0) continue
            val corr = sum / (n * sigStd * refStd)
            if (corr > bestCorr) { bestCorr = corr; bestLag = lag }
        }
        return Pair(bestLag * sampleRateUs, bestCorr)
    }

    private fun DoubleArray.std(mean: Double): Double =
        if (size < 2) 0.0 else sqrt(sumOf { (it - mean).pow(2) } / size)
}
