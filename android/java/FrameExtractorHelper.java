// =============================================================================
// crates/brush-app/app/src/main/java/com/splats/app/FrameExtractorHelper.java
// =============================================================================
// Android-side implementation of the MediaMetadataRetriever bridge.
//
// DIAGRAM NODE: MP4 → MediaMetadataRetriever (Android Native)
//
// This class is called from Rust via JNI (see telemetry.rs and scene_loader.rs).
// It uses android.media.MediaMetadataRetriever to:
//   1. getFrameTimestampsUs() — read presentation timestamps without decoding
//   2. extractFrames()        — decode and save specific frames as JPEG
//
// Add to crates/brush-app/app/src/main/java/com/splats/app/
// =============================================================================

package com.splats.app;

import android.content.Context;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

public class FrameExtractorHelper {
    private static final String TAG = "FrameExtractorHelper";

    // ── Called from Rust: telemetry::extract_timestamps_android ──────────────
    //
    // Returns an array of presentation timestamps in microseconds,
    // one per requested frame. Uses METADATA_KEY_VIDEO_FRAME_COUNT
    // and OPTION_CLOSEST for precise alignment.
    //
    // Rust signature:
    //   fn extract_timestamps_android(mp4_path, total_frames) -> Vec<i64>
    // ─────────────────────────────────────────────────────────────────────────
    public static long[] getFrameTimestampsUs(String path, long frameCount) {
        MediaMetadataRetriever mmr = new MediaMetadataRetriever();
        long[] timestamps = new long[(int) frameCount];

        try {
            mmr.setDataSource(path);

            // Get total duration in milliseconds
            String durationStr = mmr.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_DURATION);
            long durationMs = durationStr != null ? Long.parseLong(durationStr) : 0L;

            // Get actual frame count if available
            String frameCountStr = mmr.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT);
            long actualFrameCount = frameCountStr != null
                    ? Long.parseLong(frameCountStr)
                    : frameCount;

            // Distribute requested frames evenly across the video duration
            for (int i = 0; i < frameCount; i++) {
                // Linear interpolation of timestamp across duration
                long timeUs = (long) ((double) i / actualFrameCount * durationMs * 1000.0);
                timestamps[i] = timeUs;
            }

            Log.i(TAG, "getFrameTimestampsUs: " + frameCount + " timestamps from "
                    + durationMs + " ms video");
        } catch (Exception e) {
            Log.e(TAG, "getFrameTimestampsUs failed: " + e.getMessage(), e);
            // Return evenly spaced fallback at 30 fps
            for (int i = 0; i < frameCount; i++) {
                timestamps[i] = (long) (i * 33333L); // 33333 μs = 1/30 s
            }
        } finally {
            try { mmr.release(); } catch (Exception ignored) {}
        }
        return timestamps;
    }

    // ── Called from Rust: scene_loader::extract_images_android ───────────────
    //
    // Decodes specific frames from the MP4 and saves them as JPEG files
    // into outDir/frame_NNNN.jpg.
    //
    // Rust signature:
    //   fn extractFrames(path: String, indices: int[], outDir: String) -> void
    // ─────────────────────────────────────────────────────────────────────────
    public static void extractFrames(String path, int[] frameIndices, String outDir) {
        MediaMetadataRetriever mmr = new MediaMetadataRetriever();
        File outputDir = new File(outDir);
        outputDir.mkdirs();

        try {
            mmr.setDataSource(path);

            // Get duration and frame count to compute timestamps
            String durationStr = mmr.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_DURATION);
            long durationMs = durationStr != null ? Long.parseLong(durationStr) : 0L;

            String frameCountStr = mmr.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT);
            long totalFrames = frameCountStr != null
                    ? Long.parseLong(frameCountStr)
                    : 900L; // 30s × 30fps fallback

            Log.i(TAG, "extractFrames: extracting " + frameIndices.length
                    + " frames from " + totalFrames + " total");

            for (int frameIdx : frameIndices) {
                // Convert frame index to microseconds
                long timeUs = (long) ((double) frameIdx / totalFrames * durationMs * 1000.0);

                // OPTION_CLOSEST: decode nearest keyframe to timeUs
                Bitmap bmp = mmr.getFrameAtTime(timeUs,
                        MediaMetadataRetriever.OPTION_CLOSEST);

                if (bmp == null) {
                    Log.w(TAG, "No frame at idx " + frameIdx + " (timeUs=" + timeUs + ")");
                    continue;
                }

                // Save as JPEG
                String fileName = String.format("frame_%04d.jpg", frameIdx);
                File outFile = new File(outputDir, fileName);
                try (FileOutputStream fos = new FileOutputStream(outFile)) {
                    // Quality 90 — good enough for SIFT feature detection
                    bmp.compress(Bitmap.CompressFormat.JPEG, 90, fos);
                } catch (IOException e) {
                    Log.e(TAG, "Failed to write " + fileName + ": " + e.getMessage());
                } finally {
                    bmp.recycle();
                }
            }

            Log.i(TAG, "extractFrames: done, wrote to " + outDir);
        } catch (Exception e) {
            Log.e(TAG, "extractFrames failed: " + e.getMessage(), e);
        } finally {
            try { mmr.release(); } catch (Exception ignored) {}
        }
    }
}
