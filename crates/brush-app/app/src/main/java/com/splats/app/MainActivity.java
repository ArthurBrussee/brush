package com.splats.app;

import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;

import com.google.androidgamesdk.GameActivity;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.util.Log;
import android.util.TypedValue;
import android.provider.OpenableColumns;
import android.widget.Button;
import android.widget.FrameLayout;
import android.view.Gravity;
import android.view.ViewGroup.LayoutParams;
import android.view.View;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Locale;

import kotlinx.coroutines.GlobalScope;

import com.splats.app.telemetry.TelemetryPreprocessor;
import com.splats.app.telemetry.TelemetryPreprocessorCallback;
import com.splats.app.telemetry.ProcessingStage;
import com.splats.app.telemetry.PoseStampSequence;
import com.splats.app.telemetry.TelemetryProcessingReport;
import com.splats.app.telemetry.KeyframeSelectionConfig;

public class MainActivity extends GameActivity {

    static {
        System.loadLibrary("brush_app");
    }

    @SuppressLint("StaticFieldLeak")
    public static MainActivity instance;

    private static final int REQUEST_CODE_EXTRACT_FRAMES = 1001;
    private static final int REQUEST_CODE_PICK_CSV = 1002;
    private static final int FRAME_COUNT = 100;
    private static final int MAX_DIMENSION = 4096;
    private static final String TAG = "MainActivity";

    private File selectedCsvFile = null;
    private File selectedVideoFile = null;
    private boolean telemetryRunning = false;

    private void hideSystemUI() {
        getWindow().getAttributes().layoutInDisplayCutoutMode =
                WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS;

        View decorView = getWindow().getDecorView();

        WindowInsetsControllerCompat controller =
                new WindowInsetsControllerCompat(getWindow(), decorView);

        controller.hide(WindowInsetsCompat.Type.systemBars());
        controller.hide(WindowInsetsCompat.Type.displayCutout());

        controller.setSystemBarsBehavior(
                WindowInsetsControllerCompat.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE
        );
    }


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        instance = this;

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        WindowCompat.setDecorFitsSystemWindows(getWindow(), false);

        hideSystemUI();

        FilePicker.Register(this);

        FrameLayout.LayoutParams lp = new FrameLayout.LayoutParams(
                LayoutParams.WRAP_CONTENT,
                LayoutParams.WRAP_CONTENT
        );

        // ✅ FIXED POSITION (centered, slightly below .ply panel)
        lp.gravity = Gravity.CENTER_HORIZONTAL | Gravity.CENTER_VERTICAL;
        lp.setMargins(0, dp(200), 0, 0);

        LinearLayout buttonColumn = new LinearLayout(this);
        buttonColumn.setOrientation(LinearLayout.VERTICAL);
        buttonColumn.setGravity(Gravity.CENTER_HORIZONTAL);

        LinearLayout.LayoutParams btnParams = new LinearLayout.LayoutParams(
                LayoutParams.WRAP_CONTENT,
                LayoutParams.WRAP_CONTENT
        );
        btnParams.bottomMargin = dp(6);

        // ===== Extract Frames =====
        Button extractButton = new Button(this, null, android.R.attr.borderlessButtonStyle);
        extractButton.setText("Extract Frames");
        extractButton.setAllCaps(false);
        extractButton.setTextSize(13);
        extractButton.setPadding(dp(12), dp(6), dp(12), dp(6));
        extractButton.setBackgroundColor(Color.parseColor("#4A7BA7"));
        extractButton.setTextColor(Color.WHITE);
        extractButton.setLayoutParams(btnParams);

        extractButton.setOnClickListener(v -> {
            Log.i(TAG, "Extract frames button clicked");
            FilePicker.startFilePicker(REQUEST_CODE_EXTRACT_FRAMES);
        });

        // ===== CSV Button =====
        Button csvButton = new Button(this, null, android.R.attr.borderlessButtonStyle);
        csvButton.setText("Select telemetry CSV");
        csvButton.setAllCaps(false);
        csvButton.setTextSize(13);
        csvButton.setPadding(dp(12), dp(6), dp(12), dp(6));
        csvButton.setBackgroundColor(Color.parseColor("#4A7BA7"));
        csvButton.setTextColor(Color.WHITE);
        csvButton.setLayoutParams(btnParams);

        csvButton.setOnClickListener(v -> {
            Log.i(TAG, "Select telemetry CSV clicked");
            FilePicker.startCsvPicker(REQUEST_CODE_PICK_CSV);
        });

        buttonColumn.addView(extractButton);
        buttonColumn.addView(csvButton);

        addContentView(buttonColumn, lp);
    }

    private int dp(int v) {
        return Math.round(TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP,
                v,
                getResources().getDisplayMetrics()
        ));
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        if (requestCode == REQUEST_CODE_PICK_CSV) {
            try {
                if (resultCode == Activity.RESULT_OK && data != null) {
                    Uri uri = data.getData();
                    if (uri != null) {
                        try {
                            int takeFlags = Intent.FLAG_GRANT_READ_URI_PERMISSION;
                            if ((data.getFlags() & Intent.FLAG_GRANT_WRITE_URI_PERMISSION) != 0) {
                                takeFlags |= Intent.FLAG_GRANT_WRITE_URI_PERMISSION;
                            }
                            getContentResolver().takePersistableUriPermission(uri, takeFlags);
                        } catch (Exception e) {
                            Log.i(TAG, "Could not take persistable permission: " + e);
                        }

                        String displayName = queryDisplayName(uri);
                        if (!isCsvName(displayName)) {
                            Log.w(TAG, "Rejected non-CSV telemetry file: " + displayName);
                            Toast.makeText(this,
                                    "Only CSV telemetry logs are supported",
                                    Toast.LENGTH_SHORT).show();
                            return;
                        }

                        selectedCsvFile = ensureLocalFileForUri(uri, "telemetry_csv_", ".csv");

                        if (selectedCsvFile != null) {
                            Toast.makeText(this,
                                    "Telemetry CSV selected: " + selectedCsvFile.getName(),
                                    Toast.LENGTH_SHORT).show();
                        } else {
                            Toast.makeText(this,
                                    "Failed to read telemetry CSV",
                                    Toast.LENGTH_SHORT).show();
                        }

                        startTelemetryPreprocessIfReady();
                    }
                }
            } catch (Exception e) {
                Log.e(TAG, "CSV pick error", e);
            }

            super.onActivityResult(requestCode, resultCode, data);
            return;
        }

        // Keep original handling for the app's FilePicker-based flows
        if (requestCode == FilePicker.REQUEST_CODE_PICK_FILE || requestCode == REQUEST_CODE_EXTRACT_FRAMES) {

            int fd = -1;

            try {

                if (resultCode == Activity.RESULT_OK && data != null) {

                    Uri uri = data.getData();

                    if (uri != null) {
                        // Persist read permission so MediaMetadataRetriever can read later
                        try {
                            int takeFlags = Intent.FLAG_GRANT_READ_URI_PERMISSION;
                            if ((data.getFlags() & Intent.FLAG_GRANT_WRITE_URI_PERMISSION) != 0) {
                                takeFlags |= Intent.FLAG_GRANT_WRITE_URI_PERMISSION;
                            }
                            getContentResolver().takePersistableUriPermission(uri, takeFlags);
                        } catch (Exception e) {
                            // Not all URIs allow persistable permission; ignore safely
                            Log.i("MainActivity", "Could not take persistable permission: " + e);
                        }

                        // Try to open FD for the native/Rust code as before.
                        ParcelFileDescriptor parcelFileDescriptor =
                                getContentResolver().openFileDescriptor(uri, "r");

                        if (parcelFileDescriptor != null) {
                            fd = parcelFileDescriptor.detachFd();

                            // IMPORTANT: first kick off Java extraction (only for our extract request),
                            // then call FilePicker.onPicked so existing native logic still runs.
                            if (requestCode == REQUEST_CODE_EXTRACT_FRAMES) {
                                extractFrames(uri);
                            }

                            // keep original behavior for native side
                            FilePicker.onPicked(uri, fd);

                            selectedVideoFile = ensureLocalFileForUri(uri, "telemetry_video_", ".mp4");
                            startTelemetryPreprocessIfReady();

                            // return after we've done both actions
                            return;
                        } else {
                            // If no parcelFileDescriptor, still allow extraction if this was the extract request
                            if (requestCode == REQUEST_CODE_EXTRACT_FRAMES) {
                                extractFrames(uri);
                                // call native with invalid fd to preserve previous behavior
                                FilePicker.onPicked(uri, -1);
                                selectedVideoFile = ensureLocalFileForUri(uri, "telemetry_video_", ".mp4");
                                startTelemetryPreprocessIfReady();
                                return;
                            }
                        }
                    }
                }

            } catch (Exception e) {
                Log.e("MainActivity", "onActivityResult error", e);
            }

            // Ensure the native picker is notified in all failure paths, to match previous behavior
            FilePicker.onPicked(null, -1);
        }

        super.onActivityResult(requestCode, resultCode, data);
    }

    private void startTelemetryPreprocessIfReady() {
        if (telemetryRunning) return;
        if (selectedCsvFile == null || selectedVideoFile == null) return;
        if (!selectedCsvFile.exists() || !selectedVideoFile.exists()) return;

        telemetryRunning = true;
        Toast.makeText(this, "Starting telemetry preprocess...", Toast.LENGTH_SHORT).show();

        TelemetryPreprocessorCallback callback = new TelemetryPreprocessorCallback() {
            @Override
            public void onProgress(ProcessingStage stage, float fraction) {
                Log.i(TAG, "Telemetry progress: " + stage + " " + fraction);
            }

            @Override
            public void onComplete(PoseStampSequence sequence,
                                   Throwable error,
                                   TelemetryProcessingReport report) {
                telemetryRunning = false;
                if (error == null) {
                    Toast.makeText(MainActivity.this, "Telemetry preprocess complete", Toast.LENGTH_SHORT).show();
                    if (sequence != null) {
                        Toast.makeText(
                                MainActivity.this,
                                "Output: " + sequence.getLogPath().getParent(),
                                Toast.LENGTH_LONG
                        ).show();
                    }
                } else {
                    Log.e(TAG, "Telemetry preprocess failed", error);
                    String msg = error.getMessage() != null ? error.getMessage() : "Telemetry preprocess failed";
                    Toast.makeText(MainActivity.this, "Telemetry preprocess failed", Toast.LENGTH_LONG).show();
                    Toast.makeText(MainActivity.this, msg, Toast.LENGTH_LONG).show();
                }
            }
        };

        File outDir = getExternalFilesDir("telemetry");
        if (outDir == null) {
            outDir = getCacheDir();
        }
        if (!outDir.exists()) {
            outDir.mkdirs();
        }
        String sessionId = "session_" + System.currentTimeMillis();

        TelemetryPreprocessor preprocessor =
                new TelemetryPreprocessor(
                        selectedCsvFile,
                        selectedVideoFile,
                        new long[0],
                        callback,
                        new KeyframeSelectionConfig(),
                        outDir,
                        sessionId
                );
        preprocessor.start(GlobalScope.INSTANCE);
    }

    private File ensureLocalFileForUri(Uri uri, String prefix, String fallbackExt) {
        try {
            if (uri == null) return null;

            String name = queryDisplayName(uri);
            String ext = fallbackExt;
            if (name != null && name.contains(".")) {
                ext = name.substring(name.lastIndexOf('.'));
            }
            File out = new File(getCacheDir(), prefix + System.currentTimeMillis() + ext);

            try (InputStream in = getContentResolver().openInputStream(uri);
                 OutputStream outStream = new FileOutputStream(out)) {
                if (in == null) return null;
                byte[] buffer = new byte[8192];
                int read;
                while ((read = in.read(buffer)) != -1) {
                    outStream.write(buffer, 0, read);
                }
            }

            return out;
        } catch (Exception e) {
            Log.e(TAG, "Failed to cache uri " + uri, e);
            return null;
        }
    }

    private boolean isCsvName(String name) {
        if (name == null) return false;
        return name.toLowerCase(Locale.US).endsWith(".csv");
    }

    private String queryDisplayName(Uri uri) {
        if (uri == null) return null;
        Cursor cursor = null;
        try {
            cursor = getContentResolver().query(uri, null, null, null, null);
            if (cursor != null) {
                int nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (nameIndex >= 0 && cursor.moveToFirst()) {
                    return cursor.getString(nameIndex);
                }
            }
        } catch (Exception e) {
            Log.w(TAG, "Failed to get display name", e);
        } finally {
            if (cursor != null) cursor.close();
        }
        return null;
    }

    private void extractFrames(Uri videoUri) {

        // Show a simple modal progress dialog with a horizontal progress bar
        final AlertDialog[] dialogHolder = new AlertDialog[1];
        final ProgressBar[] progressBarHolder = new ProgressBar[1];
        final TextView[] statusTextHolder = new TextView[1];

        runOnUiThread(() -> {
            try {
                AlertDialog.Builder builder = new AlertDialog.Builder(MainActivity.this);
                builder.setTitle("Extracting frames");

                LinearLayout layout = new LinearLayout(MainActivity.this);
                layout.setOrientation(LinearLayout.VERTICAL);
                layout.setPadding(32, 24, 32, 8);

                ProgressBar progressBar = new ProgressBar(MainActivity.this, null, android.R.attr.progressBarStyleHorizontal);
                progressBar.setIndeterminate(false);
                progressBar.setMax(FRAME_COUNT);
                progressBar.setProgress(0);
                layout.addView(progressBar, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));

                TextView statusText = new TextView(MainActivity.this);
                statusText.setText("0 / " + FRAME_COUNT);
                statusText.setPadding(0, 12, 0, 0);
                layout.addView(statusText, new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));

                builder.setView(layout);
                builder.setCancelable(false);

                AlertDialog dialog = builder.create();
                dialog.show();

                dialogHolder[0] = dialog;
                progressBarHolder[0] = progressBar;
                statusTextHolder[0] = statusText;
            } catch (Exception e) {
                Log.w("MainActivity", "Could not show progress dialog", e);
            }
        });

        new Thread(() -> {
            try {
                MediaMetadataRetriever retriever = new MediaMetadataRetriever();
                retriever.setDataSource(this, videoUri);

                String durationStr = retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);
                long durationMs = 0;
                try {
                    durationMs = Long.parseLong(durationStr);
                } catch (Exception e) {
                    Log.w("MainActivity", "Invalid duration metadata, defaulting to 0", e);
                }

                if (durationMs <= 0) {
                    // fallback: single frame at 0
                    Bitmap single = retriever.getFrameAtTime(0);
                    if (single != null) {
                        File outDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
                        if (outDir != null && !outDir.exists()) outDir.mkdirs();
                        File f = new File(outDir, "frame_000.jpg");
                        try (FileOutputStream fos = new FileOutputStream(f)) {
                            single.compress(Bitmap.CompressFormat.JPEG, 95, fos);
                        }
                        single.recycle();
                    }
                    retriever.release();
                    final AlertDialog d = dialogHolder[0] != null ? dialogHolder[0] : null;
                    runOnUiThread(() -> {
                        if (d != null) d.dismiss();
                        Toast.makeText(MainActivity.this, "Extraction complete (1 frame)", Toast.LENGTH_SHORT).show();
                    });
                    return;
                }

                long durationUs = durationMs * 1000L;
                long stepUs = durationUs / FRAME_COUNT;

                File outputDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
                if (outputDir != null && !outputDir.exists()) outputDir.mkdirs();

                for (int i = 0; i < FRAME_COUNT; i++) {
                    long timeUs = i * stepUs;

                    Bitmap bitmap = null;
                    try {
                        if (Build.VERSION.SDK_INT >= 27) {
                            // API 27+ supports scaled retrieval. Request large target dims (no upscaling).
                            int targetW = MAX_DIMENSION;
                            int targetH = MAX_DIMENSION;
                            try {
                                bitmap = retriever.getScaledFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST, targetW, targetH);
                            } catch (Throwable t) {
                                // fallback to unscaled if scaled retrieval fails on some devices
                                bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST);
                            }
                        } else {
                            bitmap = retriever.getFrameAtTime(timeUs, MediaMetadataRetriever.OPTION_CLOSEST);
                        }

                        if (bitmap == null) {
                            Log.w("MainActivity", "Null bitmap at frame " + i);
                            // update progress UI even if null
                            final int progress = i + 1;
                            runOnUiThread(() -> {
                                if (progressBarHolder[0] != null) progressBarHolder[0].setProgress(progress);
                                if (statusTextHolder[0] != null) statusTextHolder[0].setText(progress + " / " + FRAME_COUNT);
                            });
                            continue;
                        }

                        // If bitmap is giant, downscale to avoid OOM. This keeps a lot of resolution but avoids crashes.
                        int w = bitmap.getWidth();
                        int h = bitmap.getHeight();
                        if (Math.max(w, h) > MAX_DIMENSION) {
                            float scale = (float) MAX_DIMENSION / (float) Math.max(w, h);
                            int newW = Math.max(1, Math.round(w * scale));
                            int newH = Math.max(1, Math.round(h * scale));
                            Bitmap scaled = Bitmap.createScaledBitmap(bitmap, newW, newH, true);
                            bitmap.recycle();
                            bitmap = scaled;
                        }

                        File outFile = new File(outputDir, String.format("frame_%03d.jpg", i));
                        try (FileOutputStream fos = new FileOutputStream(outFile)) {
                            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, fos);
                            fos.flush();
                        } catch (Exception e) {
                            Log.e("MainActivity", "Failed to write frame file " + outFile, e);
                        } finally {
                            bitmap.recycle();
                        }
                    } catch (OutOfMemoryError oom) {
                        Log.e("MainActivity", "OOM extracting frame " + i, oom);
                        // try to continue after an OOM; still update UI
                    } catch (Throwable t) {
                        Log.e("MainActivity", "Error extracting frame " + i, t);
                    }

                    final int progress = i + 1;
                    runOnUiThread(() -> {
                        if (progressBarHolder[0] != null) progressBarHolder[0].setProgress(progress);
                        if (statusTextHolder[0] != null) statusTextHolder[0].setText(progress + " / " + FRAME_COUNT);
                    });
                }

                retriever.release();

                runOnUiThread(() -> {
                    AlertDialog d = dialogHolder[0] != null ? dialogHolder[0] : null;
                    if (d != null) d.dismiss();
                    Toast.makeText(MainActivity.this, "Extraction finished", Toast.LENGTH_SHORT).show();
                });

            } catch (Exception e) {
                Log.e("MainActivity", "Extraction failed", e);
                runOnUiThread(() -> {
                    AlertDialog d = dialogHolder[0] != null ? dialogHolder[0] : null;
                    if (d != null) d.dismiss();
                    Toast.makeText(MainActivity.this, "Extraction failed: " + e.getMessage(), Toast.LENGTH_LONG).show();
                });
            }
        }).start();
    }
}
