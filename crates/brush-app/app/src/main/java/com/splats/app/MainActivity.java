package com.splats.app;

import androidx.core.view.WindowCompat;
import androidx.core.view.WindowInsetsCompat;
import androidx.core.view.WindowInsetsControllerCompat;

import com.google.androidgamesdk.GameActivity;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.ParcelFileDescriptor;
import android.view.View;
import android.view.WindowManager;
import android.util.Log;

import android.widget.Button;
import android.widget.FrameLayout;
import android.view.Gravity;
import android.view.ViewGroup.LayoutParams;

import java.io.File;
import java.io.FileOutputStream;

public class MainActivity extends GameActivity {

    static {
        System.loadLibrary("brush_app");
    }

    public static MainActivity instance;

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

        lp.gravity = Gravity.BOTTOM | Gravity.END;
        lp.setMargins(0, 0, 48, 48);

        Button extractButton = new Button(this);
        extractButton.setText("Extract frames");

        extractButton.setOnClickListener(v -> {
            Log.i("Brush", "Picking video");
            FilePicker.startFilePicker();
        });

        addContentView(extractButton, lp);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {

        if (requestCode == FilePicker.REQUEST_CODE_PICK_FILE) {

            try {

                if (resultCode == Activity.RESULT_OK && data != null) {

                    Uri uri = data.getData();

                    if (uri != null) {

                        extractFrames(uri);
                    }
                }

            } catch (Exception e) {

                Log.e("Brush", "Picker error", e);
            }
        }

        super.onActivityResult(requestCode, resultCode, data);
    }

    private void extractFrames(Uri videoUri) {

        new Thread(() -> {

            try {

                MediaMetadataRetriever retriever = new MediaMetadataRetriever();

                retriever.setDataSource(this, videoUri);

                String durationStr =
                        retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION);

                long durationMs = Long.parseLong(durationStr);

                long durationUs = durationMs * 1000;

                int frameCount = 100;

                long step = durationUs / frameCount;

                File outputDir =
                        getExternalFilesDir(Environment.DIRECTORY_PICTURES);

                if (outputDir != null && !outputDir.exists()) {
                    outputDir.mkdirs();
                }

                for (int i = 0; i < frameCount; i++) {

                    long timeUs = i * step;

                    Bitmap bitmap =
                            retriever.getFrameAtTime(
                                    timeUs,
                                    MediaMetadataRetriever.OPTION_CLOSEST
                            );

                    if (bitmap == null) continue;

                    File file =
                            new File(outputDir,
                                    String.format("frame_%03d.jpg", i));

                    FileOutputStream out = new FileOutputStream(file);

                    bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out);

                    out.flush();
                    out.close();

                    bitmap.recycle();
                }

                retriever.release();

                Log.i("Brush", "Frame extraction finished");

            } catch (Exception e) {

                Log.e("Brush", "Extraction failed", e);
            }

        }).start();
    }
}