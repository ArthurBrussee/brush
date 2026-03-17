package com.splats.app;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import android.util.Log;

public class FilePicker {

    @SuppressLint("StaticFieldLeak")
    private static Activity _activity;

    public static final int REQUEST_CODE_PICK_FILE = 1;
    public static final int REQUEST_CODE_PICK_CSV = 2;

    // native callback kept as before
    private static native void onFilePickerResult(int fd, String name);

    public static void Register(Activity activity) {
        _activity = activity;
    }

    // original convenience method (keeps compatibility)
    public static void startFilePicker() {
        startFilePicker(REQUEST_CODE_PICK_FILE);
    }

    // new overload to supply distinct request codes (so callers can differentiate)
    public static void startFilePicker(int requestCode) {
        if (_activity == null) {
            Log.e("FilePicker", "No activity registered");
            return;
        }

        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);

        // restrict to videos so users pick MP4/other video types
        intent.setType("video/*");

        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        intent.addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);

        Log.i("FilePicker", "Starting file picker (requestCode=" + requestCode + ")");
        _activity.startActivityForResult(intent, requestCode);
    }

    // CSV picker for telemetry ingest
    public static void startCsvPicker(int requestCode) {
        if (_activity == null) {
            Log.e("FilePicker", "No activity registered");
            return;
        }

        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.putExtra(Intent.EXTRA_MIME_TYPES, new String[] {
                "text/csv",
                "text/comma-separated-values",
                "text/plain"
        });

        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
        intent.addFlags(Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);

        Log.i("FilePicker", "Starting CSV picker (requestCode=" + requestCode + ")");
        _activity.startActivityForResult(intent, requestCode);
    }

    // Called by MainActivity; ensures null-check and preserves old native-callback behavior
    public static void onPicked(Uri uri, int fd) {
        if (uri == null) {
            Log.e("FilePicker", "onPicked: null uri");
            onFilePickerResult(-1, "invalid");
            return;
        }

        String name = "file";

        try {
            Cursor cursor = _activity.getContentResolver()
                    .query(uri, null, null, null, null);

            if (cursor != null) {
                int nameIndex = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (nameIndex >= 0 && cursor.moveToFirst()) {
                    name = cursor.getString(nameIndex);
                }
                cursor.close();
            }
        } catch (Exception e) {
            Log.e("FilePicker", "Failed to get filename", e);
        }

        Log.i("FilePicker", "Detached FD: " + fd + " name: " + name);

        onFilePickerResult(fd, name);
    }
}
