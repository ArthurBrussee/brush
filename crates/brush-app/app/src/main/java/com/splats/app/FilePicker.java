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

    // ✅ Now passes fd + filename
    private static native void onFilePickerResult(int fd, String name);

    public static void Register(Activity activity) {
        _activity = activity;
    }

    public static void startFilePicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);

        Log.i("FilePicker", "Starting file picker");
        _activity.startActivityForResult(intent, REQUEST_CODE_PICK_FILE);
    }

    public static void onPicked(Uri uri, int fd) {
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
