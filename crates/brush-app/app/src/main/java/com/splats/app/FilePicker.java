package com.splats.app;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Intent;
import android.util.Log;

public class FilePicker {
    @SuppressLint("StaticFieldLeak")
    private static Activity _activity;
    public static final int REQUEST_CODE_PICK_FILE = 1;
    private static native void onFilePickerResult(int fd);

    public static void Register(Activity activity) {
        _activity = activity;
    }

    public static void startFilePicker() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        Log.i("FilePicker", "GHello from Java!");
        _activity.startActivityForResult(intent, REQUEST_CODE_PICK_FILE);
    }

    public static void onPicked(int resultCode, int fd) {
        onFilePickerResult(fd);
    }
}
