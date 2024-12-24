This runs brush using on Android.

To run this sample:
```
# Make sure you have the Android SDK & NDK installed

# One time setup:

# Android Studio might set these for you.
export ANDROID_NDK_HOME="path/to/ndk"
export ANDROID_HOME="path/to/sdk"
# Install the rust android libraries.
rustup target add aarch64-linux-android
# Install cargo-ndk to manage building.
cargo install cargo-ndk

# Each time you change the rust code:
cd ./crates/brush-android # Make sure to run this in the android crate path.
cargo ndk -t arm64-v8a -o app/src/main/jniLibs/  build

# Nb, for best performance, build in release mode. This is separate
# from the Android Studio app build configuration.
cargo ndk -t arm64-v8a -o app/src/main/jniLibs/  build --release

# You can now either run the project from Android Studio (Android Studio does NOT build
# the rust code), or run it from the command line:
./gradlew build
./gradlew installDebug
adb shell am start -n com.splats.app/.MainActivity
```

You can also open this folder as a project in Android Studio and run things from there.
Nb: Running in Android Studio does _not_ rebuild the rust code.
