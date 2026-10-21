# TensorFlow Lite keeps native entry points that must not be stripped.
-keep class org.tensorflow.lite.** { *; }
-dontwarn org.tensorflow.lite.**
