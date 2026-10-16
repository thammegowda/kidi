-keep class ai.gowda.kidi.NativeRuntime { *; }
-keep interface ai.gowda.kidi.NativeRuntime$PartialListener { *; }
-keepclassmembers class * implements ai.gowda.kidi.NativeRuntime$PartialListener {
	public void onPartial(java.lang.String);
}
-keep interface ai.gowda.kidi.NativeRuntime$TtsProgressListener { *; }
-keepclassmembers class * implements ai.gowda.kidi.NativeRuntime$TtsProgressListener {
	public void onProgress(int,int);
}