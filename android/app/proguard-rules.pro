-keep class ai.gowda.kidi.NativeRuntime { *; }
-keep interface ai.gowda.kidi.NativeRuntime$PartialListener { *; }
-keepclassmembers class * implements ai.gowda.kidi.NativeRuntime$PartialListener {
	public void onPartial(java.lang.String);
}