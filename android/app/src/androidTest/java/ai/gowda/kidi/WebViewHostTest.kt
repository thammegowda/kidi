package ai.gowda.kidi

import android.Manifest
import android.content.pm.PackageManager
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

@RunWith(AndroidJUnit4::class)
class WebViewHostTest {
    @Test
    fun bundledSharedUiLoadsWithoutAccessoryPermissions() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            var snapshot = ""
            for (attempt in 0 until 30) {
                snapshot = evaluate(scenario,
                    "[location.origin,document.title," +
                        "document.getElementById('open-config')?.getAttribute('aria-label')," +
                        "document.getElementById('open-reader-view')?.getAttribute('aria-label')," +
                        "document.getElementById('model-onboarding-name')?.textContent," +
                        "!document.getElementById('host-legal-links')?.hidden," +
                        "document.getElementById('prompt')?.rows," +
                        "document.documentElement.classList.contains('android-host')].join('|')")
                if ("google/gemma-4-E2B-it-qat-mobile-transformers" in snapshot) break
                Thread.sleep(250)
            }
            assertTrue(snapshot, "https://appassets.androidplatform.net|Kidi|Settings|Reader" in snapshot)
            assertTrue(snapshot, "google/gemma-4-E2B-it-qat-mobile-transformers|true|2|true" in snapshot)
            scenario.onActivity { activity ->
                val webView = findWebView(activity.findViewById(android.R.id.content))
                val margins = webView.layoutParams as ViewGroup.MarginLayoutParams
                assertTrue("WebView must start below the Android status bar", margins.topMargin > 0)
            }
        }
        val permissions = instrumentation.targetContext.packageManager.getPackageInfo(
            instrumentation.targetContext.packageName,
            PackageManager.PackageInfoFlags.of(PackageManager.GET_PERMISSIONS.toLong()),
        ).requestedPermissions.orEmpty().toSet()
        assertFalse(Manifest.permission.BLUETOOTH_SCAN in permissions)
        assertFalse(Manifest.permission.NEARBY_WIFI_DEVICES in permissions)
    }

    private fun evaluate(scenario: ActivityScenario<MainActivity>, script: String): String {
        val result = AtomicReference("")
        val completed = CountDownLatch(1)
        scenario.onActivity { activity ->
            val content = activity.findViewById<ViewGroup>(android.R.id.content)
            val webView = findWebView(content)
            webView.evaluateJavascript(script) {
                result.set(it)
                completed.countDown()
            }
        }
        assertTrue("Timed out evaluating the packaged WebView", completed.await(5, TimeUnit.SECONDS))
        return result.get()
    }

    private fun findWebView(view: View): WebView {
        if (view is WebView) return view
        if (view is ViewGroup) {
            for (index in 0 until view.childCount) {
                runCatching { return findWebView(view.getChildAt(index)) }
            }
        }
        error("WebView not found")
    }
}
