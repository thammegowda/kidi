package ai.gowda.kidi

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.content.pm.ApplicationInfo
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.webkit.PermissionRequest
import android.webkit.RenderProcessGoneDetail
import android.webkit.ValueCallback
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import androidx.activity.ComponentActivity
import androidx.activity.enableEdgeToEdge
import androidx.activity.OnBackPressedCallback
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.webkit.WebMessageCompat
import androidx.webkit.WebMessagePortCompat
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewClientCompat
import androidx.webkit.WebViewCompat
import androidx.webkit.WebViewFeature
import org.json.JSONObject

class MainActivity : ComponentActivity() {
    private lateinit var webView: WebView
    private lateinit var assets: WebViewAssetLoader
    private var host: LiteAppServiceHost? = null
    private var port: WebMessagePortCompat? = null
    private var pendingPermission: PermissionRequest? = null
    private var pendingFiles: ValueCallback<Array<Uri>>? = null
    private var captureUri: Uri? = null
    private val imageStore by lazy { CameraCaptureStore(this) }

    private val microphonePermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        pendingPermission?.let { request ->
            if (granted) request.grant(arrayOf(PermissionRequest.RESOURCE_AUDIO_CAPTURE)) else request.deny()
        }
        pendingPermission = null
    }

    private val files = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        val values = mutableListOf<Uri>()
        result.data?.clipData?.let { clip ->
            repeat(clip.itemCount) { values += clip.getItemAt(it).uri }
        }
        result.data?.data?.let(values::add)
        if (result.resultCode == RESULT_OK && values.isEmpty()) captureUri?.let(values::add)
        captureUri?.takeIf { it !in values }?.let(imageStore::removeCapture)
        pendingFiles?.onReceiveValue(values.toTypedArray().takeIf { it.isNotEmpty() })
        pendingFiles = null
        captureUri = null
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        assets = WebViewAssetLoader.Builder()
            .addPathHandler("/assets/", WebViewAssetLoader.AssetsPathHandler(this))
            .build()
        webView = WebView(this)
        configureWebView()
        setContentView(webView)
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                if (webView.canGoBack()) webView.goBack() else finish()
            }
        })
        if (savedInstanceState == null) webView.loadUrl(APP_URL) else webView.restoreState(savedInstanceState)
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun configureWebView() {
        WebView.setWebContentsDebuggingEnabled(
            applicationInfo.flags and ApplicationInfo.FLAG_DEBUGGABLE != 0,
        )
        if (WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT)) {
            WebViewCompat.addDocumentStartJavaScript(
                webView,
                ANDROID_BOOTSTRAP,
                setOf(APP_ORIGIN),
            )
        }
        webView.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            allowFileAccess = false
            allowContentAccess = false
            mixedContentMode = WebSettings.MIXED_CONTENT_NEVER_ALLOW
            mediaPlaybackRequiresUserGesture = false
            setSupportMultipleWindows(false)
        }
        webView.webViewClient = object : WebViewClientCompat() {
            override fun shouldInterceptRequest(view: WebView, request: WebResourceRequest): WebResourceResponse? =
                assets.shouldInterceptRequest(request.url)?.apply {
                    responseHeaders = responseHeaders.orEmpty() + mapOf(
                        "Cache-Control" to "no-store",
                        "Content-Security-Policy" to CSP,
                    )
                }

            override fun shouldOverrideUrlLoading(view: WebView, request: WebResourceRequest): Boolean {
                if (isAppUri(request.url)) return false
                if (request.url.scheme in setOf("https", "http")) {
                    startActivity(Intent(Intent.ACTION_VIEW, request.url))
                }
                return true
            }

            override fun onPageFinished(view: WebView, url: String) {
                if (isAppUri(Uri.parse(url))) connectHost()
            }

            override fun onRenderProcessGone(view: WebView, detail: RenderProcessGoneDetail): Boolean {
                closeHost()
                view.destroy()
                recreate()
                return true
            }
        }
        webView.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                if (request.origin.toString().removeSuffix("/") != APP_ORIGIN ||
                    request.resources.any { it != PermissionRequest.RESOURCE_AUDIO_CAPTURE }) {
                    request.deny()
                    return
                }
                if (ContextCompat.checkSelfPermission(this@MainActivity, Manifest.permission.RECORD_AUDIO) ==
                    PackageManager.PERMISSION_GRANTED) {
                    request.grant(arrayOf(PermissionRequest.RESOURCE_AUDIO_CAPTURE))
                } else {
                    pendingPermission?.deny()
                    pendingPermission = request
                    microphonePermission.launch(Manifest.permission.RECORD_AUDIO)
                }
            }

            override fun onPermissionRequestCanceled(request: PermissionRequest) {
                if (pendingPermission == request) pendingPermission = null
            }

            override fun onShowFileChooser(
                webView: WebView,
                filePathCallback: ValueCallback<Array<Uri>>,
                fileChooserParams: FileChooserParams,
            ): Boolean {
                pendingFiles?.onReceiveValue(null)
                pendingFiles = filePathCallback
                val content = fileChooserParams.createIntent().apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    putExtra(Intent.EXTRA_ALLOW_MULTIPLE, fileChooserParams.mode == FileChooserParams.MODE_OPEN_MULTIPLE)
                }
                val camera = Intent(MediaStore.ACTION_IMAGE_CAPTURE).takeIf {
                    fileChooserParams.acceptTypes.any { type -> type.startsWith("image/") } &&
                        it.resolveActivity(packageManager) != null
                }?.apply {
                    captureUri = imageStore.captureUri()
                    putExtra(MediaStore.EXTRA_OUTPUT, captureUri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
                }
                val chooser = Intent.createChooser(content, fileChooserParams.title ?: "Choose a file")
                camera?.let { chooser.putExtra(Intent.EXTRA_INITIAL_INTENTS, arrayOf(it)) }
                files.launch(chooser)
                return true
            }
        }
    }

    private fun connectHost() {
        closeHost()
        if (!WebViewFeature.isFeatureSupported(WebViewFeature.CREATE_WEB_MESSAGE_CHANNEL) ||
            !WebViewFeature.isFeatureSupported(WebViewFeature.POST_WEB_MESSAGE) ||
            !WebViewFeature.isFeatureSupported(WebViewFeature.DOCUMENT_START_SCRIPT) ||
            !WebViewFeature.isFeatureSupported(WebViewFeature.WEB_MESSAGE_ARRAY_BUFFER)) {
            showFatal("Update Android System WebView to use Kidi.")
            return
        }
        val channel = WebViewCompat.createWebMessageChannel(webView)
        val nativePort = channel[0]
        val service = LiteAppServiceHost(
            context = this,
            sendText = { value -> webView.post { nativePort.postMessage(WebMessageCompat(value)) } },
            sendBinary = { value -> webView.post { nativePort.postMessage(WebMessageCompat(value)) } },
            fatal = { message -> webView.post { showFatal(message) } },
        )
        nativePort.setWebMessageCallback(object : WebMessagePortCompat.WebMessageCallbackCompat() {
            override fun onMessage(port: WebMessagePortCompat, message: WebMessageCompat?) {
                message ?: return
                when (message.type) {
                    WebMessageCompat.TYPE_STRING -> message.data?.let(service::receiveText)
                    WebMessageCompat.TYPE_ARRAY_BUFFER -> service.receiveBinary(message.arrayBuffer)
                    else -> showFatal("Unsupported Android WebMessage payload")
                }
            }
        })
        host = service
        port = nativePort
        WebViewCompat.postWebMessage(
            webView,
            WebMessageCompat(CONNECT_MESSAGE, arrayOf(channel[1])),
            Uri.parse(APP_ORIGIN),
        )
    }

    private fun showFatal(message: String) {
        closeHost()
        webView.evaluateJavascript(
            "document.body.replaceChildren(Object.assign(document.createElement('p')," +
                "{textContent:${JSONObject.quote(message)},style:'padding:24px;font:16px sans-serif'}));",
            null,
        )
    }

    private fun closeHost() {
        host?.close()
        host = null
        port?.close()
        port = null
    }

    override fun onSaveInstanceState(outState: Bundle) {
        webView.saveState(outState)
        super.onSaveInstanceState(outState)
    }

    override fun onDestroy() {
        pendingPermission?.deny()
        pendingPermission = null
        pendingFiles?.onReceiveValue(null)
        pendingFiles = null
        closeHost()
        webView.stopLoading()
        webView.destroy()
        super.onDestroy()
    }

    companion object {
        private const val APP_ORIGIN = "https://appassets.androidplatform.net"
        private const val APP_URL = "$APP_ORIGIN/assets/index.html"
        private const val CONNECT_MESSAGE = "liteapp:android:connect"
        private const val ANDROID_BOOTSTRAP =
            "globalThis.__liteappAndroidPort=new Promise(resolve=>{" +
                "addEventListener('message',event=>{" +
                "if(event.data==='liteapp:android:connect'&&event.ports.length)resolve(event.ports[0]);" +
                "});});"
        private const val CSP =
            "default-src 'self'; base-uri 'none'; connect-src 'self'; form-action 'none'; " +
                "frame-ancestors 'none'; img-src 'self' data: blob:; media-src blob:; " +
                "script-src 'self'; style-src 'self'; worker-src 'self' blob:"

        private fun isAppUri(uri: Uri) =
            uri.scheme == "https" && uri.host == "appassets.androidplatform.net"
    }
}
