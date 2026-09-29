plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "ai.gowda.kidi"
    compileSdk = 36
    buildToolsVersion = "36.0.0"

    defaultConfig {
        applicationId = "ai.gowda.kidi"
        minSdk = 29
        targetSdk = 36
        versionCode = 3
        versionName = "0.1.3"
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                targets += "kidi_android"
                arguments += listOf(
                    "-DKIDI_BUILD_TESTS=OFF",
                    "-DKIDI_BUILD_INTEGRATION_TESTS=OFF",
                    "-DKIDI_BUILD_BENCHMARKS=OFF",
                    "-DKIDI_BUILD_PYTHON=OFF",
                    "-DBUILD_TESTING=OFF",
                )
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }
    packaging.jniLibs.useLegacyPackaging = true
    sourceSets.getByName("main").res.srcDir(layout.buildDirectory.dir("generated/visionNotices"))
    sourceSets.getByName("main").assets.srcDir(layout.buildDirectory.dir("generated/legalDocuments"))
    ndkVersion = "28.0.13004108"
    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.31.6"
        }
    }
}

dependencies {
    implementation(platform("androidx.compose:compose-bom:2025.04.01"))
    implementation("androidx.activity:activity-compose:1.10.1")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.9.0")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.9.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation("io.noties.markwon:core:4.6.2")
    implementation("io.noties.markwon:ext-latex:4.6.2")
    testImplementation("junit:junit:4.13.2")
    testImplementation("com.squareup.okhttp3:mockwebserver:4.12.0")
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation(platform("androidx.compose:compose-bom:2025.04.01"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.6.1")
    androidTestImplementation("androidx.test.uiautomator:uiautomator:2.3.0")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}

val visionNotices by tasks.registering(Sync::class) {
    into(layout.buildDirectory.dir("generated/visionNotices/raw"))
    val notices = mapOf(
        "LICENSE" to "vision_tahoma_license.txt",
        "NOTICES.md" to "vision_notices.txt",
        "libs/pigzpp/LICENSE" to "vision_pigzpp_license.txt",
        "libs/pigzpp/third_party/zlib-ng/LICENSE.md" to "vision_zlib_ng_license.txt",
        "libs/pigzpp/third_party/zopfli/COPYING" to "vision_zopfli_license.txt",
        "libs/libjpeg-turbo/LICENSE.md" to "vision_jpeg_license.txt",
        "libs/libjpeg-turbo/README.ijg" to "vision_ijg_license.txt",
    )
    notices.forEach { (source, target) ->
        from(rootProject.file("../third_party/tahoma-vision/$source")) { rename { target } }
    }
    from(rootProject.file("../src/kidi/checkpoint/ggml/LICENSE")) { rename { "ggml_license.txt" } }
}
val legalDocuments by tasks.registering(Sync::class) {
    from(rootProject.file("PRIVACY.md"), rootProject.file("TERMS.md"))
    into(layout.buildDirectory.dir("generated/legalDocuments/legal"))
}
tasks.named("preBuild") { dependsOn(visionNotices, legalDocuments) }