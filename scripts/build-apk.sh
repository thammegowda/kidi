#!/usr/bin/env bash
set -euo pipefail

fail() {
    printf 'build-apk: %s\n' "$*" >&2
    exit 1
}

release_signing=false
if [[ "${1:-}" == --release ]]; then
    release_signing=true
    shift
fi
[[ $# -eq 0 ]] || fail 'Usage: scripts/build-apk.sh [--release] (or make apk / make apk-release).'
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$root"

if [[ -z "${JAVA_HOME:-}" ]]; then
    if command -v brew >/dev/null 2>&1; then
        java_prefix=$(brew --prefix openjdk@17 2>/dev/null || true)
        if [[ -n "$java_prefix" ]]; then
            JAVA_HOME="$java_prefix/libexec/openjdk.jdk/Contents/Home"
        fi
    fi
    if [[ -z "${JAVA_HOME:-}" && -x /usr/libexec/java_home ]]; then
        JAVA_HOME=$(/usr/libexec/java_home -v 17 2>/dev/null || true)
    fi
fi
[[ -n "${JAVA_HOME:-}" && -x "$JAVA_HOME/bin/javac" ]] || fail 'Set JAVA_HOME to an installed JDK 17.'
export JAVA_HOME
export ANDROID_HOME="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-$HOME/Library/Android/sdk}}"
[[ -d "$ANDROID_HOME" ]] || fail 'Set ANDROID_HOME to your Android SDK directory.'

tools="$ANDROID_HOME/build-tools/${ANDROID_BUILD_TOOLS_VERSION:-36.0.0}"
for tool in zipalign apksigner; do
    [[ -x "$tools/$tool" ]] || fail "Missing $tools/$tool; install Android SDK Build Tools."
done

output_name=kidi-release.apk
if [[ "$release_signing" == true ]]; then
    keystore="${APK_KEYSTORE:-$HOME/.local/share/kidi/keys/upload.keystore}"
    key_alias="${APK_KEY_ALIAS:-kidi-upload}"
    output_name=kidi-release-signed.apk
    [[ -r "$keystore" ]] || fail 'Release keystore not found. Create your upload key or set APK_KEYSTORE.'
    [[ "$key_alias" != androiddebugkey && ! "$keystore" -ef "$HOME/.android/debug.keystore" ]] ||
        fail 'apk-release must not use the Android debug key.'
    printf 'Signing with the configured release/upload key; no debug-key fallback.\n'
else
    keystore="${APK_KEYSTORE:-$HOME/.android/debug.keystore}"
    key_alias="${APK_KEY_ALIAS:-}"
    [[ -r "$keystore" ]] || fail 'Signing keystore not found. Build assembleDebug once to create the default debug key, or set APK_KEYSTORE and APK_KEY_ALIAS.'
fi
signing=(--ks "$keystore")
if [[ "$release_signing" == true || -n "${APK_KEYSTORE:-}" ]]; then
    [[ -n "$key_alias" ]] || fail 'APK_KEY_ALIAS is required with APK_KEYSTORE.'
    signing+=(--ks-key-alias "$key_alias")
    if [[ -n "${APK_STORE_PASSWORD:-}" ]]; then
        signing+=(--ks-pass env:APK_STORE_PASSWORD)
    fi
    if [[ -n "${APK_KEY_PASSWORD:-}" ]]; then
        signing+=(--key-pass env:APK_KEY_PASSWORD)
    fi
else
    printf 'Signing with the existing Android debug key: local testing only.\n'
    signing+=(--ks-key-alias androiddebugkey --ks-pass pass:android --key-pass pass:android)
fi

./android/gradlew -p android :app:clean :app:assembleRelease :app:lintRelease --console=plain
unsigned="$root/android/app/build/outputs/apk/release/app-release-unsigned.apk"
[[ -f "$unsigned" ]] || fail "Release APK was not produced: $unsigned"

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/kidi-apk.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT
"$tools/zipalign" -f -P 16 4 "$unsigned" "$work_dir/aligned.apk"
"$tools/apksigner" sign "${signing[@]}" --out "$work_dir/signed.apk" "$work_dir/aligned.apk"
"$tools/apksigner" verify --verbose "$work_dir/signed.apk"
"$tools/zipalign" -c -P 16 4 "$work_dir/signed.apk"

mkdir -p "$root/dist"
mv "$work_dir/signed.apk" "$root/dist/$output_name"
printf '\nSigned release APK: %s/dist/%s\n' "$root" "$output_name"