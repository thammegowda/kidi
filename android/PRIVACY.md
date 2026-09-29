# Kidi Android Privacy Policy

Effective date: September 28, 2026

This policy applies to the Kidi Android application (`ai.gowda.kidi`), provided
by the Kidi project maintainers ("we", "us", or "our"). It does not cover
independent websites, app stores, or modified versions distributed by others.

## In Brief

We do not collect your personal data through the Kidi Android app. Your chats,
photos, and speech are processed on your device, not on our servers or a remote
AI service. We do not sell your data, use it for advertising, or use your
content to train models. No account is required.

On-device processing still involves storing some data locally. Model downloads
also require connections to third-party hosting services, as explained below.

## Data on Your Device

- **Chats:** Prompts, responses, saved transcripts, conversation titles, and
  associated metadata are stored in the app's private local storage so you can
  reopen conversations. Chat search also runs locally.
- **Photos:** Photos you choose or capture for a message are copied into private
  app storage and processed on-device. Kidi does not browse your entire photo
  library. Original photos held by another app remain under that app's control.
- **Microphone audio:** Optional dictation processes audio in memory on your
  device. Kidi does not upload audio or save an audio recording. Transcribed
  text becomes part of a saved conversation if you send it as a message.
- **Settings and models:** Preferences, downloaded model files, and locally
  generated model caches are stored on your device.

The app does not include advertising or tracking SDKs, usage analytics, or an
automatic crash-report upload service. Local performance measurements, such as
generation time and token counts, are not automatically sent to us.

## Internet Connections and Third Parties

When you request a model download, Kidi contacts Hugging Face and the hosting
or content-delivery services it redirects downloads to. These services receive
the information needed to handle the connection, including your IP address,
the requested model and files, and ordinary request metadata. They may retain
server logs under their own policies. Kidi does not include your chats, photos,
audio, or transcripts in these requests.

See the [Hugging Face Privacy Policy](https://huggingface.co/privacy).
Once the required models are installed, chat, photo analysis, and dictation can
run offline.

Google Play, your device manufacturer, your keyboard, your system camera or
photo picker, and any website or app you choose to open operate independently.
They may process information under their own policies and your settings. This
policy does not promise that those services collect no data.

## Permissions and Your Choices

- **Microphone:** Requested when you use dictation. You can deny or revoke it
  in Android settings and continue using text chat.
- **Photos and camera:** You choose individual photos through Android's photo
  picker or capture a photo through a system camera app. Kidi does not request
  broad photo-library access. Camera capture uses a temporary file-sharing
  permission for the camera app.
- **Internet:** Used to retrieve model information and files. Downloads require
  your confirmation and may incur charges from your network provider.

## Retention and Deletion

Saved chats, private photo copies, preferences, and models remain locally until
removed or the app's data is cleared. Starting a new chat does not erase earlier
conversations. Removing a downloaded model does not erase chat history.

To remove all Kidi app data, use Android's app settings to clear Kidi's storage
or uninstall Kidi. Copies you saved elsewhere, original photos, and information
you shared with another app or a support service are not removed by doing so.
We do not hold a server-side copy of your conversations and cannot recover them
for you. Local deletion is not a guarantee of forensic secure erasure.

Kidi requests that Android's app backup be disabled. Device-level transfer,
backup tools, and operating-system behavior may vary; we do not control them.

## Security

Kidi uses Android's app-private storage and HTTPS for its model-hosting
connections. No software or device can guarantee complete security. Protect
your device and avoid entering sensitive information on shared or compromised
devices. On-device processing is not a promise of anonymity or protection from
someone with access to your device.

## Contact and Information You Choose to Send

For privacy questions, contact the Kidi maintainers through the
[project issue tracker](https://github.com/thammegowda/kidi/issues).
**Issues are public. Do not post private conversations, photos, recordings,
passwords, or other sensitive personal information.**

If you voluntarily contact us, we receive the information you submit and use it
to respond to your request or address the reported issue. This is separate from
automatic app collection. Public issue records may remain available until
removed under GitHub's controls and retention policies. See the
[GitHub Privacy Statement](https://docs.github.com/en/site-policy/privacy-policies/github-general-privacy-statement).
Requests concerning records held by a third party are subject to that party's
procedures and applicable law.

## Changes

We will update this policy and its effective date when the app's privacy
practices change. Material changes will be communicated as required by
applicable law, and any required consent will be obtained before new processing
begins.

For AI limitations and conditions of use, see the
[Kidi Android Terms of Use](TERMS.md).