# Upstream provenance

This component is the `example_video_common` helper from Espressif's ESP-IDF
5.5 `simple_video_server` example, as resolved for the working OV5647
diagnostic build. The source retains its original `ESPRESSIF MIT` headers.
Trailing whitespace in the upstream Kconfig was normalized; behavior is
otherwise unchanged.

It is vendored because the helper is not published as a standalone registry
component. Kidi selects only the ESP32-P4 Function EV Board v1.5 pin profile
that was verified on the diagnosed Waveshare board. Product-specific capture,
ownership, bounds and protocol behavior live outside this upstream helper.
