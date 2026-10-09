idf_component_register(
    SRCS "kidi_p4_main.cpp" "kidi_camera.cpp" "kidi_audio.cpp" "kidi_media_server.cpp"
    INCLUDE_DIRS "."
    REQUIRES console esp32_p4_platform esp_codec_dev esp_event esp_hosted esp_https_server esp_netif esp_video espressif_example_video_common json kidi_protocol kidi_ticket kidi_tls_identity mbedtls nvs_flash
)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Werror)
