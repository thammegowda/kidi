idf_component_register(
    SRCS "kidi_c6_main.cpp" "kidi_control_server.cpp"
    INCLUDE_DIRS "."
    REQUIRES bt esp_event esp_hosted esp_https_server esp_netif esp_wifi json kidi_controller_store kidi_identity kidi_invitation kidi_pairing kidi_protocol kidi_ticket kidi_tls_identity mbedtls nvs_flash
)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Werror)
