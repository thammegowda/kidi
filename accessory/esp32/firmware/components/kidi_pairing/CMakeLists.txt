idf_component_register(
    SRCS "kidi_pairing.cpp"
    INCLUDE_DIRS "."
    REQUIRES json kidi_controller_store kidi_identity kidi_protocol kidi_tls_identity mbedtls
)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Werror)
