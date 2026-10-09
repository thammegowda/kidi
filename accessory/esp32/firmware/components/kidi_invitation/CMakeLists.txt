idf_component_register(
    SRCS "kidi_invitation.cpp"
    INCLUDE_DIRS "."
    REQUIRES json kidi_identity kidi_pairing kidi_protocol mbedtls
)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Werror)
