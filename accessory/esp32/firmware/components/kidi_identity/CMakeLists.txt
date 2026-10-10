idf_component_register(
    SRCS "kidi_identity.cpp"
    INCLUDE_DIRS "."
    REQUIRES mbedtls nvs_flash
)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Werror)
