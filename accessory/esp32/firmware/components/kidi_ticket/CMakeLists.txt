idf_component_register(
    SRCS "kidi_ticket.cpp"
    INCLUDE_DIRS "."
    REQUIRES kidi_protocol mbedtls
)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Werror)
