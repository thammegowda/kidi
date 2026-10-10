idf_component_register(
    SRCS "kidi_controller_store.cpp"
    INCLUDE_DIRS "."
    REQUIRES kidi_protocol nvs_flash
)
target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Werror)
