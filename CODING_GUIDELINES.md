# Coding Guidelines

Kidi uses C++23 and a compact, standard-library-like style. These rules apply
to kidi-owned C++ code. Code under `third_party/` follows its upstream project.

## Language and APIs

- Require C++23.
- Prefer standard-library types and vocabulary: RAII, value semantics,
  `std::expected`, `std::span`, ranges, smart pointers, and algorithms.
- Keep ownership explicit. Prefer values and references; use pointers only when
  nullability or indirection is part of the contract.
- Keep public APIs small and typed. Do not expose dependency-specific types
  unless the dependency is itself the API boundary.
- Use `[[nodiscard]]` for results whose errors or values must be observed.

## Naming

- Classes, structs, enums, and other user-defined types use `TitleCase`.
- Functions and methods, except constructors and destructors, use
  `lowercase_snake_case`.
- Variables, parameters, namespaces, directories, and filenames use
  `lowercase_snake_case`.
- Private data members use `lowercase_snake_case_` with a trailing underscore.
- Enum values and constants use `SCREAMING_UPPER`, including `constexpr` and
  namespace-scope immutable values.
- Third-party API names are used as declared; do not wrap them solely to rename
  symbols.

## Structure

- Co-locate each `.h` and `.cpp` pair at the same directory level.
- Directory structure mirrors namespaces. For example,
  `src/kidi/runtime/ynn.h` declares names in `kidi::runtime`.
- Keep implementation details in unnamed namespaces in `.cpp` files.
- Prefer a focused free function over a new class when no state or lifetime is
  being modeled.

## Formatting

- `.clang-format` is authoritative for C++ formatting.
- Use four-space indentation, attached braces, left-aligned pointers and
  references, and a 120-column limit.
- Preserve intentional include groups; do not reformat unrelated code.
- Format changed C++ files before completion.

## Tests

- Test observable contracts, numerical parity, and failure boundaries.
- Avoid timing assertions, cosmetic assertions, and duplicate coverage of
  third-party libraries.
- Test code follows the same naming and formatting rules as production code.