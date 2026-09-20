# Coding Guidelines

Kidi uses C++23 and a compact, standard-library-like style. These rules apply
to kidi-owned C++ code. Code under `third_party/` follows its upstream project.

## Language and APIs

- Require C++23.
- Use trailing return types for explicit function returns: `auto function(...) -> Type`.
  Put qualifiers before the arrow, e.g. `auto size() const noexcept -> std::size_t`.
  Constructors, destructors, conversion operators, and deduced-return functions
  keep their natural syntax. Preserve C and embedded shader signatures.
- Prefer standard-library types and vocabulary: RAII, value semantics,
  `std::span`, ranges, smart pointers, and algorithms.
- In nested `kidi` namespaces, use the root error vocabulary unqualified:
  `Result<T>`, `Error`, and `ErrorCode`. External code uses `kidi::Result<T>`.
- Do not spell the underlying `std::expected<T, core::Error>` directly.
- Keep ownership explicit. Prefer values and references; use pointers only when
  nullability or indirection is part of the contract.
- Keep public APIs small and typed. Do not expose dependency-specific types
  unless the dependency is itself the API boundary.
- Keep hot-path arguments borrowed; do not copy shared owners just to dispatch
  an operation. Reuse scratch buffers and metadata capacity; allocate before
  token loops or during shape preparation/warmup. Arena reuse must respect live
  tensor aliases and unfinished device work. Do not remove lifetime ownership
  merely to avoid reference counting.
- Omit `[[nodiscard]]` in Kidi-owned code. Callers are responsible for checking
  return values and handling errors.
- Use `func(...)` for nonmutating tensor operations and `func_(Tensor&, ...)`
  for explicit in-place updates returning the same `Tensor&`. Preserve destination
  shape, dtype, and storage; aliases observe mutations. Dispatch scopes the
  thread-local `ops::is_inplace` flag with RAII and restores it on exceptions.
  Ordinary operations always select immutable execution, regardless of an outer
  flag value. Do not toggle the flag manually in model code.
- Neural layers and models derive from `Module`, implement typed `forward(...)`,
  and use `<Name>Impl` for the implementation with `KIDI_MODULE(Name)` declaring
  the `std::shared_ptr<NameImpl>` alias. Prefer shared ownership at composition
  boundaries; borrow `Impl&`/`const Impl&` or use `->forward` without copying the
  shared pointer inside hot paths.
- Register parameter members and child modules during construction. Use typed
  `ModuleList<Impl>` / `ModuleMap<Impl>` for registered containers. Module objects
  are nonmovable so registered tensor-member addresses remain valid. Use
  `load_state_dict`, not in-place weight mutation, to replace prepared parameters.

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
  `src/kidi/runtime/ynn/graph.h` declares names in `kidi::runtime::ynn`.
- Use the module ownership rules in [ARCHITECTURE.md](ARCHITECTURE.md): tensor
  storage, eager ops, shared layers, model topology, inference policy, and
  backend runtime are distinct concerns. Do not add a namespace for an import
  format or duplicate model/layer equations for each backend.
- Model/layer code operates on concrete tensors. Do not reintroduce a symbolic
  model graph, generic lowerer, lazy fallback, or recorded generation loop.
  Backend-private prepared operators are allowed and must remain invisible to
  model definitions. Use `ops::Failure` within eager computation; translate it
  to `Result<T>` at model/application boundaries. Synchronize before host reads.
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