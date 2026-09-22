# Vendored Web Dependencies

These are unmodified files from the pinned upstream release archives below.
No package-manager installation is required to build Kidi's browser app.

| Directory | Version | Source Archive | Included Files | License |
|---|---|---|---|---|
| `acorn/` | 8.15.0 | [acorn-8.15.0.tgz](https://registry.npmjs.org/acorn/-/acorn-8.15.0.tgz) | `dist/acorn.mjs` copied as `acorn.mjs` | MIT |
| `acorn-walk/` | 8.3.4 | [acorn-walk-8.3.4.tgz](https://registry.npmjs.org/acorn-walk/-/acorn-walk-8.3.4.tgz) | `dist/walk.mjs` copied as `walk.mjs` | MIT |
| `coi-serviceworker/` | 0.1.7 | [coi-serviceworker-0.1.7.tgz](https://registry.npmjs.org/coi-serviceworker/-/coi-serviceworker-0.1.7.tgz) | `coi-serviceworker.js` | MIT |
| `lucide-static/` | 0.468.0 | [lucide-static-0.468.0.tgz](https://registry.npmjs.org/lucide-static/-/lucide-static-0.468.0.tgz) | Nine SVG icons used by the chat UI | ISC |

Each directory includes its upstream `LICENSE`. The build copies the service
worker and selected icons, with their licenses, into the deployment directory.
Acorn and its walker are build-only and are not deployed.

To update, replace only these files from the new release archives, retain their
licenses, update the versions above, and run:

```bash
node --test tests/web/model_cache_test.mjs
node web/build.mjs build-web
```

Keep vendored files unmodified. Kidi's large-memory glue transform lives in
`web/wasm-glue.mjs`, outside these upstream sources.