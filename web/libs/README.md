# Web Dependency Policy

Vendor compact, self-contained upstream release files and selected icons.
Acquire bulky distributions instead of checking their file trees into Git.
Keep both kinds unmodified and package them locally with their licenses.

## Vendored releases

These files are checked in. Versions, source archives, and SHA-512 archive
integrity values are pinned below; npm acquisition never updates these copies.

| Directory | Version | Source Archive | Included Files | License |
|---|---|---|---|---|
| `acorn/` | 8.15.0 | [acorn-8.15.0.tgz](https://registry.npmjs.org/acorn/-/acorn-8.15.0.tgz) | `dist/acorn.mjs` as `acorn.mjs` | MIT |
| `acorn-walk/` | 8.3.4 | [acorn-walk-8.3.4.tgz](https://registry.npmjs.org/acorn-walk/-/acorn-walk-8.3.4.tgz) | `dist/walk.mjs` as `walk.mjs` | MIT |
| `coi-serviceworker/` | 0.1.7 | [coi-serviceworker-0.1.7.tgz](https://registry.npmjs.org/coi-serviceworker/-/coi-serviceworker-0.1.7.tgz) | `coi-serviceworker.js` | MIT |
| `dompurify/` | 3.4.15 | [dompurify-3.4.15.tgz](https://registry.npmjs.org/dompurify/-/dompurify-3.4.15.tgz) | `dist/purify.es.mjs` as `purify.es.mjs` | Apache-2.0 OR MPL-2.0 |
| `highlightjs/` | 11.12.0 | [cdn-assets-11.12.0.tgz](https://registry.npmjs.org/@highlightjs/cdn-assets/-/cdn-assets-11.12.0.tgz) | `es/highlight.min.js` as `highlight.min.js` | BSD-3-Clause |
| `js-yaml/` | 4.1.0 | [js-yaml-4.1.0.tgz](https://registry.npmjs.org/js-yaml/-/js-yaml-4.1.0.tgz) | `dist/js-yaml.mjs` as `js-yaml.mjs` | MIT |
| `lucide-static/` | 0.468.0 | [lucide-static-0.468.0.tgz](https://registry.npmjs.org/lucide-static/-/lucide-static-0.468.0.tgz) | Ten selected SVG icons | ISC |
| `marked/` | 18.0.13 | [marked-18.0.13.tgz](https://registry.npmjs.org/marked/-/marked-18.0.13.tgz) | `lib/marked.esm.js` as `marked.esm.js` | MIT |
| `readability/` | 0.6.0 | [readability-0.6.0.tgz](https://registry.npmjs.org/@mozilla/readability/-/readability-0.6.0.tgz) | `Readability.js` | Apache-2.0 |

```text
acorn-8.15.0.tgz       sha512-NZyJarBfL7nWwIq+FDL6Zp/yHEhePMNnnJ0y3qfieCrmNvYct8uvtiV41UvlSe6apAfk0fY1FbWx+NwfmpvtTg==
acorn-walk-8.3.4.tgz   sha512-ueEepnujpqee2o5aIYnvHU6C0A42MNdsIDeqy5BydrkuC5R1ZuUFnm27EeFJGoEHJQgn3uleRvmTXaJgfXbt4g==
coi-serviceworker-0.1.7.tgz sha512-bjSUqEngCPOkErY2vbyWsaIGCNRODYzlNycaREVw5s12/C8SM+RnRUUeX6pZbTtov6C52ZLY/+tvHK+BDxuUuA==
dompurify-3.4.15.tgz   sha512-EUBjM+B+lkDE41iE82DDSCfkoPGfXx8IxFxPMjNzm/Uk4xDet77rTN9wqlxlVg71kK7XGuUMv6wUxJUwwv+Xyw==
cdn-assets-11.12.0.tgz sha512-KvOKXODaiFmId9xaq3xc5xCL66wVLUuOngDbO9B/kewbFTqdGbn2nJxNhN3H5R1cgDTVj6R8vH0zgiNDEGjpDw==
js-yaml-4.1.0.tgz      sha512-wpxZs9NoxZaJESJGIZTyDEaYpl0FKSA+FB9aJiyemKhMwkxQg63h4T1KJgUGHpTqPDNRcmmYLugrRjJlBtWvRA==
lucide-static-0.468.0.tgz sha512-JvpWui2umxRyEVMoETfMzb+qKqibV/sdoqJbKmW1JLdkuhXluJKoO6NqCbvCK/vAbUuH5bTEFD4T6uECsrNcnA==
marked-18.0.13.tgz     sha512-xTxVzZsBFwunP6HDmtBkabUQEYArnP7/rMDGmPj9SlrKlQ4i8MdYVow+nJL0eOqwpUqhzBoTBRADGN6uYwPyOw==
readability-0.6.0.tgz  sha512-juG5VWh4qAivzTAeMzvY9xs9HY5rAcr2E4I7tiSSCokRFi7XIZCAu92ZkSTsIj1OPceCifL3cpfteP3pDT9/QQ==
```

Each directory retains its upstream license files. Readability's `LICENSE.md`
is named `LICENSE` locally. Acorn and its walker are build-only inputs to the
Wasm heap-index transformation and are not deployed.

To update a vendored release, download its upstream archive, verify its
checksum, replace only the selected files and licenses, and update the table
and checksum together. Run `make web-test` before committing.

## Acquired distributions

Mermaid's large standalone bundle and PDF.js's multi-file distribution are
acquired from exact versions in [`package.json`](../package.json).
[`package-lock.json`](../package-lock.json) locks their transitive dependencies
and archive integrity hashes.

With Node.js 24+ and npm, run from the repository root:

```bash
make web-deps
```

[`prepare-libs.mjs`](../tools/prepare-libs.mjs) runs `npm ci --include=dev
--ignore-scripts --no-audit --no-fund` when the installed lock changes, then stages
only Mermaid and PDF.js files here. Package installation scripts are disabled.
Their directories, preparation metadata, and `web/node_modules/` are ignored
by Git. The preparation step must never overwrite vendored libraries or icons.

Browser builds and packaging, CMake desktop assets, and Gradle Android assets
invoke the same preparation step. CLI-only and Python wheel targets do not
acquire frontend dependencies. Initial acquisition needs network access;
unchanged builds reuse prepared files, and npm's cache can support offline
reacquisition. Missing staged files are regenerated.

Application URLs remain local and unchanged. Deployment packages include their
required licenses; PDF extraction also retains packed CMaps, standard font
data, and bundled font licenses. Desktop and Android still require Node.js/npm
at build time because Mermaid is acquired rather than vendored.

To update an acquired package deliberately:

```bash
npm install --prefix web --save-exact PACKAGE@VERSION --ignore-scripts --no-audit --no-fund
make web-deps
make web-test
```

Review the manifest and lockfile together. Do not edit generated distributions
or replace local deployment assets with runtime CDN imports.