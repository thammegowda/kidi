PYTHON ?= python3
KIDI_HUB ?= $(HOME)/.cache/kidi/model-hub
KIDI_BIN ?= build-debug/kidi
KIDI_BACKEND ?= ynnpack
THREADS ?= 8
MIN_CHRF ?= 99.0
PORT ?= 8080
WEB_DIR ?= build-web
APP_CONFIG ?= examples/kidi-app.yaml
APP_PORT ?= 10010
KIDI_CUTLASS_ROOT ?=
TEST_PRESET ?= debug
TEST_BUILD := build-$(TEST_PRESET)
TEST_ENV := .cache/test-venv$(if $(filter debug-cuda,$(TEST_PRESET)),-cuda)
TEST_PYTHON := $(TEST_ENV)/bin/python
TEST_DIST := .cache/test-dist$(if $(filter debug-cuda,$(TEST_PRESET)),-cuda)
TEST_CUDA := $(if $(filter debug-cuda,$(TEST_PRESET)),ON,OFF)
TEST_BACKEND := $(if $(filter debug-cuda,$(TEST_PRESET)),cuda,$(KIDI_BACKEND))
CUDA_CMAKE_ARGS = $(if $(KIDI_CUTLASS_ROOT),-DKIDI_CUTLASS_ROOT="$(abspath $(KIDI_CUTLASS_ROOT))")

export PATH := $(CURDIR)/.cache/tools/bin:$(PATH)
export KIDI_HUB PYTHON

.PHONY: setup setup-test debug debug-cuda release release-cuda build-test build-python-test prepare-test native-test test-cuda ynnpack-test python-test web-deps web-test-deps web-test web-smoke-test extension-smoke-test extension-gmail-test extension-article-test extension-pdf-test extension-edge-test extension-edge-e2e-test extension-edge-open extension-edge-open-local extension-edge-load-local extension-edge-summary-local regression-test test app app-dev wasm extension serve apk apk-release

refresh:
	git submodule update --init --recursive --depth 10 --jobs 12

setup:
	bash scripts/setup.sh

.PHONY: setup-node
setup-node:
	bash scripts/setup.sh --node-only

setup-test:
	KIDI_TEST_PRESET="$(TEST_PRESET)" bash scripts/setup.sh --tests-only

debug debug-cuda release release-cuda:
	cmake --preset $@ $(if $(filter %-cuda,$@),$(CUDA_CMAKE_ARGS))
	cmake --build --preset $@

build-test: debug

native-test: debug
	ctest --preset debug -L native

ynnpack-test:
	$(MAKE) -C third_party/ynnpack-dev test

build-python-test: setup-test
	rm -rf "$(TEST_DIST)"
	$(TEST_PYTHON) -m pip wheel --disable-pip-version-check --no-deps --no-build-isolation . \
		--wheel-dir "$(TEST_DIST)" -Cbuild-dir="$(TEST_BUILD)/python" \
		-Ccmake.args=--preset=$(TEST_PRESET) -Ccmake.build-type=Debug -Cinstall.strip=false \
		-Ccmake.define.KIDI_ENABLE_CUDA=$(TEST_CUDA) \
		$(if $(filter ON,$(TEST_CUDA)),$(if $(KIDI_CUTLASS_ROOT),-Ccmake.define.KIDI_CUTLASS_ROOT="$(abspath $(KIDI_CUTLASS_ROOT))"))
	wheel=$$(find "$(TEST_DIST)" -name 'kidi-*.whl' -print -quit); test -n "$$wheel"; $(TEST_PYTHON) -m pip install --disable-pip-version-check --force-reinstall --no-deps "$$wheel"

python-test: build-python-test
	$(TEST_PYTHON) -m unittest discover -s tests -p '*_test.py'

web-deps: setup-node
	node web/tools/prepare-libs.mjs

web-test-deps: setup-node
	node web/tools/prepare-libs.mjs --tests

web-test: web-deps
	node --test tests/browser/*_test.mjs third_party/liteapp/tests/browser/*_test.mjs

web-smoke-test: web-test-deps
	npm --prefix web run test:pages -- "$(abspath $(WEB_DIR))"

regression-test: setup-test debug
	$(TEST_PYTHON) tests/rtg_regression.py setup
	$(TEST_PYTHON) tests/rtg_regression.py run --binary "$(KIDI_BIN)" --backend "$(KIDI_BACKEND)" --threads "$(THREADS)" --min-chrf "$(MIN_CHRF)"

prepare-test: build-python-test
	$(if $(filter debug,$(TEST_PRESET)),$(TEST_PYTHON) tests/rtg_regression.py setup)
	cmake --preset $(TEST_PRESET) -DKIDI_BUILD_INTEGRATION_TESTS=ON \
		-DKIDI_TEST_PYTHON="$(abspath $(TEST_PYTHON))" \
		-DKIDI_TEST_BACKEND="$(TEST_BACKEND)" -DKIDI_TEST_THREADS="$(THREADS)" -DKIDI_TEST_MIN_CHRF="$(MIN_CHRF)" \
		$(if $(filter ON,$(TEST_CUDA)),$(CUDA_CMAKE_ARGS))
	cmake --build --preset $(TEST_PRESET)

test test-cuda:
	$(MAKE) prepare-test TEST_PRESET=$(patsubst test%,debug%,$@)
	ctest --preset $(patsubst test%,debug%,$@)

app:
	cmake --preset release -DKIDI_BUILD_DESKTOP_APP=ON
	cmake --build --preset release --target kidi_app

app-dev: app
	build-release/kidi-app "$(APP_CONFIG)" --port "$(APP_PORT)" --dev

wasm:
	node web/tools/build.mjs "$(WEB_DIR)"

extension:
	node web/tools/build-extension.mjs

extension-smoke-test:
	npm --prefix web run test:extension -- "$(abspath build-extension/chromium)"

extension-gmail-test:
	npm --prefix web run test:gmail -- "$(abspath build-extension/chromium)"
	node web/tools/test-editor-wasm.mjs

extension-article-test:
	npm --prefix web run test:articles -- "$(abspath build-extension/chromium)"

extension-pdf-test:
	KIDI_LIVE_PAGE_URL=https://arxiv.org/pdf/1706.03762 KIDI_LIVE_EXPECT='Attention' \
		npm --prefix web run test:live-wikipedia

extension-edge-test:
	KIDI_BROWSER_CHANNEL=msedge npm --prefix web run test:extension -- "$(abspath build-extension/chromium)"
	KIDI_BROWSER_CHANNEL=msedge npm --prefix web run test:gmail -- "$(abspath build-extension/chromium)"
	node web/tools/test-editor-wasm.mjs
	KIDI_BROWSER_CHANNEL=msedge npm --prefix web run test:articles -- "$(abspath build-extension/chromium)"

extension-edge-e2e-test: extension-edge-test
	KIDI_BROWSER_CHANNEL=msedge npm --prefix web run test:live-wikipedia

extension-edge-open:
	KIDI_EDGE_PROFILE="$(HOME)/.cache/kidi/edge-extension-profile" \
		node web/tools/open-edge-extension.mjs "$(abspath build-extension/chromium)"

extension-edge-open-local:
	KIDI_EDGE_PROFILE="$(HOME)/.cache/kidi/edge-extension-profile" \
		KIDI_LOCAL_MODEL_CACHE="$(KIDI_HUB)" \
		node web/tools/open-edge-extension.mjs "$(abspath build-extension/chromium)"

extension-edge-load-local: extension-edge-open-local
	KIDI_EDGE_PROFILE="$(HOME)/.cache/kidi/edge-extension-profile" \
		npm --prefix web run test:edge-model-load

extension-edge-summary-local: extension-edge-open-local
	KIDI_EDGE_PROFILE="$(HOME)/.cache/kidi/edge-extension-profile" \
		KIDI_MODEL_LOAD_TIMEOUT_MS=900000 npm --prefix web run test:edge-summary

serve:
	$(PYTHON) -m http.server "$(PORT)" --bind 127.0.0.1 --directory "$(WEB_DIR)"

apk:
	bash scripts/build-apk.sh

apk-release:
	bash scripts/build-apk.sh --release