PYTHON ?= python3
KIDI_HUB ?= $(HOME)/.cache/kidi/model-hub
KIDI_BIN ?= build-release/kidi
BACKEND ?= ynnpack
THREADS ?= 8
MIN_CHRF ?= 99.0
PORT ?= 8080
WEB_DIR ?= build-web
TEST_PYTHON := .cache/test-venv/bin/python
TEST_DIST := .cache/test-dist

export KIDI_HUB PYTHON

.PHONY: setup-test build-test build-python-test prepare-test native-test python-test web-test regression-test test wasm serve

setup-test:
	bash tests/setup.sh

build-test:
	cmake --preset release
	cmake --build --preset release --target kidi_cli

native-test:
	cmake --preset debug -DKIDI_BUILD_INTEGRATION_TESTS=OFF
	cmake --build --preset debug
	ctest --preset debug -L native

build-python-test: setup-test
	rm -rf "$(TEST_DIST)"
	$(TEST_PYTHON) -m pip wheel --disable-pip-version-check --no-deps . --wheel-dir "$(TEST_DIST)"
	wheel=$$(find "$(TEST_DIST)" -name 'kidi-*.whl' -print -quit); test -n "$$wheel"; $(TEST_PYTHON) -m pip install --disable-pip-version-check --force-reinstall --no-deps "$$wheel"

python-test: build-python-test
	$(TEST_PYTHON) -m unittest discover -s tests -p python_cli_test.py

web-test:
	node --test tests/web/*.mjs

regression-test: setup-test build-test
	$(TEST_PYTHON) tests/rtg_regression.py run --binary "$(KIDI_BIN)" --backend "$(BACKEND)" --threads "$(THREADS)" --min-chrf "$(MIN_CHRF)"

prepare-test: build-test build-python-test
	cmake --preset debug -DKIDI_BUILD_INTEGRATION_TESTS=ON \
		-DKIDI_TEST_PYTHON="$(abspath $(TEST_PYTHON))" -DKIDI_TEST_BINARY="$(abspath $(KIDI_BIN))" \
		-DKIDI_TEST_BACKEND="$(BACKEND)" -DKIDI_TEST_THREADS="$(THREADS)" -DKIDI_TEST_MIN_CHRF="$(MIN_CHRF)"
	cmake --build --preset debug

test: prepare-test
	ctest --preset debug

wasm:
	node web/build.mjs "$(WEB_DIR)"

serve:
	$(PYTHON) -m http.server "$(PORT)" --bind 127.0.0.1 --directory "$(WEB_DIR)"