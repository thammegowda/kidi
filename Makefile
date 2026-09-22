PYTHON ?= python3
KIDI_HUB ?= $(HOME)/.cache/kidi/model-hub
KIDI_BIN ?= build-release/kidi
BACKEND ?= ynnpack
THREADS ?= 8
MIN_CHRF ?= 99.0
PORT ?= 8080
WEB_DIR ?= build-web
TEST_PYTHON := .cache/test-venv/bin/python

export KIDI_HUB PYTHON

.PHONY: setup-test regression-test test build-test wasm serve

setup-test:
	bash tests/setup.sh

build-test:
	cmake --preset release
	cmake --build --preset release --target kidi_cli

regression-test: setup-test build-test
	$(TEST_PYTHON) tests/rtg_regression.py run --binary "$(KIDI_BIN)" --backend "$(BACKEND)" --threads "$(THREADS)" --min-chrf "$(MIN_CHRF)"

test: regression-test

wasm:
	node web/build.mjs "$(WEB_DIR)"

serve:
	$(PYTHON) -m http.server "$(PORT)" --bind 127.0.0.1 --directory "$(WEB_DIR)"