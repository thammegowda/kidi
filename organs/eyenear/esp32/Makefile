UV ?= uv
PIO_ENV ?= esp32s3_compile_check
PORT ?= auto
SECONDS ?= 10
RESOLUTION ?= 2048x1536
TRANSPORT ?= auto
STREAM_SECONDS ?= 300
STREAM_RESOLUTION ?= 1280x720
TRANSFER_BYTES ?= 262144
BLOCK_BYTES ?= 4096
WIFI_SSID ?=
CAPTURE_ROOT ?= captures
TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)
CAPTURE ?= $(CAPTURE_ROOT)/capture-$(TIMESTAMP)
OUTPUT ?= $(CAPTURE).mp4
CLANG_FORMAT ?= clang-format

PIO = $(UV) run --no-project --with platformio pio
CAPTURE_TOOL = $(UV) run --no-project --with-requirements requirements-wifi.txt python capture.py
WIFI_TOOL = $(UV) run --no-project --with-requirements requirements-wifi.txt python wifi.py
LIVE_TOOL = $(UV) run --no-project --with-requirements requirements-wifi.txt python live.py
SSID_OPTION = $(if $(WIFI_SSID),--ssid "$(WIFI_SSID)",)
ENCODE_TOOL = $(UV) run --no-project --with imageio-ffmpeg python encode_video.py
FIRMWARE_SOURCES = $(wildcard src/*.cpp src/*.h tests/firmware/*.cpp)

.DEFAULT_GOAL := help
.PHONY: build compile-check dev-connect cache-wifi forget-wifi-cache speed-test status temperature monitor photo audio video video-check av encode clean format format-check test help

build: compile-check

compile-check:
	@printf '%s\n' 'Compile-only ESP32-S3 SDK adapter check; no board pin map or P4 firmware target is selected.'
	$(PIO) run --project-dir . --environment "$(PIO_ENV)"

status:
	$(CAPTURE_TOOL) status --port "$(PORT)"

temperature:
	$(CAPTURE_TOOL) temperature --port "$(PORT)"

monitor:
	port=$$($(WIFI_TOOL) port --port "$(PORT)") && $(PIO) device monitor --port "$$port" --baud 115200

dev-connect:
	$(WIFI_TOOL) setup --port "$(PORT)" $(SSID_OPTION)

cache-wifi:
	$(WIFI_TOOL) cache-network --authorize-credentials $(SSID_OPTION)

forget-wifi-cache:
	$(WIFI_TOOL) forget-wifi-cache

speed-test:
	$(UV) run --no-project --with-requirements requirements-wifi.txt python speed.py --bytes "$(TRANSFER_BYTES)" --block-size "$(BLOCK_BYTES)"

photo:
	mkdir -p "$(dir $(CAPTURE))"
	$(WIFI_TOOL) photo --port "$(PORT)" --transport "$(TRANSPORT)" --resolution "$(RESOLUTION)" --output "$(CAPTURE).jpg"

audio:
	mkdir -p "$(dir $(CAPTURE))"
	$(WIFI_TOOL) audio --port "$(PORT)" --transport "$(TRANSPORT)" --seconds "$(SECONDS)" --output "$(CAPTURE).wav"

video:
	$(LIVE_TOOL) --mode video --port "$(PORT)" --transport "$(TRANSPORT)" --seconds "$(STREAM_SECONDS)" --resolution "$(STREAM_RESOLUTION)"

video-check:
	$(LIVE_TOOL) --mode video --port "$(PORT)" --transport "$(TRANSPORT)" --seconds "$(SECONDS)" --resolution "$(STREAM_RESOLUTION)" --check --min-fps 9.5

av:
	$(LIVE_TOOL) --mode av --port "$(PORT)" --transport "$(TRANSPORT)" --seconds "$(STREAM_SECONDS)" --resolution "$(STREAM_RESOLUTION)"

encode:
	$(ENCODE_TOOL) "$(CAPTURE)" "$(OUTPUT)"

clean:
	$(PIO) run --project-dir . --environment "$(PIO_ENV)" --target clean

format:
	$(CLANG_FORMAT) -i $(FIRMWARE_SOURCES)

format-check:
	$(CLANG_FORMAT) --dry-run --Werror $(FIRMWARE_SOURCES)

test:
	$(UV) run --no-project --with-requirements requirements-wifi.txt python -m unittest discover -s tests -p 'test_*.py'

help:
	@printf '%s\n' \
		'make help              Show supported commands (default)' \
		'make compile-check     Compile SDK adapters without selecting board pins; not P4 firmware' \
		'make build             Alias for compile-check; never flashes hardware' \
		'make dev-connect       Configure the current Wi-Fi over trusted USB' \
		'make cache-wifi        Cache the selected network credential once, owner-only' \
		'make forget-wifi-cache Remove cached laptop Wi-Fi passwords' \
		'make speed-test        Verified HTTPS download/upload throughput, no media activity' \
		'make status            Query the board over USB' \
		'make temperature       Read ESP32 chip temperature without capturing' \
		'make monitor           Open serial monitor (close before capture/flash)' \
		'make photo             Prefer authenticated Wi-Fi; explicitly fall back to USB when unavailable' \
		'make audio             Wi-Fi-first PCM16 WAV recording (1-10 seconds)' \
		'make video             Video-only 720p/10 fps browser preview, microphone off' \
		'make video-check       Headless video-only check, requires at least 9.5 delivered fps' \
		'make av                Explicit experimental 720p/16 kHz audiovisual preview' \
		'make encode            Encode an existing capture directory' \
		'make format-check test Check C++ formatting and host protocol tests' \
		'make clean             Remove build output, not captures or backups' \
		'No flash/restore target exists until a new board-specific application is implemented.' \
		'Capture/setup commands require independently installed, configured compatible firmware.' \
		'Overrides: PIO_ENV (compile check), PORT, TRANSPORT, SECONDS (audio/video-check), STREAM_SECONDS (1-300), STREAM_RESOLUTION (1280x720/96x96), RESOLUTION (photos), CAPTURE'
