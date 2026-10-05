UV ?= uv
PIO_ENV ?= w11
CHIP ?= esp32s3
PORT ?= auto
BAUD ?= 460800
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
BACKUP ?= backups/flash-$(TIMESTAMP).bin
FACTORY_BACKUP ?= backups/w11-factory-backup-20261003.bin
CONFIRM_RESTORE ?= no
CLANG_FORMAT ?= clang-format

PIO = $(UV) run --no-project --with platformio pio
ESPTOOL = $(UV) run --no-project --with esptool esptool --chip "$(CHIP)" --port "$$port" --baud "$(BAUD)"
CAPTURE_TOOL = $(UV) run --no-project --with esptool python capture.py
WIFI_TOOL = $(UV) run --no-project --with-requirements requirements-wifi.txt python wifi.py
LIVE_TOOL = $(UV) run --no-project --with-requirements requirements-wifi.txt python live.py
SSID_OPTION = $(if $(WIFI_SSID),--ssid "$(WIFI_SSID)",)
ENCODE_TOOL = $(UV) run --no-project --with imageio-ffmpeg python encode_video.py
FIRMWARE_SOURCES = $(wildcard src/*.cpp src/*.h)

.PHONY: build flash dev-connect cache-wifi forget-wifi-cache speed-test status temperature sensors monitor photo audio video av encode backup restore verify-backup clean format format-check test help

build:
	$(PIO) run --project-dir . --environment "$(PIO_ENV)"

flash: build
	port=$$($(WIFI_TOOL) port --port "$(PORT)") && \
	$(ESPTOOL) --after hard-reset write-flash --flash-mode keep --flash-freq keep --flash-size keep \
		0x10000 ".pio/build/$(PIO_ENV)/firmware.bin"

status:
	$(CAPTURE_TOOL) status --port "$(PORT)"

temperature:
	$(CAPTURE_TOOL) temperature --port "$(PORT)"

sensors:
	$(CAPTURE_TOOL) sensors --port "$(PORT)"

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

av:
	$(LIVE_TOOL) --mode av --port "$(PORT)" --transport "$(TRANSPORT)" --seconds "$(STREAM_SECONDS)" --resolution "$(STREAM_RESOLUTION)"

encode:
	$(ENCODE_TOOL) "$(CAPTURE)" "$(OUTPUT)"

backup:
	@test ! -e "$(BACKUP)" || { echo "Refusing to overwrite backup: $(BACKUP)" >&2; exit 1; }
	mkdir -p "$(dir $(BACKUP))"
	port=$$($(WIFI_TOOL) port --port "$(PORT)") && \
	$(ESPTOOL) --after no-reset read-flash 0 ALL "$(BACKUP)" && \
	$(ESPTOOL) --after hard-reset verify-flash 0 "$(BACKUP)"

restore:
	@test "$(CONFIRM_RESTORE)" = yes || { echo "Restore overwrites firmware. Use CONFIRM_RESTORE=yes to proceed." >&2; exit 1; }
	@test -f "$(FACTORY_BACKUP)" || { echo "Factory backup not found: $(FACTORY_BACKUP)" >&2; exit 1; }
	port=$$($(WIFI_TOOL) port --port "$(PORT)") && \
	$(ESPTOOL) --after hard-reset write-flash --flash-mode keep --flash-freq keep --flash-size keep \
		0 "$(FACTORY_BACKUP)"

verify-backup:
	port=$$($(WIFI_TOOL) port --port "$(PORT)") && $(ESPTOOL) --after hard-reset verify-flash 0 "$(FACTORY_BACKUP)"

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
		'make build             Compile firmware without flashing (default)' \
		'make flash             Build and update the diagnostic application only' \
		'make dev-connect       Configure the current Wi-Fi over trusted USB' \
		'make cache-wifi        Cache the selected network credential once, owner-only' \
		'make forget-wifi-cache Remove cached laptop Wi-Fi passwords' \
		'make speed-test        Verified HTTPS download/upload throughput, no media activity' \
		'make status            Query the board over USB' \
		'make temperature       Read ESP32 chip temperature without capturing' \
		'make sensors           Read motion, temperatures, and battery-rail ADC as JSON' \
		'make monitor           Open serial monitor (close before capture/flash)' \
		'make photo             Prefer authenticated Wi-Fi; explicitly fall back to USB when unavailable' \
		'make audio             Wi-Fi-first PCM16 WAV recording (1-10 seconds)' \
		'make video             Live 720p browser preview, no recording' \
		'make av                Live 720p/48 kHz browser preview, no recording' \
		'make encode            Encode an existing capture directory' \
		'make backup            Save and verify the current full flash' \
		'make verify-backup     Compare factory backup with the current flash' \
		'make restore CONFIRM_RESTORE=yes  Replace firmware with factory backup' \
		'make format-check test Check C++ formatting and host protocol tests' \
		'make clean             Remove build output, not captures or backups' \
		'Overrides: PIO_ENV, CHIP, PORT, BAUD, TRANSPORT, SECONDS (audio), STREAM_SECONDS (1-300), STREAM_RESOLUTION (1280x720/96x96), RESOLUTION (photos), CAPTURE'
