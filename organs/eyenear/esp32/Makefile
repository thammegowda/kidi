UV ?= uv
PIO_ENV ?= w11
CHIP ?= esp32s3
PORT ?= /dev/cu.usbmodem1101
BAUD ?= 460800
SECONDS ?= 10
RESOLUTION ?= 2048x1536
CAPTURE_ROOT ?= captures
TIMESTAMP := $(shell date +%Y%m%d-%H%M%S)
CAPTURE ?= $(CAPTURE_ROOT)/capture-$(TIMESTAMP)
OUTPUT ?= $(CAPTURE).mp4
BACKUP ?= backups/flash-$(TIMESTAMP).bin
FACTORY_BACKUP ?= backups/w11-factory-backup-20261003.bin
CONFIRM_RESTORE ?= no
CLANG_FORMAT ?= clang-format

PIO = $(UV) run --no-project --with platformio pio
ESPTOOL = $(UV) run --no-project --with esptool esptool --chip "$(CHIP)" --port "$(PORT)" --baud "$(BAUD)"
CAPTURE_TOOL = $(UV) run --no-project --with esptool python capture.py
ENCODE_TOOL = $(UV) run --no-project --with imageio-ffmpeg python encode_video.py
FIRMWARE_SOURCES = $(wildcard src/*.cpp src/*.h)

.PHONY: build flash status temperature sensors monitor photo audio video av encode backup restore verify-backup clean format format-check test help

build:
	$(PIO) run --project-dir . --environment "$(PIO_ENV)"

flash: build
	$(ESPTOOL) --after hard-reset write-flash --flash-mode keep --flash-freq keep --flash-size keep \
		0x10000 ".pio/build/$(PIO_ENV)/firmware.bin"

status:
	$(CAPTURE_TOOL) status --port "$(PORT)"

temperature:
	$(CAPTURE_TOOL) temperature --port "$(PORT)"

sensors:
	$(CAPTURE_TOOL) sensors --port "$(PORT)"

monitor:
	$(PIO) device monitor --port "$(PORT)" --baud 115200

photo:
	mkdir -p "$(dir $(CAPTURE))"
	$(CAPTURE_TOOL) photo --port "$(PORT)" --resolution "$(RESOLUTION)" --output "$(CAPTURE).jpg"

audio:
	mkdir -p "$(dir $(CAPTURE))"
	$(CAPTURE_TOOL) audio --port "$(PORT)" --seconds "$(SECONDS)" --output "$(CAPTURE).wav"

video:
	mkdir -p "$(dir $(CAPTURE))"
	$(CAPTURE_TOOL) video --port "$(PORT)" --seconds "$(SECONDS)" --output "$(CAPTURE)"
	$(ENCODE_TOOL) "$(CAPTURE)" "$(OUTPUT)"

av:
	mkdir -p "$(dir $(CAPTURE))"
	$(CAPTURE_TOOL) av --port "$(PORT)" --seconds "$(SECONDS)" --output "$(CAPTURE)"
	$(ENCODE_TOOL) "$(CAPTURE)" "$(OUTPUT)"

encode:
	$(ENCODE_TOOL) "$(CAPTURE)" "$(OUTPUT)"

backup:
	@test ! -e "$(BACKUP)" || { echo "Refusing to overwrite backup: $(BACKUP)" >&2; exit 1; }
	mkdir -p "$(dir $(BACKUP))"
	$(ESPTOOL) --after no-reset read-flash 0 ALL "$(BACKUP)"
	$(ESPTOOL) --after hard-reset verify-flash 0 "$(BACKUP)"

restore:
	@test "$(CONFIRM_RESTORE)" = yes || { echo "Restore overwrites firmware. Use CONFIRM_RESTORE=yes to proceed." >&2; exit 1; }
	@test -f "$(FACTORY_BACKUP)" || { echo "Factory backup not found: $(FACTORY_BACKUP)" >&2; exit 1; }
	$(ESPTOOL) --after hard-reset write-flash --flash-mode keep --flash-freq keep --flash-size keep \
		0 "$(FACTORY_BACKUP)"

verify-backup:
	$(ESPTOOL) --after hard-reset verify-flash 0 "$(FACTORY_BACKUP)"

clean:
	$(PIO) run --project-dir . --environment "$(PIO_ENV)" --target clean

format:
	$(CLANG_FORMAT) -i $(FIRMWARE_SOURCES)

format-check:
	$(CLANG_FORMAT) --dry-run --Werror $(FIRMWARE_SOURCES)

test:
	$(UV) run --no-project --with esptool python -m unittest discover -s tests -p 'test_*.py'

help:
	@printf '%s\n' \
		'make build             Compile firmware without flashing (default)' \
		'make flash             Build and update the diagnostic application only' \
		'make status            Query the board over USB' \
		'make temperature       Read ESP32 chip temperature without capturing' \
		'make sensors           Read motion, temperatures, and battery-rail ADC as JSON' \
		'make monitor           Open serial monitor (close before capture/flash)' \
		'make photo             Capture a 2048x1536 JPEG; RESOLUTION=640x480 for a smaller image' \
		'make audio             Capture PCM16 WAV audio' \
		'make video             Capture silent video and encode MP4' \
		'make av                Capture simultaneous audio/video and encode MP4' \
		'make encode            Encode an existing capture directory' \
		'make backup            Save and verify the current full flash' \
		'make verify-backup     Compare factory backup with the current flash' \
		'make restore CONFIRM_RESTORE=yes  Replace firmware with factory backup' \
		'make format-check test Check C++ formatting and host protocol tests' \
		'make clean             Remove build output, not captures or backups' \
		'Overrides: PIO_ENV, CHIP, PORT, BAUD, SECONDS (1-10), RESOLUTION (photos), CAPTURE, OUTPUT, BACKUP, FACTORY_BACKUP'
