import argparse
from array import array
import json
import math
from pathlib import Path
import sys
import time
import wave

import serial
from serial.tools import list_ports

SAMPLE_RATE = 16000


def choose_port(requested):
    if requested and requested != "auto":
        return requested
    ports = [port.device for port in list_ports.comports() if port.vid == 0x303A and port.pid == 0x1001]
    if len(ports) != 1:
        raise ValueError("Connect one ESP32 accessory or select its serial port with PORT=...")
    return ports[0]


def read_exact(port, length, deadline):
    result = bytearray()
    while len(result) < length:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Received {len(result)} of {length} bytes")
        chunk = port.read(min(length - len(result), 8192))
        if chunk:
            result.extend(chunk)
    return bytes(result)


def read_payload(port, length, deadline):
    if not 0 < length <= 1024 * 1024:
        raise ValueError(f"Invalid payload length: {length}")
    payload = read_exact(port, length, deadline)
    if port.readline().strip() or port.readline().strip() != b"KIDI_END":
        raise ValueError("Invalid binary payload boundary")
    return payload


def read_clip_end(port):
    line = port.readline().strip()
    if line != b"KIDI_CLIP_END":
        raise ValueError("Missing successful clip completion marker")
    port.finish()


def capture(port, command, resolution=None, send_command=True, require_end=False):
    if send_command:
        port.write((command + "\n").encode("ascii"))
        port.flush()
    deadline = time.monotonic() + 30
    reported_resolution = None
    audio_format_verified = False
    while time.monotonic() < deadline:
        raw_line = port.readline()
        try:
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise ValueError("Unexpected binary data in the control stream; USB framing was lost") from error
        if not line:
            continue
        print(line, flush=True)
        if line.startswith("KIDI_ERROR"):
            raise RuntimeError(line)
        if command == "status" and line.startswith("KIDI_STATUS"):
            return None
        if command == "temperature" and line.startswith("KIDI_TEMPERATURE"):
            return None
        if command == "sensors" and line.startswith("KIDI_SENSORS "):
            snapshot = json.loads(line.removeprefix("KIDI_SENSORS "))
            if snapshot["schema_version"] != 1 or snapshot["timestamp_us"] < 0:
                raise ValueError("Unsupported sensor snapshot schema or timestamp")
            for name in ("chip_temperature", "imu", "board_temperature", "battery_rail"):
                reading = snapshot[name]
                if not reading["available"]:
                    print(f"{name}: {reading['error']}", file=sys.stderr)
            if "i2c_error" in snapshot:
                print(snapshot["i2c_error"], file=sys.stderr)
            return None
        if line.startswith("KIDI_PHOTO "):
            fields = dict(field.split("=") for field in line.split()[1:])
            reported_resolution = f"{fields['width']}x{fields['height']}"
        if line.startswith("KIDI_AUDIO_FORMAT "):
            fields = dict(field.split("=") for field in line.split()[1:])
            if fields != {"sample_rate": str(SAMPLE_RATE), "channels": "1", "bits": "16"}:
                raise ValueError(f"Unsupported microphone format: {fields}")
            audio_format_verified = True
        if not line.startswith("KIDI_BEGIN "):
            continue
        if resolution is not None and reported_resolution != resolution:
            raise ValueError(f"Requested {resolution}, but camera reported {reported_resolution}")
        if command.startswith("audio ") and not audio_format_verified:
            raise ValueError("Firmware did not confirm the microphone sample format; update it with make flash")
        _, kind, size = line.split()
        length = int(size)
        payload = read_payload(port, length, deadline)
        if require_end:
            read_clip_end(port)
        return kind, payload
    raise TimeoutError("No diagnostic response; check firmware and serial port")


def capture_video(port, seconds, output, with_audio=False, send_command=True, require_end=False):
    output.mkdir()
    frames = []
    if send_command:
        port.write(f"{'av' if with_audio else 'video'} {seconds}\n".encode("ascii"))
        port.flush()
    deadline = time.monotonic() + seconds + 30
    previous_timestamp = -1
    manifest = None
    audio_metadata = None
    audio_received = False
    while time.monotonic() < deadline:
        raw_line = port.readline()
        try:
            line = raw_line.decode("utf-8").strip()
        except UnicodeDecodeError as error:
            raise ValueError("Unexpected binary data in the control stream; USB framing was lost") from error
        if not line:
            continue
        if line.startswith("KIDI_ERROR"):
            raise RuntimeError(line)
        if line.startswith("KIDI_FRAME "):
            _, timestamp, size = line.split()
            timestamp = int(timestamp)
            length = int(size)
            if timestamp <= previous_timestamp or not 0 < length <= 1024 * 1024:
                raise ValueError("Invalid video frame timing or payload size")
            payload = read_payload(port, length, deadline)
            if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
                raise ValueError("Incomplete video JPEG")
            filename = f"frame-{len(frames):04d}.jpg"
            (output / filename).write_bytes(payload)
            frames.append({"filename": filename, "timestamp_us": timestamp, "bytes": length})
            previous_timestamp = timestamp
            if len(frames) % 20 == 0:
                print(f"Received {len(frames)} frames", flush=True)
        elif line.startswith("KIDI_VIDEO_END "):
            fields = dict(field.split("=") for field in line.split()[1:])
            duration = int(fields["duration_us"])
            if int(fields["frames"]) != len(frames) or len(frames) < 2:
                raise ValueError(
                    f"Video frame count invalid: received {len(frames)}, reported {fields['frames']}; at least 2 required"
                )
            if not seconds * 1000000 <= duration <= (seconds + 2) * 1000000:
                raise ValueError(f"Unexpected capture duration: {duration} us")
            manifest = {
                "duration_us": duration,
                "requested_duration_us": seconds * 1000000,
                "frames": frames,
            }
            print(f"Video capture: {len(frames)} frames over {duration / 1000000:.3f}s")
            print(f"Saved frames and timing: {output}")
            if not with_audio:
                if require_end:
                    read_clip_end(port)
                (output / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")
                return
        elif with_audio and line.startswith("KIDI_AUDIO_META "):
            audio_metadata = {
                key: int(value)
                for key, value in (field.split("=") for field in line.split()[1:])
            }
            if audio_metadata["samples"] != seconds * SAMPLE_RATE or audio_metadata["overruns"] != 0:
                raise ValueError("Missing microphone samples or DMA overruns")
            if audio_metadata["sample_rate"] != SAMPLE_RATE:
                raise ValueError("Unexpected microphone sample rate")
            measured = audio_metadata["duration_us"]
            if abs(measured - seconds * 1000000) > 200000:
                raise ValueError(f"Unexpected microphone clock duration: {measured} us")
            if audio_metadata["start_us"] != 0:
                raise ValueError("Audio does not start on the shared capture timeline")
            print(line, flush=True)
        elif with_audio and line.startswith("KIDI_BEGIN "):
            _, kind, size = line.split()
            if audio_metadata is None or kind != "pcm16" or int(size) != seconds * SAMPLE_RATE * 2:
                raise ValueError("Invalid concurrent microphone payload")
            save_audio(output / "audio.wav", read_payload(port, int(size), deadline))
            audio_received = True
        elif with_audio and line == "KIDI_AV_END":
            if manifest is None or audio_metadata is None or not audio_received:
                raise ValueError("Incomplete concurrent AV capture")
            if require_end:
                read_clip_end(port)
            manifest["audio"] = {"filename": "audio.wav", **audio_metadata}
            (output / "capture.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print("Simultaneous audio/video capture complete", flush=True)
            return
        else:
            print(line, flush=True)
    raise TimeoutError("Incomplete video capture")


def save_audio(path, payload):
    if len(payload) % 2:
        raise ValueError("PCM payload contains an incomplete sample")
    samples = array("h", payload)
    if sys.byteorder != "little":
        samples.byteswap()
    mean = sum(samples) / len(samples)
    ac_rms = math.sqrt(sum((sample - mean) ** 2 for sample in samples) / len(samples))
    clipped = sum(abs(sample) >= 32767 for sample in samples)
    print(
        f"Audio: {len(samples) / SAMPLE_RATE:.3f}s, {SAMPLE_RATE} Hz mono PCM16; "
        f"min={min(samples)} max={max(samples)} mean={mean:.1f} "
        f"AC RMS={ac_rms:.1f} clipped={clipped}/{len(samples)}"
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(payload)
    centered = array(
        "h", (max(-32768, min(32767, round(sample - mean))) for sample in samples)
    )
    if sys.byteorder != "little":
        centered.byteswap()
    centered_path = path.with_name(path.stem + "-dc-removed.wav")
    with wave.open(str(centered_path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(centered.tobytes())
    print(f"Saved raw audio: {path}")
    print(f"Saved DC-removed playback audio: {centered_path}")
    if ac_rms == 0:
        raise RuntimeError("Microphone produced a constant signal, not usable audio")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("status", "temperature", "sensors", "photo", "audio", "video", "av"))
    parser.add_argument("--port", default="auto")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument(
        "--resolution", choices=("640x480", "2048x1536"),
        help="Photo resolution; default is 2048x1536",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.kind not in ("status", "temperature", "sensors") and args.output is None:
        parser.error("--output is required for captures")
    if not 1 <= args.seconds <= 10:
        parser.error("--seconds must be 1 to 10")
    if args.resolution is not None and args.kind != "photo":
        parser.error("--resolution is only supported for photo captures")
    if args.kind == "photo" and args.resolution is None:
        args.resolution = "2048x1536"
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    port = serial.Serial(port=None, baudrate=115200, timeout=0.5, write_timeout=5)
    port.dtr = True
    port.rts = False
    port.port = choose_port(args.port)
    port.open()
    try:
        time.sleep(1)
        if args.kind in ("video", "av"):
            capture_video(port, args.seconds, args.output, with_audio=args.kind == "av")
        else:
            command = f"audio {args.seconds}" if args.kind == "audio" else args.kind
            if args.kind == "photo":
                command = f"photo {args.resolution}"
            result = capture(port, command, resolution=args.resolution)
    finally:
        port.close()
    if args.kind in ("status", "temperature", "sensors", "video", "av"):
        return
    kind, payload = result
    if args.kind == "photo":
        if kind != "jpeg" or not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
            raise ValueError("Camera payload is not a complete JPEG")
        args.output.write_bytes(payload)
        print(f"Saved JPEG: {args.output} ({len(payload)} bytes)")
    else:
        if kind != "pcm16" or len(payload) != args.seconds * SAMPLE_RATE * 2:
            raise ValueError("Unexpected audio format or sample count")
        save_audio(args.output, payload)


if __name__ == "__main__":
    main()
