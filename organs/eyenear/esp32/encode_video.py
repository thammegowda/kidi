import argparse
import json
import re
from pathlib import Path
import subprocess
import wave

import imageio_ffmpeg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    manifest = json.loads((args.frames / "capture.json").read_text())
    frames = manifest["frames"]
    end = manifest.get("requested_duration_us", manifest["duration_us"]) / 1000000
    first = frames[0]["timestamp_us"] / 1000000
    concat = []
    for index, frame in enumerate(frames):
        path = (args.frames / frame["filename"]).resolve()
        if not path.is_file() or "'" in str(path):
            raise ValueError(f"Invalid frame path: {path}")
        current = frame["timestamp_us"] / 1000000
        following = frames[index + 1]["timestamp_us"] / 1000000 if index + 1 < len(frames) else end
        duration = following - current
        if index == 0:
            duration += first
        if duration <= 0:
            raise ValueError("Non-increasing frame timestamps")
        concat.extend([f"file '{path}'", f"duration {duration:.6f}"])
    concat.append(f"file '{(args.frames / frames[-1]['filename']).resolve()}'")
    timing = args.frames / "timing.ffconcat"
    timing.write_text("\n".join(concat) + "\n")
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        executable, "-hide_banner", "-loglevel", "warning", "-n",
        "-f", "concat", "-safe", "0", "-i", str(timing),
    ]
    if "audio" in manifest:
        audio_path = args.frames / manifest["audio"]["filename"]
        with wave.open(str(audio_path), "rb") as audio:
            if (
                audio.getnframes() != manifest["audio"]["samples"]
                or audio.getframerate() <= 0
                or audio.getnchannels() != 1
                or audio.getsampwidth() != 2
            ):
                raise ValueError("Unexpected concurrent audio WAV format")
            if "sample_rate" in manifest["audio"] and audio.getframerate() != manifest["audio"]["sample_rate"]:
                raise ValueError("WAV sample rate does not match the capture metadata")
            # AAC-LC permits at most 6144 bits per 1024-sample mono frame.
            audio_bitrate = min(128000, audio.getframerate() * 6)
        command.extend(["-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0"])
    command.extend([
        "-t", f"{end:.6f}", "-vf", "fps=10", "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
    ])
    if "audio" in manifest:
        analysis = subprocess.run(
            [
                executable, "-hide_banner", "-i", str(audio_path),
                "-af", "highpass=f=80,volumedetect", "-f", "null", "-",
            ],
            capture_output=True, text=True, check=True,
        )
        peak = re.search(r"max_volume:\s*(-?\d+(?:\.\d+)?) dB", analysis.stderr)
        if peak is None:
            raise RuntimeError("Unable to measure filtered audio peak level")
        gain = min(16.0, 10 ** ((-1.0 - float(peak.group(1))) / 20))
        command.extend([
            "-af", f"highpass=f=80,volume={gain:.6f}",
            "-c:a", "aac", "-b:a", str(audio_bitrate),
        ])
        print(f"Playback audio: 80 Hz high-pass, {gain:.2f}x gain; raw WAV preserved")
    command.extend(["-movflags", "+faststart", str(args.output)])
    subprocess.run(command, check=True)
    validation = subprocess.run(
        [
            executable, "-hide_banner", "-v", "error", "-i", str(args.output),
            "-progress", "pipe:1", "-nostats", "-f", "null", "-",
        ],
        capture_output=True, text=True, check=True,
    )
    progress = dict(line.split("=", 1) for line in validation.stdout.splitlines() if "=" in line)
    if int(progress["frame"]) != round(end * 10):
        raise RuntimeError("Encoded video frame count does not match requested duration")
    print(f"Full decode passed; video output frames={progress['frame']}")
    inspection = subprocess.run(
        [executable, "-hide_banner", "-i", str(args.output)],
        capture_output=True, text=True,
    )
    if inspection.returncode != 1 or "Video: h264" not in inspection.stderr:
        raise RuntimeError(f"Unexpected video inspection result: {inspection.stderr}")
    if "audio" in manifest and "Audio: aac" not in inspection.stderr:
        raise RuntimeError("Encoded video is missing the microphone audio stream")
    print(inspection.stderr)
    print(f"Encoded and decoded successfully: {args.output}")


if __name__ == "__main__":
    main()
