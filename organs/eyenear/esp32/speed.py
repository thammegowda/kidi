import argparse
import http.client
import json
import sys
import time

import wifi

MAX_BYTES = 1024 * 1024


def pattern(offset, size):
    values = bytes(range(256))
    start = offset % 256
    return (values * ((start + size + 255) // 256))[start:start + size]


def check_status(response):
    if response.status in (401, 403):
        raise wifi.SecurityError("Speed-test owner authentication failed")
    if response.status != 200:
        raise wifi.SetupError(f"Speed test failed with HTTP {response.status}")


def transfer(profile, direction, size, block_size):
    if direction not in ("download", "upload") or not 4096 <= size <= MAX_BYTES or not 256 <= block_size <= 16384:
        raise ValueError("Invalid speed-test direction, byte count, or block size")
    connection = wifi.DeviceConnection(profile)
    try:
        connect_start = time.monotonic()
        connection.connect()
        setup_seconds = time.monotonic() - connect_start
        path = f"/v1/speed?bytes={size}&block_size={block_size}"
        headers = {"Authorization": f"Bearer {profile['token']}"}
        start = time.monotonic()
        if direction == "download":
            connection.request("GET", path, headers=headers)
            response = connection.getresponse()
            check_status(response)
            if response.getheader("X-Kidi-Speed-Protocol") != "1":
                raise wifi.SetupError("Unsupported speed-test response")
            received = 0
            while received < size:
                if time.monotonic() - start > 65:
                    raise TimeoutError(f"Speed test exceeded 65 seconds after {received} bytes")
                data = response.read(min(16384, size - received))
                if not data or data != pattern(received, len(data)):
                    raise wifi.SetupError(f"Download integrity/length failed after {received} bytes")
                received += len(data)
            if response.read(1):
                raise wifi.SetupError("Speed test returned unexpected extra bytes")
            elapsed = time.monotonic() - start
            signal = int(response.getheader("X-Kidi-RSSI"))
            device_seconds = None
        else:
            headers.update({"Content-Type": "application/octet-stream", "Content-Length": str(size)})
            connection.putrequest("POST", path)
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders()
            response = None
            try:
                for offset in range(0, size, block_size):
                    if time.monotonic() - start > 65:
                        raise TimeoutError("Upload speed test exceeded 65 seconds")
                    connection.send(pattern(offset, min(block_size, size - offset)))
            except (BrokenPipeError, ConnectionResetError) as send_error:
                try:
                    response = connection.getresponse()
                except (OSError, http.client.HTTPException):
                    raise send_error
            if response is None:
                response = connection.getresponse()
            check_status(response)
            data = response.read(4097)
            if len(data) > 4096:
                raise wifi.SetupError("Upload response exceeds its bound")
            result = json.loads(data)
            if result["protocol"] != 1 or result["bytes"] != size or result["elapsed_us"] <= 0:
                raise wifi.SetupError("Upload acknowledgement does not match the verified transfer")
            elapsed = time.monotonic() - start
            signal = result["rssi_dbm"]
            device_seconds = result["elapsed_us"] / 1e6
        return {
            "direction": direction, "bytes": size, "block_size": block_size,
            "tls_setup_seconds": setup_seconds, "transfer_seconds": elapsed,
            "mbit_per_second": size * 8 / elapsed / 1e6,
            "kib_per_second": size / elapsed / 1024,
            "rssi_dbm": signal, "device_receive_seconds": device_seconds,
            "integrity_verified": True,
        }
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=262144)
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--direction", choices=("both", "download", "upload"), default="both")
    parser.add_argument("--device")
    args = parser.parse_args()
    if not 4096 <= args.bytes <= MAX_BYTES or not 256 <= args.block_size <= 16384:
        parser.error("--bytes must be 4096..1048576; --block-size must be 256..16384")
    profile = wifi.cached_profile(args.device)
    print("Authenticated HTTPS payload test; no camera/microphone activity, no USB fallback.", flush=True)
    directions = ("download", "upload") if args.direction == "both" else (args.direction,)
    for direction in directions:
        print(f"Testing {direction}: {args.bytes} bytes, block size {args.block_size}...", flush=True)
        result = transfer(profile, direction, args.bytes, args.block_size)
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except (wifi.SetupError, OSError, ValueError, KeyError, http.client.HTTPException) as error:
        print(f"Speed test failed: {error}", file=sys.stderr)
        sys.exit(1)
