import argparse
import ctypes
from datetime import datetime, timedelta, timezone
import getpass
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import secrets
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID
import serial

import capture

ROOT = Path(__file__).resolve().parent
PRIVATE = ROOT / ".private"
WIFI_CACHE = ROOT / ".kidi.wifi.txt"
MAX_PHOTO_BYTES = 1024 * 1024
RESOLUTIONS = {"2048x1536": (2048, 1536), "640x480": (640, 480)}
CLIP_COMMANDS = ("audio",)
MAX_CLIP_BYTES = 6 * 1024 * 1024


class WifiCaptureStream:
    def __init__(self, response, connection, seconds):
        self.response = response
        self.connection = connection
        self.deadline = time.monotonic() + seconds + 30
        self.received = 0

    def prepare_read(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Wi-Fi clip exceeded its bounded transfer time")
        if self.connection.sock is not None:
            self.connection.sock.settimeout(min(25, remaining))

    def account(self, data):
        self.received += len(data)
        if self.received > MAX_CLIP_BYTES:
            raise SetupError("Wi-Fi clip exceeds the transfer-size limit")
        if not data:
            raise SetupError("Wi-Fi capture ended before its completion marker")
        return data

    def read(self, length):
        self.prepare_read()
        try:
            return self.account(self.response.read(length))
        except TimeoutError as error:
            raise TimeoutError(f"Wi-Fi clip transfer timed out after {self.received} bytes") from error

    def readline(self):
        self.prepare_read()
        try:
            line = self.account(self.response.readline(4097))
        except TimeoutError as error:
            raise TimeoutError(f"Wi-Fi clip control transfer timed out after {self.received} bytes") from error
        if len(line) > 4096 or not line.endswith(b"\n"):
            raise SetupError("Invalid Wi-Fi control-line framing")
        return line

    def finish(self):
        self.prepare_read()
        if self.response.read(1):
            raise SetupError("Unexpected data after the Wi-Fi capture completion marker")


class SetupError(RuntimeError):
    pass


class CredentialUnavailable(SetupError):
    pass


class CredentialCancelled(SetupError):
    pass


class WifiUnavailable(SetupError):
    pass


class SecurityError(SetupError):
    pass


def private_directory(path):
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise SecurityError("Private configuration directory must not be a symlink")
    if os.name != "nt":
        path.chmod(0o700)
    else:
        user = subprocess.run(["whoami"], capture_output=True, text=True, check=True).stdout.strip()
        result = subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r", f"{user}:(OI)(CI)F"],
            capture_output=True,
        )
        if result.returncode:
            raise SecurityError("Could not restrict the private directory's Windows permissions")


def save_private(path, data, protect_directory=True):
    if protect_directory:
        private_directory(path.parent)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists() or path.is_symlink():
        raise SecurityError("Unsafe private configuration path")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        if os.name == "nt":
            user = subprocess.run(["whoami"], capture_output=True, text=True, check=True).stdout.strip()
            permissions = subprocess.run(
                ["icacls", str(temporary), "/inheritance:r", "/grant:r", f"{user}:F"],
                capture_output=True,
            )
            if permissions.returncode:
                raise SecurityError("Could not restrict the private file's Windows permissions")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_wifi_cache():
    if WIFI_CACHE.is_symlink():
        raise SecurityError("Wi-Fi cache must not be a symlink")
    if not WIFI_CACHE.exists():
        return {}
    if os.name != "nt" and (
        WIFI_CACHE.stat().st_mode & 0o077 or WIFI_CACHE.stat().st_uid != os.getuid()
    ):
        raise SecurityError("Wi-Fi cache must belong to this account and have owner-only permissions")
    if WIFI_CACHE.stat().st_size > 65536:
        raise SecurityError("Wi-Fi cache exceeds its size limit")
    try:
        data = json.loads(WIFI_CACHE.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise SecurityError("Wi-Fi cache is invalid; remove it before requesting credentials again") from error
    if (
        not isinstance(data, dict) or data.get("schema_version") != 1
        or not isinstance(data.get("networks"), dict)
        or not all(
            isinstance(ssid, str) and 1 <= len(ssid.encode()) <= 32 and "\0" not in ssid
            and isinstance(password, str) and "\0" not in password
            and (not password or 8 <= len(password.encode()) <= 64)
            for ssid, password in data["networks"].items()
        )
    ):
        raise SecurityError("Wi-Fi cache has invalid profile data")
    return data["networks"]


def cache_wifi_credentials(ssid, password):
    if (
        not 1 <= len(ssid.encode()) <= 32 or "\0" in ssid or "\0" in password
        or (password and not 8 <= len(password.encode()) <= 64)
    ):
        raise SetupError("Invalid selected Wi-Fi profile")
    networks = read_wifi_cache()
    networks[ssid] = password
    data = {"schema_version": 1, "networks": networks}
    save_private(WIFI_CACHE, (json.dumps(data, indent=2) + "\n").encode(), protect_directory=False)


def generate_identity():
    device_id = uuid.uuid4().hex
    hostname = f"kidi-{device_id}.local"
    token = secrets.token_hex(32)
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
    now = datetime.now(timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name).public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(hostname)]), critical=False)
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(key.public_key()), critical=False)
        .add_extension(
            x509.KeyUsage(True, False, False, False, False, True, True, False, False), critical=True
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
        .sign(key, hashes.SHA256())
    )
    return {
        "device_id": device_id, "hostname": hostname, "token": token,
        "certificate": certificate.public_bytes(serialization.Encoding.PEM).decode(),
    }, key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def profile_path(device_id):
    if not re.fullmatch(r"[0-9a-f]{32}", device_id):
        raise SecurityError("Invalid device identity")
    return PRIVATE / f"{device_id}.json"


def load_profile(device_id):
    path = profile_path(device_id)
    if not path.exists():
        raise SecurityError("This device is already owned, but this laptop has no owner credential")
    if path.is_symlink():
        raise SecurityError("Private credential file must not be a symlink")
    if os.name != "nt" and path.stat().st_mode & 0o077:
        raise SecurityError("Private credential file permissions must be owner-only")
    profile = json.loads(path.read_text())
    if profile["device_id"] != device_id or not re.fullmatch(r"[0-9a-f]{64}", profile["token"]):
        raise SecurityError("Invalid saved owner credential")
    expected = f"kidi-{device_id}.local"
    if profile["hostname"] != expected:
        raise SecurityError("Saved identity hostname mismatch")
    return profile


def cached_profile(device_id=None):
    if device_id:
        profile = load_profile(device_id)
        if "address" not in profile:
            raise SetupError("Accessory setup is incomplete; reconnect USB and retry")
        return profile
    profiles = [
        load_profile(path.stem) for path in PRIVATE.glob("*.json")
        if re.fullmatch(r"[0-9a-f]{32}", path.stem)
    ]
    profiles = [profile for profile in profiles if "address" in profile]
    if len(profiles) != 1:
        raise SetupError("Connect USB for setup or select one saved accessory with --device")
    return profiles[0]


def save_profile(profile):
    save_private(profile_path(profile["device_id"]), (json.dumps(profile, indent=2) + "\n").encode())


def choose_port(requested):
    return capture.choose_port(requested)


def open_serial(port):
    device = serial.Serial(port=None, baudrate=115200, timeout=0.5, write_timeout=5)
    device.dtr = True
    device.rts = False
    device.port = port
    device.open()
    time.sleep(0.5)
    return device


def usb_request(device, command, timeout=35):
    device.write((command + "\n").encode())
    device.flush()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = device.readline()
        if line.startswith(b"KIDI_WIFI_REPLY "):
            response = json.loads(line[len(b"KIDI_WIFI_REPLY "):])
            if not response.get("ok"):
                error = response.get("error", "USB setup failed")
                if response.get("code") == "WIFI_UNAVAILABLE":
                    raise WifiUnavailable(error)
                if response.get("code") == "OWNER_REQUIRED":
                    raise SecurityError(error)
                raise SetupError(error)
            return response
        if line.startswith(b"KIDI_ERROR unknown command"):
            raise WifiUnavailable("Installed firmware has no Wi-Fi setup interface; run make flash")
    raise WifiUnavailable("USB Wi-Fi setup response timed out")


def macos_helper():
    private_directory(PRIVATE)
    bundle = PRIVATE / "KidiWifiSetup.app"
    executable = bundle / "Contents/MacOS/kidi-wifi"
    source = ROOT / "tools/macos_wifi.swift"
    plist = ROOT / "tools/macos_wifi.plist"
    if not executable.exists() or executable.stat().st_mtime < max(source.stat().st_mtime, plist.stat().st_mtime):
        executable.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(plist, bundle / "Contents/Info.plist")
        subprocess.run(
            ["swiftc", "-framework", "AppKit", "-framework", "CoreLocation", "-framework", "CoreWLAN",
             "-framework", "Security", "-framework", "LocalAuthentication",
             str(source), "-o", str(executable)], check=True,
        )
        subprocess.run(["codesign", "--force", "--sign", "-", str(bundle)], capture_output=True, check=True)
    return bundle, executable


def macos_ssid():
    ports = subprocess.run(
        ["networksetup", "-listallhardwareports"], capture_output=True, text=True, check=True
    ).stdout
    match = re.search(r"Hardware Port: Wi-Fi\s+Device: (\S+)", ports)
    if not match:
        raise CredentialUnavailable("No macOS Wi-Fi interface was found")
    result = subprocess.run(
        ["networksetup", "-getairportnetwork", match.group(1)], capture_output=True, text=True, check=True
    ).stdout.strip()
    if result.startswith("Current Wi-Fi Network: "):
        ssid = result.split(": ", 1)[1]
        if ssid not in ("<redacted>", ""):
            return ssid
    bundle, _ = macos_helper()
    response = PRIVATE / f"network-{uuid.uuid4().hex}.json"
    try:
        print("Reading the current Wi-Fi name through macOS.", flush=True)
        subprocess.run(["open", "-W", "-n", str(bundle), "--args", str(response)], check=True)
        if not response.exists():
            raise CredentialUnavailable("macOS Wi-Fi permission helper did not return a result")
        data = json.loads(response.read_text())
        if "error" in data:
            raise CredentialUnavailable(data["error"])
        return data["ssid"]
    finally:
        response.unlink(missing_ok=True)


def macos_password(ssid):
    _, executable = macos_helper()
    result = subprocess.run(
        [str(executable), "credential"],
        input=json.dumps({"ssid": ssid}).encode(), capture_output=True,
    )
    if not result.stdout:
        raise CredentialUnavailable("macOS did not authorize access to this Wi-Fi credential")
    data = json.loads(result.stdout)
    if data.get("code") in ("CANCELLED", "DENIED"):
        raise CredentialCancelled(data["error"])
    if result.returncode or "error" in data:
        raise CredentialUnavailable(data.get("error", "Selected Wi-Fi credential access failed"))
    return data["password"]


def windows_profile(include_password=True):
    from ctypes import wintypes

    class Guid(ctypes.Structure):
        _fields_ = [("data1", wintypes.DWORD), ("data2", wintypes.WORD),
                    ("data3", wintypes.WORD), ("data4", ctypes.c_ubyte * 8)]

    class Interface(ctypes.Structure):
        _fields_ = [("guid", Guid), ("description", wintypes.WCHAR * 256), ("state", ctypes.c_int)]

    class Ssid(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("data", ctypes.c_ubyte * 32)]

    class Association(ctypes.Structure):
        _fields_ = [("ssid", Ssid), ("bss_type", ctypes.c_int), ("bssid", ctypes.c_ubyte * 6),
                    ("phy_type", ctypes.c_int), ("phy_index", wintypes.DWORD),
                    ("quality", wintypes.DWORD), ("rx_rate", wintypes.DWORD), ("tx_rate", wintypes.DWORD)]

    class Security(ctypes.Structure):
        _fields_ = [("enabled", wintypes.BOOL), ("one_x", wintypes.BOOL),
                    ("authentication", ctypes.c_int), ("cipher", ctypes.c_int)]

    class Connection(ctypes.Structure):
        _fields_ = [("state", ctypes.c_int), ("mode", ctypes.c_int),
                    ("profile", wintypes.WCHAR * 256), ("association", Association), ("security", Security)]

    wlan = ctypes.WinDLL("wlanapi")
    wlan.WlanOpenHandle.argtypes = [wintypes.DWORD, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD),
                                   ctypes.POINTER(wintypes.HANDLE)]
    wlan.WlanEnumInterfaces.argtypes = [wintypes.HANDLE, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    wlan.WlanQueryInterface.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(Guid), ctypes.c_int, ctypes.c_void_p,
        ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.DWORD),
    ]
    wlan.WlanGetProfile.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(Guid), wintypes.LPCWSTR, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD),
    ]
    wlan.WlanFreeMemory.argtypes = [ctypes.c_void_p]
    wlan.WlanCloseHandle.argtypes = [wintypes.HANDLE, ctypes.c_void_p]
    handle = wintypes.HANDLE()
    negotiated = wintypes.DWORD()
    if wlan.WlanOpenHandle(2, None, ctypes.byref(negotiated), ctypes.byref(handle)):
        raise CredentialUnavailable("Windows Wi-Fi access failed")
    interfaces = ctypes.c_void_p()
    try:
        if wlan.WlanEnumInterfaces(handle, None, ctypes.byref(interfaces)):
            raise CredentialUnavailable("Windows Wi-Fi interface discovery failed")
        count = ctypes.cast(interfaces, ctypes.POINTER(wintypes.DWORD))[0]
        entries = (Interface * count).from_address(interfaces.value + 8)
        active = [entry for entry in entries if entry.state == 1]
        if len(active) != 1:
            raise CredentialUnavailable("Windows must have one active Wi-Fi connection for automatic setup")
        data = ctypes.c_void_p()
        size = wintypes.DWORD()
        opcode_type = wintypes.DWORD()
        if wlan.WlanQueryInterface(handle, ctypes.byref(active[0].guid), 7, None,
                                   ctypes.byref(size), ctypes.byref(data), ctypes.byref(opcode_type)):
            raise CredentialUnavailable("Windows current Wi-Fi query failed")
        try:
            connection = ctypes.cast(data, ctypes.POINTER(Connection)).contents
            ssid = bytes(connection.association.ssid.data[:connection.association.ssid.length]).decode("utf-8")
            if not include_password:
                return ssid, None
            if connection.security.one_x:
                raise CredentialUnavailable("Enterprise Wi-Fi credential copying is not supported")
            if not connection.security.enabled:
                return ssid, ""
            xml = ctypes.c_void_p()
            flags = wintypes.DWORD(4)
            access = wintypes.DWORD()
            result = wlan.WlanGetProfile(
                handle, ctypes.byref(active[0].guid), connection.profile, None,
                ctypes.byref(xml), ctypes.byref(flags), ctypes.byref(access),
            )
            if result:
                raise CredentialUnavailable("Windows Wi-Fi profile access was denied")
            try:
                return ssid, password_from_windows_xml(ctypes.wstring_at(xml))
            finally:
                wlan.WlanFreeMemory(xml)
        finally:
            wlan.WlanFreeMemory(data)
    finally:
        if interfaces.value:
            wlan.WlanFreeMemory(interfaces)
        wlan.WlanCloseHandle(handle, None)


def password_from_windows_xml(text):
    root = ET.fromstring(text)
    namespace = {"w": "http://www.microsoft.com/networking/WLAN/profile/v1"}
    key = root.find(".//w:sharedKey", namespace)
    if key is None or key.findtext("w:protected", namespaces=namespace) != "false":
        raise CredentialUnavailable("Windows did not authorize plaintext Wi-Fi key access")
    password = key.findtext("w:keyMaterial", namespaces=namespace)
    if password is None:
        raise CredentialUnavailable("Windows profile contains no usable Wi-Fi key")
    return password


def network_credentials(requested_ssid=None, authorize_credentials=False):
    networks = read_wifi_cache()
    if requested_ssid:
        ssid = requested_ssid
    elif authorize_credentials:
        ssid = current_network_ssid()
    elif len(networks) == 1:
        ssid = next(iter(networks))
    else:
        raise CredentialUnavailable("Select a cached network with WIFI_SSID=...; use make cache-wifi explicitly for a new network")
    if ssid in networks:
        print("Using the private Wi-Fi credential cache for the selected network.", flush=True)
        return ssid, networks[ssid]
    if not authorize_credentials:
        raise CredentialUnavailable("Selected network is not cached; run make cache-wifi explicitly. No credentials were requested.")
    system = platform.system()
    if system == "Darwin":
        try:
            password = macos_password(ssid)
        except CredentialUnavailable:
            if not sys.stdin.isatty():
                raise
            print("Saved credential access was unavailable. Enter the selected network password locally.")
            password = getpass.getpass(f"Wi-Fi password for {ssid!r}: ")
    elif system == "Windows":
        try:
            connected_ssid, password = windows_profile()
            if ssid != connected_ssid:
                raise CredentialUnavailable("Requested Wi-Fi differs from the connected Windows profile")
        except CredentialUnavailable:
            if not sys.stdin.isatty():
                raise
            password = getpass.getpass(f"Wi-Fi password for {ssid!r}: ")
    else:
        raise CredentialUnavailable("Automatic current-network setup supports macOS and native Windows")
    cache_wifi_credentials(ssid, password)
    print("Cached only the selected Wi-Fi credential in private .kidi.wifi.txt.", flush=True)
    return ssid, password


def current_network_ssid():
    system = platform.system()
    if system == "Darwin":
        return macos_ssid()
    if system == "Windows":
        return windows_profile(include_password=False)[0]
    raise CredentialUnavailable("Automatic network discovery supports macOS and native Windows")


def setup_usb(device, requested_ssid=None, check_network=False):
    info = usb_request(device, "wifi-info")
    if info["protocol"] != 1:
        raise SetupError("Unsupported Wi-Fi setup protocol")
    if not check_network and not info["configured"]:
        raise WifiUnavailable("Wi-Fi is not configured; run make dev-connect")
    profile = None
    selected_ssid = requested_ssid
    if check_network and selected_ssid is None and not info["configured"]:
        networks = read_wifi_cache()
        if len(networks) != 1:
            raise CredentialUnavailable("Select a cached Wi-Fi network or run make cache-wifi explicitly; no credentials were requested")
        selected_ssid = next(iter(networks))
    if info["configured"]:
        profile = load_profile(info["device_id"])
        if profile["certificate"] != info["certificate"]:
            raise SecurityError("USB device certificate does not match this laptop's trusted identity")
        same_network = selected_ssid is None or hashlib.sha256(selected_ssid.encode()).hexdigest() == info["ssid_hash"]
        if same_network and check_network and requested_ssid is None and not info["ready"]:
            deadline = time.monotonic() + 20
            print("Waiting for the accessory to rejoin its saved Wi-Fi; no credential request.", flush=True)
            while not info["ready"] and time.monotonic() < deadline:
                time.sleep(1)
                info = usb_request(device, "wifi-info", timeout=3)
                if info["device_id"] != profile["device_id"] or info["certificate"] != profile["certificate"]:
                    raise SecurityError("Accessory identity changed while reconnecting")
            if not info["ready"]:
                raise WifiUnavailable("Saved Wi-Fi did not reconnect; no password was requested")
        if info["ready"] and same_network:
            profile.update(address=info["address"], port=info["port"])
            save_profile(profile)
            return profile
        if not check_network:
            raise WifiUnavailable("Accessory Wi-Fi is not connected; using development fallback")
    ssid, password = network_credentials(selected_ssid)
    if not 1 <= len(ssid.encode()) <= 32 or "\0" in ssid or "\0" in password:
        raise SetupError("Wi-Fi profile cannot be represented by the accessory")
    if password and not 8 <= len(password.encode()) <= 64:
        raise SetupError("Wi-Fi password must be a valid personal-network key")
    print("Configuring the selected Wi-Fi profile over USB; credentials are not logged.", flush=True)
    created = profile is None
    private_key = None
    if created:
        profile, private_key = generate_identity()
        profile["hardware_id"] = info["hardware_id"]
        save_profile(profile)
    request = {"ssid": ssid, "password": password, "token": profile["token"]}
    if created:
        request.update(
            device_id=profile["device_id"], certificate=profile["certificate"], private_key=private_key,
        )
    # Keep the credential if acknowledgement is lost after the device commits.
    reply = usb_request(device, "wifi-configure " + json.dumps(request, separators=(",", ":")))
    if reply["device_id"] != profile["device_id"] or reply["certificate"] != profile["certificate"]:
        raise SecurityError("Device identity changed during setup")
    if not reply["ready"]:
        raise WifiUnavailable("Device joined without a ready secure photo server")
    profile.update(address=reply["address"], port=reply["port"])
    save_profile(profile)
    return profile


class DeviceConnection(http.client.HTTPSConnection):
    def __init__(self, profile):
        ipaddress.ip_address(profile["address"])
        context = ssl.create_default_context(cadata=profile["certificate"])
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        super().__init__(profile["hostname"], port=profile["port"], timeout=25, context=context)
        self.address = profile["address"]

    def connect(self):
        try:
            raw = socket.create_connection((self.address, self.port), timeout=5)
        except OSError as error:
            try:
                raw = socket.create_connection((self.host, self.port), timeout=5)
            except OSError:
                raise WifiUnavailable("Wi-Fi capture server is unreachable; check LAN peer access") from error
        try:
            raw.settimeout(self.timeout)
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
            self.sock.settimeout(self.timeout)
        except ssl.SSLError as error:
            raw.close()
            raise SecurityError("Device TLS verification/handshake failed; refusing USB fallback") from error
        except OSError:
            raw.close()
            raise


def wifi_photo(profile, resolution):
    connection = DeviceConnection(profile)
    try:
        connection.connect()
        connection.request(
            "POST", "/v1/photo", json.dumps({"resolution": resolution}),
            {"Authorization": f"Bearer {profile['token']}", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status in (401, 403):
            raise SecurityError("Device rejected the owner credential; refusing USB fallback")
        if response.status != 200:
            raise SetupError(f"Wi-Fi photo failed with HTTP {response.status}")
        length = int(response.getheader("Content-Length", "0"))
        if not 0 < length <= MAX_PHOTO_BYTES or response.getheader("Content-Type") != "image/jpeg":
            raise SetupError("Invalid Wi-Fi photo response format/size")
        expected = RESOLUTIONS[resolution]
        actual = (int(response.getheader("X-Kidi-Width", "0")), int(response.getheader("X-Kidi-Height", "0")))
        if actual != expected or response.getheader("X-Kidi-Transport") != "wifi":
            raise SetupError("Wi-Fi photo metadata does not match the requested capture")
        payload = response.read(length)
        if len(payload) != length:
            raise SetupError("Incomplete Wi-Fi photo transfer")
        return payload
    finally:
        connection.close()


def wifi_status(profile):
    connection = DeviceConnection(profile)
    try:
        connection.connect()
        connection.request("GET", "/v1/status", headers={"Authorization": f"Bearer {profile['token']}"})
        response = connection.getresponse()
        if response.status in (401, 403):
            raise SecurityError("Wi-Fi endpoint rejected the owner credential")
        if response.status != 200:
            raise SetupError(f"Wi-Fi status failed with HTTP {response.status}")
        payload = response.read(4097)
        if len(payload) > 4096:
            raise SetupError("Wi-Fi status response is too large")
        data = json.loads(payload)
        if data["device_id"] != profile["device_id"] or data["protocol"] != 1 or data["transport"] != "wifi":
            raise SecurityError("Authenticated Wi-Fi endpoint identity mismatch")
        return data
    finally:
        connection.close()


def wifi_clip(profile, kind, seconds, output):
    if kind != "audio":
        raise SetupError("Wi-Fi video/AV use the live preview client, not saved clips")
    connection = DeviceConnection(profile)
    try:
        connection.connect()
        connection.request(
            "POST", f"/v1/{kind}", json.dumps({"seconds": seconds}),
            {"Authorization": f"Bearer {profile['token']}", "Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status in (401, 403):
            raise SecurityError("Device rejected the owner credential; refusing USB fallback")
        if response.status != 200:
            raise SetupError(f"Wi-Fi {kind} capture failed with HTTP {response.status}")
        if (
            response.getheader("Content-Type") != "application/vnd.kidi.capture"
            or response.getheader("X-Kidi-Transport") != "wifi"
            or response.getheader("X-Kidi-Capture-Protocol") != "1"
        ):
            raise SetupError("Unsupported Wi-Fi capture response")
        stream = WifiCaptureStream(response, connection, seconds)
        if kind == "audio":
            media_kind, payload = capture.capture(
                stream, f"audio {seconds}", send_command=False, require_end=True,
            )
            if media_kind != "pcm16" or len(payload) != seconds * capture.SAMPLE_RATE * 2:
                raise SetupError("Unexpected Wi-Fi audio format or sample count")
            capture.save_audio(output, payload)
        else:
            capture.capture_video(
                stream, seconds, output, with_audio=kind == "av",
                send_command=False, require_end=True,
            )
        print(f"Saved WIFI {kind} capture: {output}")
    finally:
        connection.close()


def wifi_capture(profile, args):
    if args.command == "photo":
        save_photo(args.output, wifi_photo(profile, args.resolution), "wifi")
    else:
        wifi_clip(profile, args.command, args.seconds, args.output)


def usb_capture(device, args):
    if args.command in ("video", "av"):
        capture.capture_video(device, args.seconds, args.output, with_audio=args.command == "av")
        print(f"Saved USB {args.command} capture: {args.output}")
        return
    command = f"photo {args.resolution}" if args.command == "photo" else f"audio {args.seconds}"
    kind, payload = capture.capture(
        device, command, resolution=args.resolution if args.command == "photo" else None,
    )
    if args.command == "photo":
        if kind != "jpeg":
            raise SetupError("USB capture returned the wrong media format")
        save_photo(args.output, payload, "usb")
    else:
        if kind != "pcm16" or len(payload) != args.seconds * capture.SAMPLE_RATE * 2:
            raise SetupError("Unexpected USB audio format or sample count")
        capture.save_audio(args.output, payload)
        print(f"Saved USB audio capture: {args.output}")


def save_photo(path, payload, transport):
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise SetupError("Capture is not a complete JPEG")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    print(f"Saved {transport.upper()} photo: {path} ({len(payload)} bytes)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("port", "setup", "cache-network", "forget-wifi-cache", "photo", *CLIP_COMMANDS))
    parser.add_argument("--port")
    parser.add_argument("--device", help="Saved accessory identity for Wi-Fi-only use")
    parser.add_argument("--ssid", help="Explicit network override if OS discovery is unavailable")
    parser.add_argument("--authorize-credentials", action="store_true", help="One-time OS credential access; cache-network only")
    parser.add_argument("--transport", choices=("auto", "wifi", "usb"), default="auto")
    parser.add_argument("--resolution", choices=tuple(RESOLUTIONS), default="2048x1536")
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.authorize_credentials and args.command != "cache-network":
        parser.error("--authorize-credentials is only allowed with cache-network")
    if args.command == "port":
        print(choose_port(args.port))
        return
    if args.command == "cache-network":
        network_credentials(args.ssid, authorize_credentials=args.authorize_credentials)
        return
    if args.command == "forget-wifi-cache":
        if WIFI_CACHE.is_symlink():
            raise SecurityError("Refusing to remove a symlinked Wi-Fi cache")
        WIFI_CACHE.unlink(missing_ok=True)
        print("Removed the laptop Wi-Fi password cache; device identity/configuration was unchanged.")
        return
    if args.command in ("photo", *CLIP_COMMANDS) and args.output is None:
        parser.error("--output is required for captures")
    if not 1 <= args.seconds <= 10:
        parser.error("--seconds must be 1 to 10")
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    if args.command in ("photo", *CLIP_COMMANDS):
        args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command in ("photo", *CLIP_COMMANDS) and args.transport != "usb":
        try:
            profile = cached_profile(args.device)
        except SecurityError:
            raise
        except SetupError:
            if args.transport == "wifi":
                raise
        else:
            try:
                wifi_capture(profile, args)
                return
            except WifiUnavailable:
                if args.transport == "wifi":
                    raise
    device = None
    try:
        try:
            device = open_serial(choose_port(args.port))
        except (ValueError, OSError, serial.SerialException):
            if args.command == "setup" or args.transport == "usb" or args.port not in (None, "", "auto"):
                raise
            profile = cached_profile(args.device)
            wifi_capture(profile, args)
            return
        if args.command == "setup":
            profile = setup_usb(device, args.ssid, check_network=True)
            wifi_status(profile)
            print(f"Secure Wi-Fi photo endpoint ready: {profile['address']}:{profile['port']}")
            return
        if args.transport == "usb":
            usb_capture(device, args)
            return
        try:
            profile = setup_usb(device, args.ssid)
            wifi_capture(profile, args)
        except (WifiUnavailable, CredentialUnavailable) as error:
            if args.transport != "auto":
                raise
            print(f"Wi-Fi unavailable: {error}. Using explicit USB development fallback.", file=sys.stderr)
            usb_capture(device, args)
    finally:
        if device is not None:
            device.close()


if __name__ == "__main__":
    try:
        main()
    except (SetupError, OSError, ValueError, RuntimeError, http.client.HTTPException, subprocess.CalledProcessError) as error:
        print(f"Wi-Fi client error: {error}", file=sys.stderr)
        sys.exit(1)
