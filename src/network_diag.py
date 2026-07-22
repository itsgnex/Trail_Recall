import socket
import subprocess
import urllib.request

from .config import (
    DEFAULT_ANDROID_TRAIL_URL,
    DEFAULT_MENTRA_HLS_URL,
    DEFAULT_MENTRA_RTMP_URL,
    DEFAULT_MENTRA_RTSP_URL,
    EXPECTED_MAC_IP,
    EXPECTED_PHONE_IP,
)


def port_open(host, port, timeout=0.5):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, int(port))) == 0


def local_ip():
    for iface in ("en0", "en1"):
        try:
            value = subprocess.check_output(["ipconfig", "getifaddr", iface], text=True, timeout=2).strip()
            if value:
                return value
        except Exception:
            pass
    return ""


def http_get(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            return True, f"HTTP {response.status} {body}".strip()
    except Exception as exc:
        return False, str(exc)


def stream_probe(url, timeout=5):
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        url,
        "-an",
        "-sn",
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        return subprocess.run(cmd, capture_output=True, timeout=timeout).returncode == 0
    except Exception:
        return False


def print_network_check(android_url=None):
    android_url = (android_url or DEFAULT_ANDROID_TRAIL_URL).rstrip("/")
    mac_ip = local_ip()
    print("NETWORK_CHECK")
    print(f"macIp={mac_ip or 'UNKNOWN'}")
    print(f"expectedMacIp={EXPECTED_MAC_IP}")
    print(f"phoneIp={EXPECTED_PHONE_IP}")
    if mac_ip and mac_ip != EXPECTED_MAC_IP:
        print(f"warning=Mac IP changed; update MENTRA_STREAM_HOST_IP or docs to {mac_ip}")
    ok, detail = http_get(f"{android_url}/health")
    print(f"androidHealth={ok} detail={detail}")
    print(f"port8765InUse={port_open('127.0.0.1', 8765)}")
    print(f"port8767InUse={port_open('127.0.0.1', 8767)}")
    for port in (1935, 8554, 8888, 8889):
        print(f"mediaMtxPort{port}={port_open('127.0.0.1', port)}")
    print(f"rtmpPublisher={stream_probe(DEFAULT_MENTRA_RTMP_URL)}")
    print(f"rtspAvailable={stream_probe(DEFAULT_MENTRA_RTSP_URL)}")
    hls_ok, hls_detail = http_get(DEFAULT_MENTRA_HLS_URL)
    print(f"hlsAvailable={hls_ok} detail={hls_detail[:160]}")
