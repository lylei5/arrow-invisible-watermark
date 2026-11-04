#!/usr/bin/env python3
"""ARROW 解码服务冒烟测试（只用标准库）。

用法：
    python tools/smoke_test.py --url http://127.0.0.1:8501/decode \
        --image samples/inner/ARROW_13800138000.png --token "$DECODE_TOKEN"

成功条件：HTTP 200 且 accepted=true；带 --expect-digits 时还会核对号码。
"""

import argparse
import hashlib
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
import uuid


def build_multipart(image_path, field="image", extra_fields=None):
    boundary = "----arrow" + uuid.uuid4().hex
    name = os.path.basename(image_path)
    content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
    with open(image_path, "rb") as handle:
        payload = handle.read()

    parts = []
    for key, value in (extra_fields or {}).items():
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
            f"{value}\r\n".encode("utf-8")
        )
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
        f"filename=\"{name}\"\r\nContent-Type: {content_type}\r\n\r\n".encode("utf-8")
    )
    parts.append(payload)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}", payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8501/decode")
    parser.add_argument("--image", required=True)
    parser.add_argument("--token", default=os.environ.get("DECODE_TOKEN", ""))
    parser.add_argument("--carrier", default="inner", choices=["inner", "edge"])
    parser.add_argument("--expect-digits", default="13800138000")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    body, content_type, payload = build_multipart(args.image)
    url = args.url
    if "carrier=" not in url:
        url += ("&" if "?" in url else "?") + "carrier=" + args.carrier

    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", content_type)
    request.add_header("X-Image-SHA256", hashlib.sha256(payload).hexdigest())
    if args.token:
        request.add_header("X-Decode-Token", args.token)

    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            status = response.status
            text = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        status = error.code
        text = error.read().decode("utf-8", "replace")
    except Exception as error:  # noqa: BLE001 - 报告网络层失败即可
        print(f"FAIL 请求失败: {error}")
        return 2

    print(f"HTTP {status}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f"FAIL 响应不是 JSON: {text[:200]}")
        return 2
    print(json.dumps(data, ensure_ascii=False, indent=2)[:1200])

    if status != 200:
        print("FAIL 非 200")
        return 1
    if data.get("accepted") is not True:
        print(f"FAIL 未被接受: {data.get('reason_text') or data.get('reason')}")
        return 1
    if args.expect_digits and str(data.get("digits")) != args.expect_digits:
        print(f"FAIL 号码不符: {data.get('digits')} != {args.expect_digits}")
        return 1
    print("OK accepted=true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
