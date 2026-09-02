"""校验 Hugging Face Space 的 README 配置头，避免推送时被服务端打回。"""

import re
import sys

LIMITS = {"short_description": 60, "title": 100}


def main(path: str) -> int:
    text = open(path, encoding="utf-8").read()
    block = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if not block:
        print("README.md 缺少 YAML 配置头", file=sys.stderr)
        return 1
    fields = {}
    for line in block.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    if fields.get("sdk") != "docker":
        print(f"sdk 应为 docker，当前是 {fields.get('sdk')!r}", file=sys.stderr)
        return 1
    for key, cap in LIMITS.items():
        value = fields.get(key)
        if value and len(value) > cap:
            print(f"{key} 有 {len(value)} 字符，超过上限 {cap}", file=sys.stderr)
            return 1
    print("  README 配置头检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
