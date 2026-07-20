#!/usr/bin/env python3
"""Render a local HTML/SVG blueprint preview to PNG with Chrome or Edge.

Use this when CairoSVG is unavailable, which is common on Windows because
native Cairo libraries are not always installed. The script has no Python
package dependencies; it shells out to an installed browser in headless mode.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys


def browser_candidates() -> list[str]:
    names = ["chrome", "google-chrome", "chromium", "chromium-browser", "msedge", "microsoft-edge"]
    found = [path for name in names if (path := shutil.which(name))]
    windows_roots = [
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMFILES(X86)"),
        os.environ.get("LOCALAPPDATA"),
    ]
    suffixes = [
        r"Google\Chrome\Application\chrome.exe",
        r"Microsoft\Edge\Application\msedge.exe",
    ]
    for root in windows_roots:
        if not root:
            continue
        for suffix in suffixes:
            candidate = pathlib.Path(root) / suffix
            if candidate.exists():
                found.append(str(candidate))
    deduped: list[str] = []
    for item in found:
        if item not in deduped:
            deduped.append(item)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("html", help="local diagram.html path")
    parser.add_argument("png", help="output diagram.png path")
    parser.add_argument("--width", type=int, default=1800)
    parser.add_argument("--height", type=int, default=1400)
    parser.add_argument("--browser", help="explicit chrome/msedge executable")
    args = parser.parse_args()

    html_path = pathlib.Path(args.html).resolve()
    png_path = pathlib.Path(args.png).resolve()
    if not html_path.exists():
        raise SystemExit(f"HTML not found: {html_path}")

    browser = args.browser
    if not browser:
        candidates = browser_candidates()
        if not candidates:
            raise SystemExit("No Chrome/Chromium/Edge executable found for headless screenshot")
        browser = candidates[0]

    png_path.parent.mkdir(parents=True, exist_ok=True)
    url = html_path.as_uri()
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={args.width},{args.height}",
        f"--screenshot={png_path}",
        url,
    ]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        if "--headless=new" in cmd:
            cmd[1] = "--headless"
            result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    if not png_path.exists() or png_path.stat().st_size == 0:
        raise SystemExit(f"PNG was not written or is empty: {png_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
