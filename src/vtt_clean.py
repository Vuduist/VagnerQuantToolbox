"""VTT -> чистый текст для конспектов. Убирает таймкоды, теги, дубли."""

import re
import sys
from pathlib import Path


def vtt_to_text(path: str) -> str:
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    lines = []
    prev = ""
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        if "-->" in line:
            continue
        line = re.sub(r"<[^>]+>", "", line)          # теги <c>, <00:..>
        line = line.replace("&amp;", "&").strip()
        if not line or line == prev:
            continue
        # склейка "развёрнутых" дублей авто-субтитров
        if prev and line.startswith(prev):
            lines[-1] = line
        elif prev and prev.endswith(line):
            pass
        else:
            lines.append(line)
        prev = line
    text = " ".join(lines)
    text = re.sub(r"\s+", " ", text)
    return text


if __name__ == "__main__":
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.vtt")):
        if "-orig" in f.stem and (src / f.name.replace("-orig", "")).exists():
            continue
        txt = vtt_to_text(str(f))
        vid = f.stem.split(".")[0]
        (out / f"{vid}.txt").write_text(txt, encoding="utf-8")
        print(f"{vid}: {len(txt)} символов")
