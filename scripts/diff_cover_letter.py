"""Compare Editorial Manager's pasted cover-letter text with the source copy."""
from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="paper/submission/cover_letter.md")
    parser.add_argument("--pasted", required=True,
                        help="plain-text file saved from the Editorial Manager text box")
    args = parser.parse_args()
    source_path, pasted_path = Path(args.source), Path(args.pasted)
    source = source_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    pasted = pasted_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if source == pasted:
        print("cover letter paste diff: identical")
        return 0
    diff = difflib.unified_diff(
        source.splitlines(keepends=True), pasted.splitlines(keepends=True),
        fromfile=str(source_path), tofile=str(pasted_path),
    )
    sys.stdout.writelines(diff)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
