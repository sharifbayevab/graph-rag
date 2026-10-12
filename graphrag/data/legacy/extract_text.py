# graph/extract_text.py
"""
Unified text extraction for the corpus.

  .docx -> docx2txt
  .doc  -> macOS `textutil -convert txt` (native, reads Cyrillic; no install)
  .pdf  -> pypdf if available, else marked unparsed
  .md/.txt -> read directly

Also splits the text into a "header block" (used for the document's OWN metadata)
vs the body, and detects when a single .doc holds multiple concatenated documents
(e.g. a committee QARORI followed by its XULOSA).
"""
import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import docx2txt


# Body-start markers: self-metadata must be extracted from text BEFORE these,
# because the body cites many other (repealed/amended) acts with their own dates.
_BODY_START = re.compile(
    r"(?:^|\n)\s*(?:"
    r"1\s*-\s*модда|1\s*-\s*modda|"
    r"1\s*-\s*боб|1\s*-\s*bob|"
    r"умумий\s+қоидалар|umumiy\s+qoidalar"
    r")",
    re.IGNORECASE,
)

# Strong document-type headers. Matched ONLY against a standalone line (after
# collapsing inter-letter spacing), so the body word "қарор қилади" or a cited
# "...қарорига" never triggers a false type. A genuine header is its own line:
# "Қ А Р О Р И" / "X U L O S A" / "ҚОНУНИ".
_HEADER_KEYWORDS = [
    ("resolution", re.compile(r"^(ҚАРОР|ҚАРОРИ|QAROR|QARORI)$", re.IGNORECASE)),
    ("committee_conclusion", re.compile(r"^(ХУЛОСА|XULOSA)$", re.IGNORECASE)),
    ("postanovlenie", re.compile(r"^(ПОСТАНОВЛЕНИЕ)$", re.IGNORECASE)),
    ("law", re.compile(r"^(ҚОНУНИ|QONUNI|ҚОНУН|QONUN)$", re.IGNORECASE)),
    ("stenogram", re.compile(r"^(СТЕНОГРАФИКҲИСОБОТ|STENOGRAFIKHISOBOT|СТЕНОГРАММА|СТЕНОГРАФИКОТЧЕТ)$", re.IGNORECASE)),
]


def _doc_to_text(path: Path) -> str:
    """
    Extract text from a legacy binary .doc. Cross-platform: macOS `textutil`,
    else LibreOffice headless (robust for Cyrillic), else catdoc / antiword.
    Whichever backend is installed is used; install one on the host (see README).
    """
    # macOS native
    if shutil.which("textutil"):
        out = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(path)],
                             capture_output=True, text=True, timeout=120)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    # LibreOffice headless (Linux/macOS) — handles Uzbek Cyrillic reliably
    soffice = shutil.which("libreoffice") or shutil.which("soffice")
    if soffice:
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([soffice, "--headless", "--convert-to", "txt:Text",
                            "--outdir", td, str(path)],
                           capture_output=True, timeout=180)
            outp = os.path.join(td, path.stem + ".txt")
            if os.path.exists(outp):
                return Path(outp).read_text(encoding="utf-8", errors="replace")
    # catdoc (good Cyrillic via -d utf-8)
    if shutil.which("catdoc"):
        out = subprocess.run(["catdoc", "-d", "utf-8", str(path)],
                             capture_output=True, text=True, timeout=120)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    # antiword (UTF-8 mapping)
    if shutil.which("antiword"):
        out = subprocess.run(["antiword", "-m", "UTF-8.txt", str(path)],
                             capture_output=True, text=True, timeout=120)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout
    return ""


def file_text(path: Path) -> str:
    """Extract raw text from a single file, by extension."""
    ext = path.suffix.lower()
    if ext == ".docx":
        return docx2txt.process(str(path)) or ""
    if ext == ".doc":
        return _doc_to_text(path)
    if ext in (".md", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)
        except Exception:
            return ""
    return ""


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def header_block(text: str, max_chars: int = 1600, max_lines: int = 18) -> str:
    """Return the leading block before the body starts (for self-metadata)."""
    m = _BODY_START.search(text)
    head = text[: m.start()] if m else text
    head = head[:max_chars]
    # keep only non-blank lines before applying max_lines: some documents start with a
    # stray page-number and dozens of blank lines, which would otherwise push the real
    # title/date block past max_lines and yield a useless header (e.g. just "10").
    lines = [ln for ln in head.splitlines() if ln.strip()]
    return "\n".join(lines[:max_lines]).strip() or head.strip()


def find_type_headers(text: str) -> List[tuple]:
    """
    Return [(doc_type, char_position)] for standalone type-header lines, in order.
    A line qualifies only if, after removing all whitespace, it equals a keyword
    (optionally followed by ':'). This avoids matching body words.
    """
    hits = []
    pos = 0
    for line in text.splitlines():
        despaced = re.sub(r"\s+", "", line).rstrip(":").rstrip(".")
        if 3 <= len(despaced) <= 24:
            for dtype, rx in _HEADER_KEYWORDS:
                if rx.match(despaced):
                    hits.append((dtype, pos))
                    break
        pos += len(line) + 1
    return hits


def looks_multi_document(text: str) -> bool:
    """
    Heuristic: a single file holds >1 logical document when two DIFFERENT strong
    type-headers appear far apart (e.g. a committee QARORI then a XULOSA).
    """
    hits = find_type_headers(text)
    seen = []
    for dtype, pos in hits:
        if not seen or (dtype != seen[-1][0] and pos - seen[-1][1] > 800):
            seen.append((dtype, pos))
    return len({d for d, _ in seen}) >= 2 and len(seen) >= 2
