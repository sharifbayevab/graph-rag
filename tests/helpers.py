"""Test helpers: minimal .docx builder, fake token counter, fake encoder."""
import hashlib
import zipfile
from xml.sax.saxutils import escape

import numpy as np

_NS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'


def R(text: str, sup: bool = False) -> str:
    rpr = '<w:rPr><w:vertAlign w:val="superscript"/></w:rPr>' if sup else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'


def P(*runs: str, num: int | None = None, ilvl: int = 0) -> str:
    ppr = (f'<w:pPr><w:numPr><w:ilvl w:val="{ilvl}"/><w:numId w:val="{num}"/></w:numPr></w:pPr>'
           if num is not None else "")
    body = "".join(r if r.startswith("<w:") else R(r) for r in runs)
    return f"<w:p>{ppr}{body}</w:p>"


def make_docx(path, body_xml: str, numbering_xml: str | None = None, header_xml: str | None = None):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("word/document.xml",
                   f'<?xml version="1.0" encoding="UTF-8"?><w:document {_NS}><w:body>{body_xml}</w:body></w:document>')
        if numbering_xml:
            z.writestr("word/numbering.xml",
                       f'<?xml version="1.0" encoding="UTF-8"?><w:numbering {_NS}>{numbering_xml}</w:numbering>')
        if header_xml:
            z.writestr("word/header1.xml",
                       f'<?xml version="1.0" encoding="UTF-8"?><w:hdr {_NS}>{header_xml}</w:hdr>')
    return path


def words(text: str) -> int:
    """Fake token counter: one token per whitespace-separated word."""
    return len(text.split())


class FakeEncoder:
    """Deterministic hash-based unit vectors; same text → same vector."""
    model_name = "fake"
    slug = "fake"
    dim = 8

    def _vec(self, t: str) -> np.ndarray:
        h = hashlib.sha256(t.encode()).digest()
        v = np.frombuffer(h[: self.dim * 4], dtype=np.uint32).astype(np.float32) - 2**31
        return v / np.linalg.norm(v)

    def encode_passages(self, texts, **kw):          # accepts strict= / progress= like Encoder
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def encode_queries(self, texts, **kw):
        return self.encode_passages(texts)
