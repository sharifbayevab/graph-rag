"""Document record contract (spec §2.1). One JSON file per source document."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class Node:
    node_id: str
    type: str            # документ | глава | статья | часть | пункт | подпункт | преамбула |
                         # реквизиты | заголовок | блок | вопрос | выступление
    title: str
    number: str | None   # canonical number ("21", "122.1", "a")
    parent_id: str | None
    start: int           # char span [start, end) in DocRecord.text
    end: int


@dataclass
class Chunk:
    chunk_id: str        # "<parent>#<par>"
    parent: str          # μ(c): deepest node whose span contains the chunk
    par: int             # 1-based ordinal of the chunk inside its chunking unit
    tokens: int
    src: str             # paragraph range inside the document, "абз. 3-5"
    text: str            # exact substring DocRecord.text[start:end]
    text_norm: str       # internal representation (Latin, ISO dates, canonical numbers)
    start: int
    end: int
    # Contextual text for indexing/embedding: to_norm(breadcrumb) + "\n" + text_norm (not a
    # substring). Last with a default so positional constructors (…, text_norm, start, end) work.
    index_text: str = ""


@dataclass
class Meta:
    type: str
    date: str | None
    organ: str
    lang: list[str]
    status: str
    section: str
    title: str | None


@dataclass
class Source:
    store: str
    file: str
    pages: str | None
    aliases: list[str] = field(default_factory=list)


@dataclass
class DocRecord:
    doc_id: str
    meta: Meta
    source: Source
    text: str
    structure: list[Node]
    chunks: list[Chunk]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, indent=1) + "\n"

    @classmethod
    def from_json(cls, s: str) -> "DocRecord":
        d = json.loads(s)
        return cls(
            doc_id=d["doc_id"],
            meta=Meta(**d["meta"]),
            source=Source(**d["source"]),
            text=d["text"],
            structure=[Node(**n) for n in d["structure"]],
            chunks=[Chunk(**c) for c in d["chunks"]],
        )
