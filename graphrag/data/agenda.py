"""Verified LLM agenda segmentation of stenograms (Task 11b).

The regex agenda triggers of structure.parse_stenogram miss unquoted and ASR-style announcements
(the 2025 transcripts), so whole sessions collapse into a few untitled вопрос groups. Here gpt-4.1
reads the numbered CHAIR turns of a stenogram (head, and tail of long turns) and returns the agenda
as [{"start_turn", "title"}]. Every item is verified against the source before use:

* start_turn is a chair-turn index that was in the input, and the items strictly increase;
* the title fuzzy-matches (difflib ratio >= MIN_RATIO) a word window of the announcing turn or of
  the next NEXT_PARAS paragraphs, and the emitted title is that VERBATIM window (snapped), never
  the model's own words;
* when fewer than half of the items verify, the answer is not used (regex fallback).

Answers are cached in cache/agenda/<sha256(model, prompt version, input)>.json (tracked in git), each
record carrying PROMPT_DIGEST (sha256 of PROMPT + SCHEMA) which must match on read, so
rebuilds are offline and byte-identical. API calls happen only when explicitly allowed (config
`agenda.api: true` or env GRL_AGENDA_API=1), at most `max_calls` per run (AgendaBudgetExceeded).
Without a cached answer and without API permission the caller falls back to the regex path.
"""
from __future__ import annotations

import difflib
import hashlib
import json
import logging
import os
import re
import threading
from collections import Counter
from pathlib import Path

log = logging.getLogger(__name__)

PROMPT_VERSION = 3          # v2: start_turn must be a bracketed chair-turn number; v3: title = the name only
MIN_RATIO = 0.85
NEXT_PARAS = 2
HEAD_CHARS = 700            # of every chair turn
TAIL_CHARS = 400            # of a chair turn longer than HEAD_CHARS + TAIL_CHARS ("…" between)
PROMPT = """You segment the verbatim transcript (stenogram) of a plenary session of the Senate of the \
Oliy Majlis of Uzbekistan into its agenda items. The text is Uzbek (Latin or Cyrillic script), sometimes \
Russian, and may contain speech-recognition errors.

You get the session title and the numbered list of the CHAIR's turns only ("Raislik qiluvchi"), each given \
by its first characters and, for long turns, also its last characters after " … ". Turn numbers are \
positions in the full transcript; other speakers' turns lie between them and are not shown.

Return every agenda item that the session actually takes up, in order, as \
{"start_turn": <number of the chair turn where the chair opens the consideration of that item>, \
"title": "<the item's name copied exactly from that turn>"}.

Rules:
- start_turn MUST be one of the numbers in square brackets below (a chair turn). Never give the number \
of the presenter's or any other speaker's turn: the item starts at the chair turn that announces it, \
even when the announcement is at the very end of that chair turn.
- An item starts where the chair announces it for consideration ("Navbatdagi masala …", "Kun tartibidagi \
birinchi masala …", "… koʻrib chiqishga oʻtamiz", "Endi … haqidagi masalaga oʻtamiz", "Oxirgi masala …", \
"Keyingi qonun …" and similar), or, when the chair reads out the name of the next law or item right after \
the vote on the previous one. Laws, resolutions, reports and information of officials, hearings, \
parliamentary inquiries, elections and appointments, approval of Senate Council decisions - each is an item.
- NOT an item: reading or approving the agenda as a whole, procedural votes, breaks, the opening and the \
closing of the session, questions, debate and remarks inside an item.
- If the session continues on another day, an item continued after a break is not a new item.
- The title must be copied character for character from the text of the start turn (same script, same \
spelling, including its errors). For an act name written in quotes give the text inside the outermost \
quotes; otherwise give the phrase that names the item (the act, report or question), without the \
words around it: no introductory words such as "Navbatdagi masala", "Birinchi masala", and nothing after \
the name such as "haqida", "koʻrib chiqamiz", "koʻrib chiqishga oʻtamiz", the committee or the speaker. \
Do not translate, correct, shorten in the middle or complete it. If the name is cut off by the excerpt, \
copy only the visible part.
- Give no other text than the JSON object."""
SCHEMA = {
    "type": "object",
    "properties": {"items": {"type": "array", "items": {
        "type": "object",
        "properties": {"start_turn": {"type": "integer"}, "title": {"type": "string"}},
        "required": ["start_turn", "title"], "additionalProperties": False}}},
    "required": ["items"], "additionalProperties": False}

PROMPT_DIGEST = hashlib.sha256(
    (PROMPT + "\n" + json.dumps(SCHEMA, sort_keys=True)).encode("utf-8")).hexdigest()   # checked on cache read
_WS = re.compile(r"\s+")
_EDGE = " \t\n.,;:!?«»“”„\"'()–—-"
_APOS = str.maketrans({"ʻ": "ʼ", "‘": "ʼ", "’": "ʼ", "'": "ʼ", "`": "ʼ"})


class AgendaBudgetExceeded(RuntimeError):
    """More agenda API calls than `max_calls` would be made in one run."""


def _key(s: str) -> str:
    return _WS.sub(" ", s.translate(_APOS)).strip().lower()


def snap_title(title: str, text: str, min_ratio: float = MIN_RATIO) -> str | None:
    """The word window of `text` (sizes len(title) ± 2 words) that best matches `title` by
    character-level difflib ratio (case, whitespace and apostrophe variants ignored), verbatim with
    single spaces and without edge punctuation / quotes; None below `min_ratio`."""
    want = _key(title.strip(_EDGE))
    if not want:
        return None
    words = text.split()
    n = len(want.split())
    best, best_key = None, (min_ratio, 0)
    for size in range(max(1, n - 2), n + 3):
        for i in range(0, max(len(words) - size, 0) + 1):
            cand = " ".join(words[i:i + size]).strip(_EDGE)
            if not cand:
                continue
            sm = difflib.SequenceMatcher(None, want, _key(cand), autojunk=False)
            if sm.real_quick_ratio() < best_key[0] or sm.quick_ratio() < best_key[0]:
                continue
            r = sm.ratio()
            key = (r, -abs(size - n))
            if key > best_key:
                best, best_key = cand, key
    return best


_OPENQ, _CLOSEQ = "«“„", "»”"


def _cut_unopened_quote(span: str) -> str:
    """"Elektron tijorat toʼgʼrisida”gi Qonunni" → "Elektron tijorat toʼgʼrisida": the title is the
    text inside the outermost quotes, so the span ends at the LAST closing quote that has no opening
    one in the span (a verbatim prefix; nested act names such as "“A toʼgʼrisida”gi Qonunning
    19-moddasiga … kiritish haqida”gi Qonun" keep their inner part)."""
    depth, cut = 0, None
    for i, ch in enumerate(span):
        if ch in _OPENQ:
            depth += 1
        elif ch in _CLOSEQ:
            if depth == 0:
                cut = i
            else:
                depth -= 1
    return (span[:cut].strip(_EDGE) or span) if cut is not None else span


_AP = "[ʼʻ‘’']"                                      # every apostrophe spelling of toʼgʼrisida
_TERM = rf"(?:to{_AP}g{_AP}risida|haqida|тўғрисида|ҳақида)"
# "…toʼgʼrisidagi qonun" / "…haqidagi masalani" / "…toʼgʼrisida qonun" → "…toʼgʼrisida"
_TRAIL = re.compile(rf"^(?P<keep>.*{_TERM}[»”]?)\s*(?:-?(?:gi|dagi|ги|даги)\s+)?"
                    r"(?:qonun|masala|loyiha|қонун|масала|лойиҳа)(?:ni|si|sini|ни|си|сини|i|и)?$", re.IGNORECASE)
# "…toʼgʼrisidagi qaror loyihasini" → "…toʼgʼrisidagi qaror loyihasi" (the item, without the accusative)
_TRAIL_ACC = re.compile(rf"^(?P<keep>.*{_TERM}[»”]?\s*-?(?:gi|dagi|ги|даги)\s+(?:qaror|қарор)\s+(?:loyihasi|лойиҳаси))"
                        r"(?:ni|ни)$", re.IGNORECASE)


def _unmatched_closes(span: str) -> int:
    depth = n = 0
    for ch in span:
        if ch in _OPENQ:
            depth += 1
        elif ch in _CLOSEQ:
            if depth:
                depth -= 1
            else:
                n += 1
    return n


def tidy_title(span: str, source: str) -> str:
    """Deterministic, still-verbatim clean-up of a snapped title (fix round 1):
    * "…toʼgʼrisidagi qonun" / "…haqidagi masalani" / "…toʼgʼrisida qonun" → the prefix ending at
      toʼgʼrisida / haqida (any apostrophe spelling, + its closing quote), only when the rest is just
      [-gi/-dagi] + qonun | masala | loyiha(si); "…toʼgʼrisidagi qaror loyihasini" loses the accusative;
    * a closing quote without an opening one ("Davlat boji toʼgʼrisida”gi … haqida") gets back the
      opening quote(s) that stand right before the span in `source` (stripped as edge punctuation).
    The result is a substring of the whitespace-normalized `source` (else `span` is returned)."""
    norm = " ".join(source.split())
    m = _TRAIL.match(span) or _TRAIL_ACC.match(span)
    out = m["keep"] if m else span
    need = _unmatched_closes(out)
    if need:
        at = norm.find(out)
        while at >= 0:
            j = at
            while j > 0 and norm[j - 1] in _OPENQ and at - j < need:
                j -= 1
            if j < at:
                out = norm[j:at] + out
                break
            at = norm.find(out, at + 1)
    return out if out in norm else span


def excerpt(text: str, head: int = HEAD_CHARS, tail: int = TAIL_CHARS) -> str:
    t = _WS.sub(" ", text).strip()
    return t if len(t) <= head + tail else f"{t[:head]} … {t[-tail:]}"


def build_input(title: str, chair: list[tuple[int, str]]) -> str:
    """The user message: session title + "[k] excerpt" lines of the chair turns."""
    lines = [f"Session: {title or '(untitled)'}", "", "Chair turns:"]
    lines += [f"[{k}] {excerpt(t)}" for k, t in chair]
    return "\n".join(lines)


def verify(items: list[dict], chair_text: dict[int, str], context: dict[int, str],
           min_ratio: float = MIN_RATIO) -> tuple[list[tuple[int, str]], Counter]:
    """Verified (start_turn, verbatim title) pairs of a model answer, plus drop / repair counts.

    `chair_text[k]` is the full text of chair turn k; `context[k]` the NEXT_PARAS paragraphs after
    it. A start_turn that is not a chair turn (the model named the presenter's turn) is repaired to
    the nearest preceding chair turn, but only if the title verifies there. Two items on the same
    start turn (an item settled inside the chair's turn, then the next one announced) keep the
    later one, which the following turns discuss; any other non-increasing item is dropped."""
    out: list[tuple[int, str]] = []
    why: Counter = Counter()
    chairs = sorted(chair_text)
    for it in items:
        k, title = it.get("start_turn"), it.get("title")
        if not isinstance(title, str) or not title.strip():
            why["no_title"] += 1
            continue
        if not isinstance(k, int) or k < 0:
            why["bad_turn"] += 1
            continue
        repaired = k not in chair_text
        if repaired:
            k = next((c for c in reversed(chairs) if c < k), None)
            if k is None:
                why["bad_turn"] += 1
                continue
        src = chair_text[k]
        span = snap_title(title, src, min_ratio)
        if span is None:
            src = f"{chair_text[k]}\n{context.get(k, '')}"
            span = snap_title(title, src, min_ratio)
        if span is None:
            why["bad_turn" if repaired else "title_not_found"] += 1
            continue
        span = tidy_title(_cut_unopened_quote(span), src)
        if out and k < out[-1][0]:
            why["not_increasing"] += 1
            continue
        if out and k == out[-1][0]:
            why["same_turn_replaced"] += 1
            out.pop()
        why["repaired_turn"] += repaired
        out.append((k, span))
    return out, why


def _write_json(f: Path, rec: dict) -> None:
    tmp = f.with_name(f.name + ".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, f)                                  # atomic: a crash never leaves half a record


class AgendaSegmenter:
    """(session title, paragraphs, turns) → verified [(start turn, title)] or None (use the regex path).

    `turns` are parse_stenogram's dicts {"a", "b", "speaker"} (paragraph range [a, b)); `is_chair`
    selects the chair turns that are shown to the model."""

    def __init__(self, model: str, cache_dir: Path, client=None, max_calls: int = 60,
                 allow_api: bool = False, min_ratio: float = MIN_RATIO):
        self.model, self.cache_dir, self.max_calls = model, Path(cache_dir), max_calls
        self.allow_api, self.min_ratio = allow_api, min_ratio
        self._client = client
        self.stats: Counter = Counter()
        self._lock = threading.Lock()

    def cache_file(self, user: str) -> Path:
        h = hashlib.sha256(f"{self.model}\n{PROMPT_VERSION}\n{user}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _call(self, user: str) -> str:
        with self._lock:
            if self.stats["api_calls"] + 1 > self.max_calls:
                raise AgendaBudgetExceeded(f"agenda budget of {self.max_calls} API calls per run would be exceeded")
            self.stats["api_calls"] += 1
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(max_retries=5)
        r = self._client.chat.completions.create(
            model=self.model, temperature=0,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "agenda", "strict": True, "schema": SCHEMA}},
            messages=[{"role": "system", "content": PROMPT}, {"role": "user", "content": user}])
        return (r.choices[0].message.content or "").strip()

    def answer(self, user: str) -> str | None:
        """Raw model answer for `user`: cached, else fetched when allowed, else None."""
        f = self.cache_file(user)
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
            if rec.get("model") == self.model and rec.get("prompt_version") == PROMPT_VERSION \
                    and rec.get("prompt_digest") == PROMPT_DIGEST:
                self.stats["cache_hits"] += 1
                return rec["answer"]
        if not self.allow_api:
            self.stats["offline_miss"] += 1
            return None
        ans = self._call(user)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        _write_json(f, {"model": self.model, "prompt_version": PROMPT_VERSION, "prompt_digest": PROMPT_DIGEST,
                        "answer": ans,
                        "input_sha256": hashlib.sha256(user.encode("utf-8")).hexdigest(),
                        "input_chars": len(user)})
        return ans

    def __call__(self, title: str, P: list[str], turns: list[dict], is_chair) -> list[tuple[int, str]] | None:
        self.stats["docs"] += 1
        chair = {k: "\n".join(P[tr["a"]:tr["b"]]) for k, tr in enumerate(turns) if is_chair(tr["speaker"])}
        if not chair:
            return None
        user = build_input(title, sorted(chair.items()))
        raw = self.answer(user)
        if raw is None:
            return None
        try:
            items = json.loads(raw)["items"]
            assert isinstance(items, list)
        except (ValueError, KeyError, TypeError, AssertionError):
            self.stats["bad_json"] += 1
            return None
        context = {k: "\n".join(P[turns[k]["b"]:turns[k]["b"] + NEXT_PARAS]) for k in chair}
        got, why = verify(items, chair, context, self.min_ratio)
        self.stats["items"] += len(items)
        self.stats["verified"] += len(got)
        self.stats.update({(r if r in ("repaired_turn", "same_turn_replaced") else f"dropped_{r}"): n
                           for r, n in why.items() if n})
        if not got or 2 * len(got) < len(items):
            self.stats["fallback_regex"] += 1
            log.info("agenda: %d/%d items verified for %r; regex fallback", len(got), len(items), title)
            return None
        self.stats["used"] += 1
        return got
