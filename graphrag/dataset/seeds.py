"""Seeds of the synthetic dataset (M3 Task 14, decision M3-R10): single = one fragment (facts of one vertex);
multi-hop = a graph walk (graphrag.dataset.walks). Documents are split first (data/gold/splits.json); a seed never spans
two splits. Cells split × pattern × language are exact (M3-R3 п. 1); a cell the pool cannot fill is a shortage
(reported, never backfilled). Every choice is an order by sha256(seed:key)."""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

# M3-R20 (user): one multi-hop pattern of genuine bridge / aggregation walks (graphrag.dataset.walks.quality_walks);
# the legacy walk patterns stay valid for the library functions and old artifacts
PATTERNS = ("single", "multi")
LEGACY_MULTI = ("refers_internal", "refers_cross", "amends")
HOP = {"single": "single", "multi": "multi", "refers_internal": "multi", "refers_cross": "multi", "amends": "multi"}
SAMPLING_ORDER = ("amends", "refers_cross", "refers_internal", "multi", "single")
ID_TYPES = ("реквизиты", "заголовок", "преамбула")          # act / session names: never a single-hop seed (M3-R9)


@dataclass(frozen=True)
class Seed:
    seed_id: str
    pattern: str
    lang: str
    chunks: tuple[str, ...]
    docs: tuple[str, ...]
    split: str
    section: str
    walk: tuple[tuple[str, str, str], ...] = ()

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Seed":
        return cls(**{**d, "chunks": tuple(d["chunks"]), "docs": tuple(d["docs"]),
                      "walk": tuple(tuple(h) for h in d.get("walk", ()))})


def _h(seed, key: str) -> str:
    return hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()


def qreal_exclusion(items, records, pairs) -> set[str]:
    """D8: Q-real gold (core ∪ support) ∪ alt (alt_equiv ∪ alt_mention) ∪ context (M3-R9: chunks naming the cited act /
    session) ∪ the fragments at the same structural path in the other documents of the gold document's linked group
    (translations / editions)."""
    from graphrag.eval.gold import strict_doc_groups
    group = strict_doc_groups(sorted(r.doc_id for r in records), pairs)
    members = defaultdict(list)
    for d, g in group.items():
        members[g].append(d)
    parent = {c.chunk_id: c.parent for r in records for c in r.chunks}
    by_node = defaultdict(list)
    for r in records:
        for c in r.chunks:
            by_node[c.parent].append(c.chunk_id)
    out: set[str] = set()
    for it in items:
        out |= set(it.gold) | set(it.alt) | set(it.context)
        for cid in it.gold:
            _, _, suffix = parent[cid].partition("/")
            for d in members[group[it.doc_id]]:
                if d != it.doc_id:
                    out |= set(by_node.get(f"{d}/{suffix}" if suffix else d, []))
    return out


def block_of(records, size: int) -> dict[str, str]:
    """chunk_id -> "<doc_id>#B<k>": `size` consecutive fragments of one document (by start) -- the ds_facts unit."""
    out = {}
    for r in records:
        for i, c in enumerate(sorted(r.chunks, key=lambda c: c.start)):
            out[c.chunk_id] = f"{r.doc_id}#B{i // size}"
    return out


def allocate(n: int, shares: dict[str, float]) -> list[str]:
    raw = {k: n * v / sum(shares.values()) for k, v in shares.items()}
    cnt = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (-(raw[k] - cnt[k]), k))[: n - sum(cnt.values())]:
        cnt[k] += 1
    seq = []
    while len(seq) < n:
        for k in sorted(cnt):
            if cnt[k] > 0:
                seq.append(k)
                cnt[k] -= 1
    return seq


def cell_counts(n: int, shares: dict[str, float]) -> dict[str, int]:
    c = Counter(allocate(n, shares))
    return {k: c.get(k, 0) for k in shares}


def content_chunks(records) -> set[str]:
    """M3-R14 (2): fragments that can bear a fact -- not an act / session-name type (ID_TYPES: реквизиты, заголовок,
    преамбула) and not boilerplate by the graph's chunk classes (graphrag.kg.semantic.chunk_reason with
    config/graph.yaml related_defaults + related: related_boilerplate formulas / clauses / gazette lists, sentence
    classes such as greeting, chair procedure, session opening / closing)."""
    from graphrag import config
    from graphrag.kg.semantic import DEFAULTS, chunk_reason, compile_sentence_rules, speaker_names
    cfg = {**DEFAULTS, **(config.load("graph").get("related") or {})}
    rules = compile_sentence_rules(cfg.get("sentence_rules"))
    speakers = speaker_names(records, cfg)
    names = {w for x in speakers for w in x.split()}
    out = set()
    for r in records:
        ptype = {n.node_id: n.type for n in r.structure}
        for c in r.chunks:
            t = ptype.get(c.parent)
            if t not in ID_TYPES and chunk_reason(c, t, r.meta.section, cfg, rules, speakers, names) is None:
                out.add(c.chunk_id)
    return out


def single_pool(records, ok: set[str]) -> list[tuple]:
    ntype = {n.node_id: n.type for r in records for n in r.structure}
    return [((c.chunk_id,), (r.doc_id,), ()) for r in records for c in r.chunks
            if c.chunk_id in ok and ntype.get(c.parent) not in ID_TYPES]


def _overlap_ok(chunks, by_chunk) -> bool:
    """M3-R12 (1): a seed shares at most one fragment with any other seed and never contains (or is contained in) the
    whole C_ref of another -- so a single-fragment seed never reuses a picked fragment."""
    x = set(chunks)
    for c in x:
        for y in by_chunk.get(c, ()):
            inter = len(x & y)
            if inter > 1 or inter == len(y) or inter == len(x):
                return False
    return True


def sample_seeds(pool, splits: dict[str, str], doc_section: dict[str, str], quotas: dict[str, int],
                 ratios: dict[str, float], lang_shares: dict[str, float], section_shares: dict[str, float],
                 max_per_doc_cell, seed, forced=(), prefer=None,
                 max_per_chunk: int | None = None,
                 cell_quota: dict[tuple[str, str], int | None] | None = None) -> tuple[list[Seed], dict[str, int]]:
    """Cells split × pattern × language (× section for single) filled exactly from `pool` items (chunks, docs, walk)
    in sha256(seed:chunks) order (`prefer`-ed singles first). Inside a (pattern, split[, section]) cell the next item
    goes to the language with the lowest filled share of its need (have / want; ties: config order), so a scarce
    pool is spread in proportion to need (M3-R14 (4)). Seeds share <= 1 fragment pairwise, never a whole C_ref
    (M3-R12 (1)); a fragment is in at most `max_per_chunk` seeds (M3-R14 (3)). Shortage = what a cell could not get;
    never backfilled. `cell_quota` {(pattern, split): n | None} overrides the ratio share of one split (M3-R15): an int
    is that split's seed demand, None = every available seed (languages still in proportion; no shortage recorded).
    `max_per_doc_cell`: an int, or doc_id -> cap (M3-R21: doc_cap, proportional to the document's size)."""
    cell_quota = cell_quota or {}
    cap = max_per_doc_cell if callable(max_per_doc_cell) else (lambda d, _n=int(max_per_doc_cell): _n)
    by_chunk: dict[str, list[set]] = defaultdict(list)
    have, per_doc = Counter(), Counter()

    def take(chunks):
        x = set(chunks)
        for c in x:
            by_chunk[c].append(x)

    def fits(chunks) -> bool:
        if max_per_chunk is not None and any(len(by_chunk.get(c, ())) >= max_per_chunk for c in chunks):
            return False
        return _overlap_ok(chunks, by_chunk)

    for f in forced:
        take(f.chunks)
        have[(f.pattern, f.split, f.lang, f.section if f.pattern == "single" else None)] += 1
        per_doc.update((f.pattern, f.split, f.lang, d) for d in f.docs)
    picked, shortage = list(forced), Counter()
    for pat in SAMPLING_ORDER:
        items = sorted((x for x in pool.get(pat, []) if len({splits[d] for d in x[1]}) == 1),
                       key=lambda x: (not (prefer and pat == "single" and prefer(x[0])), _h(seed, "|".join(x[0]))))
        for sp, m in cell_counts(quotas.get(pat, 0), ratios).items():
            unbounded = (pat, sp) in cell_quota and cell_quota[(pat, sp)] is None
            if (pat, sp) in cell_quota:
                m = sum(1 for x in items if splits[x[1][0]] == sp) if unbounded else int(cell_quota[(pat, sp)])
            want: dict[tuple, int] = {}
            for lang, n in cell_counts(m, lang_shares).items():
                for sec, k in (cell_counts(n, section_shares).items() if pat == "single" else [(None, n)]):
                    want[(lang, sec)] = k
            for sec in dict.fromkeys(sec for _, sec in want):
                cand = [x for x in items if splits[x[1][0]] == sp and (sec is None or doc_section[x[1][0]] == sec)]
                langs = [lang for lang, s2 in want if s2 == sec]
                cursor = {lang: 0 for lang in langs}
                active = [lang for lang in langs if have[(pat, sp, lang, sec)] < want[(lang, sec)]]
                while active:
                    lang = min(active, key=lambda l: (have[(pat, sp, l, sec)] / want[(l, sec)], langs.index(l)))
                    i = cursor[lang]
                    while i < len(cand):
                        chunks, docs, walk = cand[i]
                        i += 1
                        if all(per_doc[(pat, sp, lang, d)] < cap(d) for d in docs) and fits(chunks):
                            take(chunks)
                            per_doc.update((pat, sp, lang, d) for d in docs)
                            have[(pat, sp, lang, sec)] += 1
                            picked.append(Seed(f"{pat}-{_h(seed, '|'.join(chunks))[:10]}", pat, lang,
                                               tuple(chunks), tuple(docs), sp, doc_section[docs[0]], tuple(walk)))
                            break
                    else:
                        active.remove(lang)                     # pool exhausted for this language
                    cursor[lang] = i
                    if lang in active and have[(pat, sp, lang, sec)] >= want[(lang, sec)]:
                        active.remove(lang)
                for lang in langs:
                    if have[(pat, sp, lang, sec)] < want[(lang, sec)] and not unbounded:
                        shortage[f"{pat}|{sp}|{lang}"] += want[(lang, sec)] - have[(pat, sp, lang, sec)]
    return sorted(picked, key=lambda s: s.seed_id), dict(sorted(shortage.items()))


def doc_cap(records, floor: int = 3, share: float = 0.02):
    """M3-R21 (controller): seeds per document per cell = max(floor, ceil(share x the document's fragments))."""
    import math
    n = {r.doc_id: len(r.chunks) for r in records}
    return lambda d: max(int(floor), math.ceil(float(share) * n.get(d, 0)))


def effective_needs(final: dict[str, int], got: dict[str, int], splits=("val", "test"), src: str = "amends",
                    dst: str = "refers_internal") -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    """M3-R15: in `splits`, the `src` pattern needs only what it got (quality seeds available), and its shortfall
    against the final need moves to `dst` of the same split -- same language first; a language `dst` cannot cover
    moves to the other languages' surplus in that split (largest surplus first, ties by name). `dst` never falls
    below its own final need per language. Keys "pattern|split|lang". Returns (effective final need, transfer
    {"split|lang": n} actually planned per destination language, below {cell: missing})."""
    eff = dict(final)
    transfer: dict[str, int] = {}
    below: dict[str, int] = {}
    for sp in splits:
        langs = sorted({k.split("|")[2] for k in final if k.split("|")[1] == sp})
        short = {l: max(0, final.get(f"{src}|{sp}|{l}", 0) - got.get(f"{src}|{sp}|{l}", 0)) for l in langs}
        for l in langs:
            eff[f"{src}|{sp}|{l}"] = final.get(f"{src}|{sp}|{l}", 0) - short[l]
        add = dict(short)
        base = {l: final.get(f"{dst}|{sp}|{l}", 0) for l in langs}
        g = {l: got.get(f"{dst}|{sp}|{l}", 0) for l in langs}
        spill = 0
        for l in langs:
            over = base[l] + add[l] - g[l]
            if over > 0:
                take = min(over, add[l])
                add[l] -= take
                spill += take
        for l in sorted(langs, key=lambda l: (-(g[l] - base[l] - add[l]), l)):
            room = max(0, g[l] - base[l] - add[l])
            k = min(room, spill)
            add[l] += k
            spill -= k
        if spill:                                        # nobody could take it: charge it where it came from
            for l in langs:
                if short[l] and spill:
                    k = min(spill, short[l])
                    add[l] += k
                    spill -= k
        for l in langs:
            eff[f"{dst}|{sp}|{l}"] = base[l] + add[l]
            transfer[f"{sp}|{l}"] = add[l]
    for k, n in eff.items():
        if got.get(k, 0) < n:
            below[k] = n - got.get(k, 0)
    return dict(sorted(eff.items())), dict(sorted(transfer.items())), dict(sorted(below.items()))


def dup_content(records, min_chars: int = 40) -> dict[str, dict]:
    """M3-R25: duplicate content across documents -- chunk_id -> {key, twins, splits_of_twins} for every fragment
    whose normalized text (wsn, >= min_chars) also occurs in ANOTHER document (the same Senate decision ingested
    twice, an article repeated verbatim in two editions of a law, a repeated amendment head). key = sha256 of the
    text; twins = the other chunk ids with that text (all documents)."""
    from graphrag.eval.gold import wsn
    by: dict[str, list[str]] = defaultdict(list)
    for r in records:
        for c in r.chunks:
            t = wsn(c.text)
            if len(t) >= int(min_chars):
                by[hashlib.sha256(t.encode("utf-8")).hexdigest()].append(c.chunk_id)
    out = {}
    for k, ids in by.items():
        if len({i.split("/", 1)[0] for i in ids}) > 1:
            for i in ids:
                out[i] = {"key": k, "twins": sorted(x for x in ids if x != i)}
    return out


def content_signature(chunks, dup: dict) -> tuple:
    """The content of a C_ref: duplicate fragments by their text key, others by id (order-free)."""
    return tuple(sorted(dup[c]["key"] if c in dup else c for c in chunks))


def cross_split_twin(chunks, dup: dict, split_of: dict[str, str]) -> bool:
    """A fragment of `chunks` has a twin (same text) in a document of ANOTHER split: a leak whatever the question."""
    for c in chunks:
        s = split_of.get(c.split("/", 1)[0])
        if any(split_of.get(t.split("/", 1)[0]) != s for t in (dup.get(c) or {}).get("twins", ())):
            return True
    return False


# ---- seed hygiene (generator skip log): unnameable documents and low-content chunks ------------------------------
import re as _re

_TITLE_WORDS = _re.compile(r"(?i)тўғрисида|ҳақида|toʻgʻrisida|toʼgʼrisida|to'g'risida|haqida|қарор|qaror|қонун|qonun|"
                           r"кодекс|kodeks|хулоса|xulosa|majlis|мажлис|ратификац|ratifikats|о\s|об\s|закон|постановлен|"
                           r"заключени|axborot|ахборот|ҳисобот|hisobot|маъруза|maʼruza|доклад|информац|сўров|soʻrov|soʼrov|запрос")


def bad_title(title: str) -> str | None:
    """Why a document's act title cannot name it (None = usable): a generic heading of <= 4 words without a title
    word ("ЙЎЛ ХАРИТАСИ", "Ақлли сув", "Tashkent Cotton Textile"), a letterhead (mostly upper case / mixed English),
    a slogan-like sentence, or a truncated quoted title."""
    t = " ".join((title or "").split())
    if not t:
        return "empty"
    words = t.split()
    letters = [c for c in t if c.isalpha()]
    if len(letters) > 20 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        return "letterhead"
    if _re.search(r"\b(THE|OF|AND|CENTRE|CENTER|REPUBLIC)\b", t):
        return "letterhead"
    if len(words) <= 4 and not _TITLE_WORDS.search(t):
        return "generic_heading"
    if (t.count('"') + t.count("“") + t.count("”")) % 2 == 1 or t.count("«") != t.count("»"):
        return "truncated"
    if _re.match(r"(?i)(biz|men|мен|биз|bu|бу)\s", t) or (len(words) > 12 and not _TITLE_WORDS.search(t)):
        return "sentence"
    return None


_HEADER_ONLY = _re.compile(r"(?i)(киритилсин|kiritilsin|внести следующие|вносятся следующие)\s*:?\s*$")
_GAZETTE = _re.compile(r"(?i)(ахборотномаси|axborotnomasi|ведомости|qonunchilik maʼlumotlari|қонунчилик маълумотлари)")


def low_content(text: str, speaker: str | None = None) -> str | None:
    """A chunk generators keep skipping: an amending article's header only (citations + "… киритилсин:"), a headless
    list tail (starts lower-case), or a chair's filler line."""
    t = (text or "").strip()
    if _HEADER_ONLY.search(t) and (len(t) < 600 or _GAZETTE.search(t)):
        return "amending_header"
    if t[:1].islower() or t[:1] in "–—-;,":
        return "list_tail"
    if speaker and _re.match(r"(?i)rais|раис", speaker) and len(t) < 500:
        return "chair_filler"
    return None
