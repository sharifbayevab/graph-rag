# GENERATOR INSTRUCTIONS: legal QA dataset generator (version 8, rulings M3-R23 / M3-R25 / M3-R29 / M3-R31 / M3-R32)

These rules apply to ANY generator model or agent that receives a work packet: the same rules, the same code checks,
the same strict reader.

You write single-turn question-answer pairs for a legal retrieval dataset over Uzbek parliamentary documents: laws,
Senate decisions, committee conclusions, plenary transcripts and other documents. A retrieval agent will be trained
to find the fragments your question needs, so every pair must be natural, self-contained and fully supported by the
fragments of its seed.

You receive one WORK PACKET (JSON). Write your output to the packet's `output.path`: one JSON object per line
(JSONL), exactly one line per seed of the packet, in packet order. Write nothing else in that file. Do not read other
packets, earlier outputs, or anything outside the packet. Do not use outside knowledge. The packet is the only source.

After you write your file, every line goes through automatic code checks (listed below) and then through an
independent blind judge with a per-fact ablation. After that, a separate strict reader, who is not you, reads every
pair using the 7-criterion rubric below. Only pairs the reader rates GOOD enter the dataset. **If no GOOD pair is
possible for a seed, output `skip`.** A skip costs nothing. A weak pair wastes a verification call and is rejected
anyway.

## 1. The packet

- `seeds[]`: each seed has `seed_id`, a `cell` {split, pattern, lang} and `LANG`, the target language of the question
  and the answer.
- **Single seed** (`pattern: single`): one fragment `F1` with its `act` (the document's own title), `where` (the
  document title plus the article, part or item path), `kind` (law / Senate decision / committee conclusion / plenary
  transcript / other document), `amended` (acts the fragment amends or cites, from the graph) and the full `text`.
  A plenary-transcript fragment also carries `session` {`date` (ISO), `title` (e.g. "Senatning 6-yalpi majlisi
  29-30.04.25")} and `speaker` (the name or role of the person speaking, e.g. "Raislik qiluvchi"), when known.
  From the verified legal graph a fragment may also carry `discussed_act` (the law / draft of the agenda item a speech
  belongs to) and, for an amending-law item, `amended_act` + `amended_unit` (the one act and article / part it amends).
- **Multi seed** (`pattern: multi`): an ANCHOR fragment `A` and up to 8 NEIGHBOUR fragments `N1`…`N8`. Each neighbour
  has `source`, `relation` and `edges`, the graph link to the anchor:
  - `refers_to`: the anchor cites this provision, or this fragment deals with the provision the anchor cites;
  - `amends`: the anchor amends or repeals this provision, or this fragment deals with the same amended provision;
  - `concept`: this fragment defines a term the anchor regulates;
  - `related`: a topically related fragment.
- `facts[]` (when present): atomic facts extracted earlier and checked against the text. Each has `fact_id`, `type`
  (number / name / term / clause), `statement`, `value` and `quote`, and the value and quote are verbatim source
  spans. When `facts_cached` is false, there are no facts, and you cite verbatim quotes yourself.

## 2. Output schema (one line per seed)

```json
{"seed_id": "...", "question": "...", "answer": "...", "lang": "<cell.lang>",
 "used_facts": [{"fact_id": "F1.0", "value": "..."} or {"fragment": "F1", "quote": "...", "value": "..."}],
 "neighbor_id": null or "N3", "sources": ["F1"] or ["A", "N3"], "notes": "..."}
```

or, when no good pair exists:

```json
{"seed_id": "...", "skip": true, "reason": "..."}
```

- `lang` must equal the seed's `cell.lang`.
- `used_facts`: the facts the pair rests on.
  - Cite a packet fact by its `fact_id`, and copy its `value` unchanged.
  - Otherwise give `fragment`, a `quote` and a `value`. The quote is a verbatim span of that fragment's text, 20–300
    characters, copied character by character in the original script, never joined with "…". The value is the
    shortest span inside the quote that answers the question. A number keeps its unit or noun ("30 кун",
    "5 фоиз"), and a value is never a bare generic word or the act's own name.
  - **Every value is at most 300 characters**, clause values included. Never use a whole amended part or article
    as a value; cite the shortest clause that answers. Code rejects a longer value.
  - Every value you list must be stated in the answer. Inflection, transliteration or, for a Russian answer over an
    Uzbek text, a translation all count as stated. For a translated name or term, add the original once in «»
    after it.
- **single**: `neighbor_id` is null and `sources` is ["F1"]. Use facts of F1 only.
- **multi**: `neighbor_id` is the one neighbour you chose, and `sources` is ["A", "<neighbor_id>"]. `used_facts` must
  contain:
  - one ANCHOR fact: a fact only the anchor states, which the reader needs in order to reach the neighbour, e.g. the
    link span "64-моддасида" together with what the anchor says about it;
  - at least one NEIGHBOUR fact that the answer states and the question does not reveal.
  - Every value must come from A or from the chosen neighbour.
- `notes`: one short sentence for the controller, explaining why the pair is genuine or why you chose that neighbour.
  The reader never sees your notes.
- `skip.reason`: say why no GOOD pair exists: a filler line, boilerplate, only a list header, no fact a citizen
  would ask about, no genuinely needed neighbour, and so on.

## 3. Rules for the question and the answer (ds_generate v5 rules)

- **ONE information need.** Never join two questions ("X ким ва Y қанча?", "кто … и сколько …?", "… and what …?").
- **single**: ask about one fact of the fragment.
- **multi (bridge)**: pick the neighbour only if it is GENUINELY needed together with the anchor:
  - the question starts from the anchor (what the anchor refers to, amends or uses);
  - its answer is a fact that only the neighbour states: the cited article's term, the definition of the term, the
    old or new wording, the body named in the cited provision;
  - without the neighbour the question cannot be answered, and without the anchor the reader would not know which
    provision is meant.

  The question may POINT at the anchor ("the provision that Art. 24 of Law X refers to"). It must **never restate
  the anchor's content** (its rule, its list, its principle), and it must **never name the neighbour's topic or
  wording** ("the order of creditors' claims", "the definition of an insurer"). Either would let one fragment alone
  answer the question, and the verification ablation then rejects the pair. The link is never a made-up condition
  ("if …") that the text does not state. If only topical similarity, a restatement, or the same fact in both
  fragments links them, output `skip` with the reason "no genuinely needed neighbour".
- **No definition padding.** Do not pick a `concept` neighbour, or any neighbour, only to add a definition, a
  restatement or context the answer does not need. If the anchor alone answers the question, the neighbour is
  padding: skip.
- **Mandatory multi self-test** (write a pair only if all answers are "no"):
  1. Could the ANCHOR alone, as written, answer the question fully?
  2. Could the NEIGHBOUR alone, as written, answer the question fully, including knowing which provision is
     meant?
  3. Does the question repeat a clause of either fragment or name the neighbour's subject?

  If any answer is "yes", output `skip` with the reason "one fragment suffices".
- **multi (aggregation)**: this is also valid when anchor and neighbour deal with the SAME provision. Ask one question
  whose answer needs one value from EACH fragment: what each act changed in it, how it read before and after, or which
  acts or bodies deal with it.
- **No leak.** The question must not reveal any value it asks for, in any form: verbatim, inflected, translated,
  transliterated, as digits or words, or as a synonym. No pointers ("ушбу / мазкур / bu hujjat / данный / настоящий /
  этот документ").
- **Self-contained.** A reader who has NEVER seen the fragment must understand exactly what is asked. Name the concrete
  subject: which draft, which villages, which law, which body. If the fragment lists several items, either ask for
  all of them or name the specific one. Do not build a pair on a transcript filler line.
- **Act identity.**
  - Name the act by the fragment's own `act` title: verbatim, transliterated, or in Russian, its standard Russian
    rendering.
  - **Amending laws: name the AMENDED provision.** A fragment of an amending law ("… ўзгартириш ва қўшимчалар
    киритиш тўғрисида") changes another act. The question names that amended act and its article ("ст. 347
    Таможенного кодекса", "«Товар белгилари тўғрисида»ги Қонуннинг 30-моддаси") -- from the fragment TEXT, or from
    the packet's `amended_act` + `amended_unit` when the text does not name it (the graph's single AMENDS target is
    verified); only when neither gives it, output `skip` ("amended act not named"). NEVER use the amending law's
    own article or item ("Қонуннинг 2-моддаси билан киритилаётган норма", "статья 6 Закона о внесении изменений") as
    the locus of the rule; code rejects an article reference over an amending law that names no amended act.
  - Every act title you write must be the packet's `act` title (or an amended act named in the text), faithfully
    rendered; a shortened, re-worded or garbled title fails the code check and the reader.
  - Never name an act, decree or programme that the text only mentions, and never use the year's slogan as a name.
  - Never cite an article, part or item number without its act.
  - An `act` may end with a date "(2023-02-14)" and a bracketed document id "[QONUNLAR-f79c702c]" that tell
    same-titled documents apart. Never put the bracketed id in a question. Use the date only when it is needed to
    identify the act, and write it in natural words.
- **Plenary transcripts.** A question over a transcript must identify the session by its DATE, taken from `session`:
  - a one-day sitting: that day, month and year ("2025 йил 17 мартдаги Сенат ялпи мажлисида");
  - a MULTI-DAY sitting (the title shows a range, e.g. "29-30.04.25"): the FULL range ("2025 йил 29–30 апрелдаги
    ялпи мажлисда" / "на пленарном заседании Сената 29–30 апреля 2025 года"), never one day of it, unless the
    fragment text itself states the day;
  - code checks that the year, month and a day of the sitting are present.

  Every date you write must be one of `session.dates` (or their full range); a date, day or range from anywhere
  else -- the title's wording, the text, your memory -- is rejected, and so is a sitting number alone ("59-yalpi
  majlis") without its date.
  **Non-contiguous sittings**: write the days EXACTLY as `session.dates` lists them; a range "a–b" only for
  consecutive days (code rejects a range that spans a day missing from `session.dates`).
  - BAD: "2024 йил 18–21 декабрдаги ялпи мажлисда" when `session.dates` = 2024-12-18, 2024-12-19, 2024-12-21.
    GOOD: "2024 йил 18–19 ва 21 декабрдаги ялпи мажлисда" (ru: "18–19 и 21 декабря 2024 года").
  **No relative years**: never write «жорий йил», «бу йил», «ўтган йил», «joriy yil», «текущий год», «в этом
  году», «в прошлом году» alone -- resolve them to the actual year from the meta / text (the sitting's year for
  "this year"), or give the year right after them (code rejects a relative year without one).
  - BAD: "Ўзбекистон биринчи миллий маърузасини жорий йилнинг июль ойида тақдим этади." GOOD: "Ўзбекистон биринчи
    миллий маърузасини 2023 йилнинг июль ойида тақдим этади." (the sitting is 2–3 March 2023).
  A speech question must name the LAW (or draft) the speech discusses: from the fragment text, or from the packet's
  `discussed_act` (the agenda item of the sitting); only when neither names it, output `skip` ("the discussed law
  is not named").

  Name the speaker ONLY from the packet's `speaker` field ("Эркин Файзиевич Гадоев", "раислик қилувчи"), never a name
  guessed from the text, a neighbouring speech or outside knowledge; if `speaker` is absent, describe the role the
  text itself gives. "A speaker", "one of the senators" or an undated "plenary session" fail.
  **A speaker's title, position, role or stance** ("сенатор", "депутат", "вазир", "раис ўринбосари", "қўмита раиси",
  "қўллаб-қувватлади" / "supported", "қарши чиқди" / "opposed") may be stated ONLY if the fragment text or the meta
  states it; otherwise refer to the speaker by the name exactly as `speaker` gives it, and report what they said,
  not their stance (code rejects a role word absent from the fragment text and meta).
  - BAD: "2024 йил 18 майдаги ялпи мажлисда сенатор Мубин Мирзаев қонунни қўллаб-қувватлаб, жарима миқдори
    ҳақида нима деди?" when neither the text nor `speaker` says he is a senator or that he supported the law.
    GOOD: "2024 йил 18 майдаги Сенат ялпи мажлисида Мубин Муҳидинович Мирзаевнинг айтишича, тадбиркор қонунни
    такроран бузса, қанча жарима солинади?"
- **Attribute correctly.**
  - A speech is what a speaker said at a Senate plenary session (ru: "на пленарном заседании Сената").
  - A conclusion is a committee's. A decision is the Senate's.
  - Do not call a speech a law or a decision, and do not call a conclusion on an adopted law a "draft".
- **Names from the fragments only.** Every proper name (person, place, organisation, act) in the question and the
  answer must occur in the fragment texts or their act titles or `amended` names. Never import names from elsewhere,
  and never resolve "ушбу туман" to a district the text does not name.
- **Paraphrase -- no blanked-sentence copies.** Ask about the rule's effect in your own words. NEVER take a fragment
  sentence, replace one element with a question word and keep the rest in the same order ("Qonun Senatga … qachon
  kelib tushgan?" over "Qonun Senatga … 27-fevral kuni kelib tushgan"); this is rejected in any language, including
  a translation or a transliteration of the sentence. Do not restate the whole fragment so that the answer only
  repeats one phrase. Code measures how much of the question is one fragment sentence in order (see §4), but it
  cannot see a TRANSLATED copy -- the readers can, and reject it. Examples from the readers' verdicts:
  - BAD (ru over an Uzbek item, translated with only the term blanked): "В какой срок постановление Сената … поручает
    Министерству здравоохранения разработать проект Кодекса об охране здоровья населения и внести его в Кабинет
    Министров?" over "Sogʼliqni saqlash vazirligi ikki oy muddatda … kodeksi loyihasini ishlab chiqsin va Vazirlar
    Mahkamasiga taqdim qilsin."
    GOOD: "Сколько времени Сенат дал Минздраву на подготовку проекта кодекса об охране здоровья населения?"
  - BAD (ru over an Uzbek article, translated with the term blanked): "В какой срок суд, вынесший решение, обязан
    направить поступившую кассационную жалобу (протест) вместе с делом в суд кассационной инстанции согласно
    статье 226 …?" over "Ҳал қилув қарорини қабул қилган суд шикоят (протест) келиб тушган кундан эътиборан беш
    кунлик муддатда уни кассация инстанцияси судига иш билан бирга юбориши шарт."
    GOOD: "Сколько дней есть у суда первой инстанции, чтобы переслать кассационную жалобу вместе с делом по статье
    226 Кодекса об административном судопроизводстве?"

  Rule of thumb: first decide WHAT a citizen wants to know, then ask it in your own short words; never start from
  the fragment's sentence.

- **Three more reader-flagged defects (wave 1) and their fixes:**
  1. BAD (ru answer with a pasted Uzbek figure): "Работы … не завершены полностью по 23 проектам первого сезона
     (февраль–март) («23 та лойиҳа»)." GOOD: "Работы по ремонту и оснащению не завершены по 23 проектам первого
     сезона (февраль–март)." -- the figure is already translated; no Uzbek gloss.
  2. BAD (amending law's own article as the locus, amended act not in the text): "…»ги Қонуннинг 2-моддаси билан
     киритилаётган нормага кўра, товар белгиси эгасининг инсофсиз рақобат қилгани аниқланса, гувоҳнома нима бўлади?"
     over a fragment that only quotes the new wording. GOOD: `skip` -- the fragment never names the amended act (the
     trademark law), so the provision cannot be identified.
  3. BAD (speech without the law it discusses): "2024-yil 23–24-oktabrdagi Senat yalpi majlisida Ibroxim
     Ashiraliyevich Sultanov qonun bilan Soliq kodeksiga kiritilayotgan oʻzgartirishlarga koʻra … qanday koeffitsiyent
     qoʻllanilishini aytdi?" -- the speech only says "qonun", so the law is not identified. GOOD: name it from the
     text, or from `discussed_act` ("… ipakchilikni rivojlantirishga oid qonun …" only if that is what
     `discussed_act` says); `skip` only when neither names it.
- **Question word.** The interrogative must fit the answer type and keep the rule's logic, and a condition is never
  inverted:
  - threshold -> "up to what amount / at what size" (not "from what amount … not");
  - fee or manner -> "is it paid / on what terms";
  - term -> "within what period";
  - body -> "which body".
- **Style.** Ask as a citizen or official would. Vary the form (ким / нима / қачон / қанча муддатда / қандай шартда /
  қайси ҳолда; kim / nima / qachon / qaysi; кто / что / когда / в какой срок / при каком условии). End with "?".
- **The answer** is 1–3 sentences. It states every value you use and nothing unsupported. It contains no number that
  is not in the fragments.
- **Script and language.** Write the question and the answer ENTIRELY in LANG, fluently and grammatically:
  - uz-Latn: Latin letters only (oʻ gʻ with ʻ). Transliterate Cyrillic values and titles letter by letter.
  - uz-Cyrl: Cyrillic only (ў ғ қ ҳ). Never ʻ / ʼ or any Latin letter, not even inside quotes. Transliterate Latin
    values and titles letter by letter (Oʻzbekiston -> Ўзбекистон).
  - ru: ENTIRELY Russian, in Cyrillic, in the question AND the answer. Translate every value, term, name and act
    title, keeping numbers exact, with the standard legal terms of the GLOSSARY. The Uzbek original of ONE name or
    term may follow its translation once in «» (at most 6 words, no figures); never paste Uzbek clauses, quantities
    or glosses («23 та лойиҳа», «maqsadga muvofiq emas deb topdi») and never leave an Uzbek word outside «» (code
    rejects any ў қ ғ ҳ outside «», a second Uzbek «» span, or an answer gloss with figures or over 6 words). An
    Uzbek or Latin-script title alone inside a Russian question is wrong. Render every Latin-script name in Russian
    (Cyrillic) too ("Gadoyev" -> "Гадоев"), with the standard Russian form of place names ("Чартакский район").
  - Write as a native speaker would: fluent, idiomatic Uzbek (correct case endings, no calques from Russian) and
    natural Russian legal style; re-read every question and answer once for grammar before you write the line.

## 4. Code checks your line must pass (automatic, M3-R3 / M3-R20 / M3-R22 / M3-R29)

Any single failure rejects the line. The thresholds are calibrated on 340 real user questions.

1. Schema: the seed_id is in the packet; every value has at most 300 characters; lang = cell.lang; the question ends with "?" and has at least 15
   characters; the answer is non-empty; `sources` and `neighbor_id` match the pattern; every `fact_id` exists; every
   `quote` is a verbatim span (whitespace, case and quote marks are normalised) of the named fragment, and its value
   lies inside the quote.
2. Values: every typed value (number / date / name / term, 8 tokens or fewer) must be stated in the answer as whole
   tokens. Inflection and script do not matter, and number words count as digits. A number value must keep all of
   its numbers. Clause values and translated names or terms go to the judge.
3. Chain: single, every used fact comes from F1; multi, the answer states a neighbour fact, and the anchor and the
   neighbour each contribute a fact that the other fragment's text does not contain (otherwise the pair is
   `redundant`).
4. Leak: no value the answer states appears in the question, and no number of an answer value appears in the
   question (numbers of the act titles are exempt).
5. Numbers: every number in the question and the answer occurs in the fragments or their act/where metadata.
6. Script purity (as in §3), and no pointer words ("ушбу ҳужжат", "данный документ", "this document").
7. Compound: two interrogatives joined by a conjunction (ва / va / и / ҳамда / also) are rejected.
8. Article without act: an article, part or item number must come with a distinctive act name or a code.
9. Foreign names: capitalised names and «quoted» spans must occur in the fragments, act titles or amended acts.
10. Wrong act: a quoted act title (3 or more content words) must be the fragment's own act or an amended act.
11. Copy: the longest run of consecutive question tokens copied from a fragment (act titles and numbers exempt) is at
    most **7** tokens for uz-Cyrl / uz-Latn, and at most **6** for ru. Transliteration counts as copying.
12. Paraphrase: the share of the question's content lemmas found in the fragments (titles and numbers exempt) is at
    most **0.8462** for uz-Cyrl / uz-Latn, and at most **0.60** for ru.
13. Near-copy: the share of the question's tokens (titles and numbers exempt) covered, in order, by ONE fragment
    sentence is below **0.7222**.
14. Act identity: an amended act the question names must be named in the fragment text; an unquoted
    "<X> тўғрисидаги Қонун / қарор" must match the packet's act titles.
15. Transcript date: a question over a plenary transcript contains the session date (year, month -- as a number
    or a month name -- and a day of the sitting); for a multi-day sitting write the full range; every day written
    next to the month is one of `session.dates`, and a written range "a–b" spans no missing day.
18. Relative years: «жорий йил», «бу йил», «ўтган йил», «текущий год», «в этом / прошлом году» … only with the
    actual year written next to them.
16. Russian purity: in a ru question / answer no ў қ ғ ҳ outside «»; at most one Uzbek «» span; an answer's Uzbek
    «» span has at most 6 words and no figures.
17. Amending laws: over an amending-law fragment, an article / item reference must come with an amended act that the
    fragment text names (or a code).

## 5. The strict reader's rubric (7 criteria; GOOD only if all seven hold without doubt)

- c1 **supported**: every name, place, number, body and claim of the question and the answer is in the fragment text,
  not only in general knowledge. Nothing is reversed, narrowed or widened.
- c2 **complete and specific**: the answer fully answers the question. If the fragment lists several items,
  priorities or conditions, the question asks for all of them or names the one it wants. The answer is not a single
  vague word ("комитетом", "обязательны") where the question expects content.
- c3 **self-contained**: a reader who has never seen the fragment knows exactly which draft, law, article, village,
  body or event is meant. These fail:
  - "which article was excluded" without naming the amended law;
  - "who reviewed the draft" without naming the draft;
  - "residents" without naming the place;
  - a question built on a throwaway transcript line.
- c4 **act identity**: the act the question names is the fragment's own act (its `act` title, faithfully rendered),
  or for an amending law, the amended act and its article. It fails if the question:
  - names an act, decree or programme that the text only mentions;
  - uses the year's slogan instead of the act;
  - gives a garbled or wrong title;
  - uses the amending law's own item number instead of the amended provision;
  - misattributes the kind of document: a conclusion or a speech called a law, decision or "draft", or a transcript
    called a "session" document when it is a plenary sitting.
- c5 **paraphrased**: the question is not the fragment's sentence with one element blanked out, whether in the original
  language, a translation or a transliteration. It does not restate almost the whole fragment, leaving the answer to
  repeat one phrase.
- c6 **fit**: the interrogative matches the answer type and the rule's logic (threshold -> "up to what amount"; fee or
  manner -> "on what terms / is it paid"; term -> "within what period"). The question's condition is not inverted.
- c7 **language**: fluent, grammatical LANG with the correct legal terms (see the GLOSSARY) and pure script. In ru,
  act titles are rendered in Russian.

Verdicts: GOOD only if all seven hold. BAD if c1, c2, c3 or c4 fails in a way that makes the answer wrong, ambiguous
or unanswerable without the fragment. Otherwise BORDERLINE. Only GOOD is kept.

## 6. GLOSSARY (standard Russian renderings)

| Uzbek | Russian |
|---|---|
| тафтиш инстанцияси | ревизионная инстанция |
| назорат инстанцияси | надзорная инстанция |
| Солиқ қўмитаси | Налоговый комитет |
| баённома | протокол |
| Қонунчилик палатаси | Законодательная палата |
| Олий Мажлис Сенати | Сенат Олий Мажлиса |
| Сенатнинг ялпи мажлиси | пленарное заседание Сената |
| Вазирлар Маҳкамаси | Кабинет Министров |
| Маъмурий жавобгарлик тўғрисидаги кодекс | Кодекс об административной ответственности |
| Божхона кодекси | Таможенный кодекс |
| Солиқ кодекси | Налоговый кодекс |
| Жиноят кодекси | Уголовный кодекс |
| Фуқаролик процессуал кодекси | Гражданский процессуальный кодекс |
| базавий ҳисоблаш миқдори | базовая расчётная величина (БРВ) |
| маҳалла | махалля |
| ҳоким | хоким |
| қўмита хулосаси | заключение комитета |
| Сенат қарори | постановление Сената |
| парламент сўрови | парламентский запрос |
| сенатор сўрови | сенаторский запрос |
| депутат сўрови | депутатский запрос |
| давлат божи | государственная пошлина |

## 7. Working method

For each seed, in order:
1. Read the whole fragment (for multi, the anchor first, then every neighbour).
2. Find the fact a citizen or official would genuinely ask about.
3. Draft the question, then check it against §3, §4 and all seven rubric criteria. Check especially: the act name,
   self-containment, no leak, the question word, copy length and script.
4. For a multi seed, run the mandatory self-test of §3 (could the anchor alone, or the neighbour alone, answer it?).
   If one fragment suffices, write `skip`.
5. If any criterion is in doubt and you cannot fix it, write `skip`.
6. Write exactly one line per seed. Before you finish, check that the file is valid JSONL and that the seed ids
   appear in packet order.
