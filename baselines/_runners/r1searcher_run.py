"""R1-Searcher baseline, step 2 (Spark GPU): the released checkpoint, NO fine-tuning, as a retrieval agent over our
base chunks; per-question retrieval on the val split and, with the identical frozen config, the test split.

  HF_HOME=~/abdurakhmon/.cache/huggingface PYTHONPATH=<grl checkout> \\
    python r1searcher_run.py --bundle <dir from r1searcher_prep.py> --out r1searcher-val-k12.jsonl

Mirrors RUCAIBox/R1-Searcher evaluation/eval_search_loacl.py (README command: --prompt_type v0 --temp 0.0):
prompt base_prompt_v0 verbatim (raw text, no chat template), stop strings ["<|im_end|>", "<|endoftext|>",
"<|end_of_query|>", "</answer>"], temperature 0 (greedy; top_p 0.95 has no effect), max_tokens 512 per generation
round, topk 5 docs per search, query clean-up replace('"','') / ("'",'') / ('\\t',' ') / ('...',''), document block
"<|end_of_query|>\\n\\n<|begin_of_documents|>\\n(1)doc\\n(2)doc\\n...<|end_of_documents|>\\n\\n" appended to
prompt + generated_text.strip(); round k == 9 ends the question as "many_retrieve" (so <= 9 searches).
Engine: vLLM (official; --engine hf = transformers fallback). Deviations (recorded in the meta file): a fixed 18 GiB
vLLM KV cache instead of gpu_memory_utilization 0.95 (shared unified memory), and the search backend = our bge-m3 dense
retriever over the 16,643 base chunks instead of bge-large-en-v1.5 over KILT Wikipedia 100-word passages."""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

CKPTS = {"qwen": ("XXsongLALA/Qwen-2.5-7B-base-RAG-RL", "bc1b424ee27133445a6430df9c93d46a9453a6eb"),
         "llama": ("XXsongLALA/Llama-3.1-8B-instruct-RAG-RL", "f1ee0b61809536a7b2b6984b98d88a737d402173")}
ENCODER, ENC_REV = "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"

PROMPT_V0 = """The User asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process in the mind and then provides the User with the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, the Assistant can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>". **A query must involve only a single triple**.
Then, the system will provide the Assistant with helpful information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>".\n\nUser:{question}\nAssistant: <think>"""
# OpenRLHF-RAG/openrlhf/datasets/prompts_dataset.py, apply_chat_template branch (the Llama training scripts pass
# --apply_chat_template): system prompt + user question through the tokenizer's chat template, then "<think>".
SYS_TRAIN = """You are a helpful assistant.
Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer.
The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>".
During the thinking process, **you can perform searching for uncertain knowledge** if necessary with the format of "<|begin_of_query|> search query (only list keywords, such as **"keyword_1 keyword_2 ..."**)<|end_of_query|>". **A query must involve only a single triple**.
Then, the search system will provide you with the retrieval information with the format of "<|begin_of_documents|> ...search results... <|end_of_documents|>"."""
# eval_search_loacl.py, the commented-out chat variant of process_text (sys_prompt_wo_doc_token).
SYS_EVAL_CHAT = """You are a helpful assistant. Given a question, you should answer it by first thinking about the reasoning process in the mind and then providing the final answer. The output format of reasoning process and final answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "<think> reasoning process here </think>\n\n<answer> final answer here </answer>". During the thinking process, you can perform searching for uncertain knowledge if necessary with the format of "<|begin_of_query|> search query (only keywords) here <|end_of_query|>"."""


def build_prompt(question: str, variant: str, tok=None) -> str:
    if variant == "v0":
        return PROMPT_V0.format(question=question)
    sys_p = {"chat_train": SYS_TRAIN, "chat_eval": SYS_EVAL_CHAT}[variant]
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": question}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "<think>"
STOPS = ["<|im_end|>", "<|endoftext|>", "<|end_of_query|>", "</answer>"]
MAX_TOKENS, TOPK, MAX_ROUNDS, K = 512, 5, 9, 12


class Retriever:
    """bge-m3 dense top-k over the base chunks; query encoded like graphrag.retrieval (query_internal, no prefix)."""

    def __init__(self, bundle: Path):
        from sentence_transformers import SentenceTransformer
        self.rows = [json.loads(l) for l in open(bundle / "corpus.jsonl", encoding="utf-8")]
        self.vecs = torch.from_numpy(np.load(bundle / "vecs.npy")).cuda()
        self.enc = SentenceTransformer(ENCODER, device="cuda", revision=ENC_REV)
        self.enc.max_seq_length = 512

    def search(self, queries: list[str], k: int) -> list[list[int]]:
        from graphrag.data.normalize import query_internal
        q = self.enc.encode([query_internal(t) for t in queries], normalize_embeddings=True, convert_to_tensor=True)
        return (q.float() @ self.vecs.T).topk(k, dim=1).indices.tolist()

    def doc(self, i: int) -> str:     # KILT-style "title text" passage, on one line like the official corpus
        r = self.rows[i]
        return " ".join(f"{r['title']} {r['text']}".split())


def cut(text: str) -> tuple[str, str | None]:
    """vLLM stop semantics: text up to the earliest stop string (excluded) and that string, else (text, None)."""
    hits = [(text.find(s), s) for s in STOPS if s in text]
    if not hits:
        return text, None
    p, s = min(hits)
    return text[:p], s


@torch.no_grad()
def _gen_batch(model, tok, seqs: list[list[int]]) -> list[tuple[str, str | None]]:
    """Greedy-generate one left-padded batch; on CUDA OOM split it in half (results unchanged up to bf16 numerics)."""
    n = max(len(e) for e in seqs)
    ids = torch.full((len(seqs), n), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), n), dtype=torch.long)
    for r, e in enumerate(seqs):                          # left padding
        ids[r, n - len(e):] = torch.tensor(e)
        att[r, n - len(e):] = 1
    try:
        g = model.generate(input_ids=ids.cuda(), attention_mask=att.cuda(), max_new_tokens=MAX_TOKENS,
                           do_sample=False, stop_strings=STOPS, tokenizer=tok, pad_token_id=tok.pad_token_id,
                           eos_token_id=tok.convert_tokens_to_ids("<|endoftext|>"))
    except torch.OutOfMemoryError:
        if len(seqs) == 1:
            raise
        del ids, att
        torch.cuda.empty_cache()
        h = len(seqs) // 2
        return _gen_batch(model, tok, seqs[:h]) + _gen_batch(model, tok, seqs[h:])
    out = [cut(tok.decode(g[r, n:], skip_special_tokens=False)) for r in range(len(seqs))]
    del g
    torch.cuda.empty_cache()
    return out


def generate(model, tok, prompts: list[str], max_batch: int, max_pad_tokens: int) -> list[tuple[str, str | None]]:
    enc = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: -len(enc[i]))
    out: list = [None] * len(prompts)
    i = 0
    while i < len(order):
        L = len(enc[order[i]]) + MAX_TOKENS
        idx = order[i:i + max(1, min(max_batch, max_pad_tokens // L))]
        i += len(idx)
        for j, o in zip(idx, _gen_batch(model, tok, [enc[j] for j in idx])):
            out[j] = o
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--questions", default="val.jsonl", help="val.jsonl or test.jsonl in the bundle")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--max-pad-tokens", type=int, default=64000)
    ap.add_argument("--mem-gb", type=float, default=36.0, help="hard cap on this process's CUDA allocations")
    ap.add_argument("--engine", choices=["vllm", "hf"], default="vllm", help="vllm = the official engine")
    ap.add_argument("--ckpt", choices=sorted(CKPTS), default="qwen")
    ap.add_argument("--prompt", choices=["v0", "chat_train", "chat_eval"], default="v0")
    ap.add_argument("--kv-cache-gib", type=float, default=18.0, help="vLLM KV cache size (official: "
                    "gpu_memory_utilization 0.95; the Spark's unified memory is shared, so a fixed ~18 GiB KV cache)")
    a = ap.parse_args()
    t0 = time.time()
    if a.engine == "hf":
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, a.mem_gb * 2**30 / total))
    bundle = Path(a.bundle)
    qs = [json.loads(l) for l in open(bundle / a.questions, encoding="utf-8")]
    qs = qs[:a.limit] if a.limit else qs
    ret = Retriever(bundle)
    MODEL, REVISION = CKPTS[a.ckpt]
    from transformers import AutoTokenizer
    ptok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    if a.engine == "vllm":                      # official: LLM(...) + SamplingParams(temp, top_p=0.95, 512, stop)
        from vllm import LLM, SamplingParams
        llm = LLM(model=MODEL, revision=REVISION, tensor_parallel_size=1,
                  gpu_memory_utilization=0.30, kv_cache_memory_bytes=int(a.kv_cache_gib * 2**30),
                  trust_remote_code=True)
        sp = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=MAX_TOKENS, stop=STOPS)

        def gen(prompts):
            return [(o.outputs[0].text, o.outputs[0].stop_reason) for o in llm.generate(prompts, sp)]
    else:
        from transformers import AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = "<|endoftext|>"
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16).cuda().eval()

        def gen(prompts):
            return generate(model, tok, prompts, a.max_batch, a.max_pad_tokens)

    st = [{"prompt": build_prompt(q["q"], a.prompt, ptok), "store": "", "docs": [], "queries": [], "end": None,
           "pred": None} for q in qs]
    state, k0, spent = Path(a.out + ".state.json"), 0, 0.0      # per-round checkpoint, resumed after a restart
    if state.exists():
        z = json.loads(state.read_text(encoding="utf-8"))
        st, k0, spent = z["st"], z["next_round"], z["elapsed_s"]
        print(f"resumed at round {k0} ({spent:.0f}s spent before)", flush=True)
    t0 -= spent
    for k in range(k0, MAX_ROUNDS + 1):
        act = [i for i, s in enumerate(st) if s["end"] is None]
        if not act:
            break
        if k == MAX_ROUNDS:                     # official: round 9's output is discarded -> "many_retrieve"
            for i in act:
                st[i]["end"] = "many_retrieve"
            break
        outs = gen([st[i]["prompt"] for i in act])
        searching = []
        for i, (text, stop) in zip(act, outs):
            s = st[i]
            if "<answer>" in text and stop == "</answer>":
                s["pred"] = text.split("<answer>")[-1]
                s["store"] += text + "</answer>"
                s["end"] = "finished"
            elif "<|begin_of_query|>" in text and stop == "<|end_of_query|>":
                q = text.split("<|begin_of_query|>")[-1]
                q = q.replace('"', "").replace("'", "").replace("\t", " ").replace("...", "")
                if q:
                    s["prompt"] += text.strip()
                    s["store"] += text.strip()
                    s["queries"].append(q)
                    searching.append(i)
                else:
                    s["store"] += text.strip()
                    s["end"] = "query_inst_error"
            else:
                s["store"] += text
                s["end"] = "shot_down"
        if searching:
            hits = ret.search([st[i]["queries"][-1] for i in searching], TOPK)
            for i, h in zip(searching, hits):
                docs = [re.sub(r"^\d+\s+", "", ret.doc(d)) for d in h]
                block = "".join(f"({j + 1}){t}\n" for j, t in enumerate(docs))
                add = "<|end_of_query|>\n\n" + "<|begin_of_documents|>\n" + block + "<|end_of_documents|>\n\n"
                st[i]["prompt"] += add
                st[i]["store"] += add
                st[i]["docs"].extend(h)
        n_end = sum(s["end"] is not None for s in st)
        state.write_text(json.dumps({"st": st, "next_round": k + 1, "elapsed_s": time.time() - t0},
                                    ensure_ascii=False), encoding="utf-8")
        print(f"round {k}: {len(act)} active, {len(searching)} searched, {n_end}/{len(st)} ended, "
              f"{time.time() - t0:.0f}s", flush=True)

    out = Path(a.out)
    with open(out, "w", encoding="utf-8") as f, open(out.with_suffix(".traces.jsonl"), "w", encoding="utf-8") as ft:
        for q, s in zip(qs, st):
            seen, first = set(), []
            for d in s["docs"]:
                if d not in seen:
                    seen.add(d)
                    first.append(d)
            first = first[:K]
            ids = [ret.rows[d]["id"] for d in first]
            gold = set(q["C_ref"])
            f.write(json.dumps({"id": q["id"], "retrieved": ids, "recall12": len(gold & set(ids)) / len(gold),
                                "tokens12": int(sum(ret.rows[d]["tokens"] for d in first)),
                                "n_search": len(s["queries"]), "answered": s["end"] == "finished",
                                "queries": s["queries"]}, ensure_ascii=False) + "\n")
            ft.write(json.dumps({"id": q["id"], "lang": q["lang"], "stop_reason_final": s["end"], "pred_ans": s["pred"],
                                 "gen_text_store": s["store"]}, ensure_ascii=False) + "\n")
    print(json.dumps({"runtime_s": round(time.time() - t0, 1), "n": len(st)}))


if __name__ == "__main__":
    main()
