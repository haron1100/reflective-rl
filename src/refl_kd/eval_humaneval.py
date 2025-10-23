import argparse, os, json, csv, urllib.request, time, math, inspect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.logging import set_verbosity_error
from peft import PeftModel
from tqdm import tqdm

HUMAN_EVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
DEFAULT_PATH = "/kaggle/working/HumanEval.jsonl.gz"
DEFAULT_SAMPLES = "/kaggle/working/humaneval_samples.jsonl"
DEFAULT_RESULTS = "/kaggle/working/humaneval_per_task.csv"

def ensure_humaneval(path: str = DEFAULT_PATH) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading HumanEval dataset to {path} ...")
    urllib.request.urlretrieve(HUMAN_EVAL_URL, path)
    print("Download complete.")
    return path

@torch.inference_mode()
def gen_batch(model, tok, prompts, max_new_tokens=320):
    """
    Greedy generation for a batch of prompts.
    Returns list[str] of *generated tails* (no prompts included).
    """
    inputs = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    attn = inputs["attention_mask"]
    prompt_lens = attn.sum(dim=1)  # [B]
    out = model.generate(
        **inputs,
        do_sample=False,               # pass@1 = deterministic
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True,
        use_cache=True,
    )
    seq = out.sequences  # [B, prompt_len + gen_len_max]
    B = seq.size(0)
    tails = []
    for i in range(B):
        tail = seq[i, prompt_lens[i].item():]
        tails.append(tok.decode(tail, skip_special_tokens=True))
    return tails

def batched(iterable, bs):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) == bs:
            yield buf
            buf = []
    if buf:
        yield buf

def main():
    # Quiet transformer verbosity + tokenizers threads
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_verbosity_error()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--student_adapter", type=str, default=None)
    ap.add_argument("--humaneval_path", type=str, default=DEFAULT_PATH)
    ap.add_argument("--samples_out", type=str, default=DEFAULT_SAMPLES)
    ap.add_argument("--results_csv", type=str, default=DEFAULT_RESULTS)
    ap.add_argument("--gen_batch_size", type=int, default=8, help="GPU batch size for generation")
    ap.add_argument("--eval_workers", type=int, default=max(os.cpu_count() or 2, 2),
                    help="CPU workers for running tests if supported by HumanEval")
    ap.add_argument("--max_new_tokens", type=int, default=320)
    args = ap.parse_args()

    he_path = ensure_humaneval(args.humaneval_path)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    # Use fp16 on CUDA, fp32 otherwise
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    if args.student_adapter and os.path.isdir(args.student_adapter):
        model = PeftModel.from_pretrained(model, args.student_adapter)
        print(f"Loaded student adapter from {args.student_adapter}")
    model.eval()

    # (Optional) slightly faster inference path if available
    try:
        model = model.to_bettertransformer()
        print("Using BetterTransformer fast path.")
    except Exception:
        pass

    # HumanEval helpers
    try:
        from human_eval.data import read_problems
        from human_eval.evaluation import evaluate_functional_correctness, check_correctness
    except Exception as e:
        raise SystemExit("Install human-eval: pip install 'git+https://github.com/openai/human-eval'") from e

    problems = read_problems(he_path)
    items = list(problems.items())
    task_ids = [tid for tid, _ in items]
    prompts = [task["prompt"] for _, task in items]

    # ---- 1) Generate completions in batches on GPU
    os.makedirs(os.path.dirname(args.samples_out), exist_ok=True)
    with open(args.samples_out, "w", encoding="utf-8") as fout:
        pbar = tqdm(batched(list(zip(task_ids, prompts)), args.gen_batch_size),
                    total=math.ceil(len(prompts) / args.gen_batch_size),
                    desc="Generating HumanEval samples", ncols=100)
        for chunk in pbar:
            tids, prmpts = zip(*chunk)
            tails = gen_batch(model, tok, prmpts, max_new_tokens=args.max_new_tokens)
            for tid, comp in zip(tids, tails):
                fout.write(json.dumps({"task_id": tid, "completion": comp}) + "\n")

    # ---- 2) Try the parallel harness first (fast), else fall back to per-task loop
    print(f"Evaluating with up to {args.eval_workers} workers...")
    use_parallel = False
    sig = inspect.signature(evaluate_functional_correctness)
    if "n_workers" in sig.parameters:
        try:
            t0 = time.time()
            results = evaluate_functional_correctness(
                sample_file=args.samples_out,
                problem_file=he_path,
                k=[1],
                timeout=7.0,
                n_workers=args.eval_workers,
            )
            use_parallel = True
        except Exception:
            use_parallel = False

    total, solved = 0, 0
    if not use_parallel:
        # Fallback: manual per-task evaluation (slower)
        os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
        with open(args.samples_out, "r", encoding="utf-8") as fin, \
             open(args.results_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["task_id", "passed", "elapsed_sec"])
            for line in tqdm(fin, total=len(items), ncols=100, desc="Running test suites"):
                rec = json.loads(line)
                task_id = rec["task_id"]
                completion = rec["completion"]
                problem = problems[task_id]

                t0 = time.time()
                res = check_correctness(problem, completion, timeout=7.0)
                passed = bool(res.get("passed", False)) if isinstance(res, dict) else bool(res)
                elapsed = time.time() - t0

                writer.writerow([task_id, int(passed), f"{elapsed:.2f}"])
                total += 1
                solved += int(passed)

        pass_at_1 = solved / max(1, total)
        print(json.dumps({
            "tasks_total": total,
            "tasks_solved": solved,
            "pass@1": pass_at_1
        }, indent=2))
        print(f"Summary: solved {solved}/{total} | pass@1 = {pass_at_1:.3f}")
        print(f"Per-task CSV saved to: {args.results_csv}")
        print(f"Samples JSONL saved to: {args.samples_out}")
    else:
        # Parallel harness returns a dict like {"pass@1": x, ...}
        pass_at_1 = results.get("pass@1", None)
        print(json.dumps(results, indent=2))
        print(f"Summary: solved {int(round(pass_at_1*len(items)) if pass_at_1 is not None else 0)}/{len(items)} | pass@1 = {pass_at_1:.3f}" if pass_at_1 is not None else "No pass@1")
        # Also write the per-task CSV by reusing check_correctness quickly (serial), but keep the batch gen speed-up.
        os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
        with open(args.samples_out, "r", encoding="utf-8") as fin, \
             open(args.results_csv, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerow(["task_id", "passed", "elapsed_sec"])
            for line in tqdm(fin, total=len(items), ncols=100, desc="Writing per-task CSV"):
                rec = json.loads(line)
                task_id = rec["task_id"]
                completion = rec["completion"]
                problem = problems[task_id]
                # quick check to mark pass/fail (single-thread)
                t0 = time.time()
                res = check_correctness(problem, completion, timeout=7.0)
                passed = bool(res.get("passed", False)) if isinstance(res, dict) else bool(res)
                elapsed = time.time() - t0
                writer.writerow([task_id, int(passed), f"{elapsed:.2f}"])
        print(f"Per-task CSV saved to: {args.results_csv}")
        print(f"Samples JSONL saved to: {args.samples_out}")

if __name__ == "__main__":
    main()
