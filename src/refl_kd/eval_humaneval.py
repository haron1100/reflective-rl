import argparse, os, json, urllib.request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils.logging import set_verbosity_error
from peft import PeftModel
from tqdm import tqdm

HUMAN_EVAL_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
DEFAULT_PATH = "/kaggle/working/HumanEval.jsonl.gz"
DEFAULT_SAMPLES = "/kaggle/working/humaneval_samples.jsonl"

def ensure_humaneval(path: str = DEFAULT_PATH) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading HumanEval dataset to {path} ...")
    urllib.request.urlretrieve(HUMAN_EVAL_URL, path)
    print("Download complete.")
    return path

def gen(model, tok, prompt, max_new_tokens=320):
    out = model.generate(
        **tok(prompt, return_tensors="pt").to(model.device),
        do_sample=False,                 # deterministic for pass@1
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True,
    )
    return tok.decode(out.sequences[0], skip_special_tokens=True)

def main():
    # Quiet transformer verbosity + tokenizer threads
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    set_verbosity_error()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--student_adapter", type=str, default=None)
    ap.add_argument("--passk", type=int, default=1)
    ap.add_argument("--humaneval_path", type=str, default=DEFAULT_PATH)
    ap.add_argument("--samples_out", type=str, default=DEFAULT_SAMPLES)
    args = ap.parse_args()

    he_path = ensure_humaneval(args.humaneval_path)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, device_map="auto")
    if args.student_adapter and os.path.isdir(args.student_adapter):
        model = PeftModel.from_pretrained(model, args.student_adapter)
        print(f"Loaded student adapter from {args.student_adapter}")
    model.eval()

    try:
        from human_eval.data import read_problems
        from human_eval.evaluation import evaluate_functional_correctness
    except Exception as e:
        raise SystemExit("Install human-eval: pip install 'git+https://github.com/openai/human-eval'") from e

    problems = read_problems(he_path)
    items = list(problems.items())

    # Generate completions (pass@1) with a progress bar
    os.makedirs(os.path.dirname(args.samples_out), exist_ok=True)
    with open(args.samples_out, "w", encoding="utf-8") as f:
        for task_id, task in tqdm(items, desc="Generating HumanEval samples", ncols=100):
            code = gen(model, tok, task["prompt"], max_new_tokens=320)
            f.write(json.dumps({"task_id": task_id, "completion": code}) + "\n")

    print("Evaluating (this may take a minute)...")
    # IMPORTANT: your Kaggle install expects k to be a LIST
    results = evaluate_functional_correctness(
        sample_file=args.samples_out,
        problem_file=he_path,
        k=[args.passk],
        timeout=7.0,
    )
    # Extract the metric you asked for
    key = f"pass@{args.passk}"
    print(json.dumps(results, indent=2))
    print("pass@1:" if args.passk == 1 else key + ":", results.get(key))

if __name__ == "__main__":
    main()
