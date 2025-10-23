import argparse, os, json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def gen(model, tok, prompt, max_new_tokens=320):
    out = model.generate(
        **tok(prompt, return_tensors="pt").to(model.device),
        do_sample=False, max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True
    )
    return tok.decode(out.sequences[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--student_adapter", type=str, default=None)
    ap.add_argument("--passk", type=int, default=1)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map="auto")

    if args.student_adapter and os.path.isdir(args.student_adapter):
        model = PeftModel.from_pretrained(model, args.student_adapter)
        print(f"Loaded student adapter from {args.student_adapter}")

    try:
        from human_eval.data import read_problems
        from human_eval.evaluation import evaluate_functional_correctness
    except Exception as e:
        print("Install human-eval from https://github.com/openai/human-eval or use EvalPlus.")
        return

    problems = read_problems()
    samples = []
    for task_id, task in problems.items():
        prompt = task["prompt"]
        code = gen(model, tok, prompt, max_new_tokens=320)
        samples.append({"task_id": task_id, "completion": code})

    results = evaluate_functional_correctness(samples=samples, problem_file=None, timeout=7.0)
    print(json.dumps(results, indent=2))
    print("pass@1:", results.get("pass@1"))

if __name__ == "__main__":
    main()
