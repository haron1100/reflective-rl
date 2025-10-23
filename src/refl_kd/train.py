# src/refl_kd/train.py
import argparse, os, csv, time, math, random
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm, trange

from .utils import TrainCfg
from .data import load_mbpp
from .prompts import format_reflection_prompt, format_retry_prompt
from .evaluator import run_tests_simple
from .models import load_model_tokenizer, add_lora_adapters
from .kd import collect_teacher_topk, kd_kl_topk


# ---- Logits sanitizer to prevent NaN/Inf sampling errors (e.g., on MPS) ----
class SanitizeLogits(LogitsProcessor):
    def __init__(self, clamp: float = 1e3):
        super().__init__()
        self.clamp = clamp
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = torch.nan_to_num(scores, neginf=-self.clamp, posinf=self.clamp)
        return scores.clamp(min=-self.clamp, max=self.clamp)


@torch.inference_mode()
def gen_with_scores(model, tok, prompt: str, max_new_tokens: int, temperature: float):
    """
    Single-sample generator used for baseline (simple and clear).
    Returns:
      gen_text: decoded generated tail (no prompt)
      scores:   list[T] of logits [1, V] per step
      gen_ids:  [1, T] token ids of generated tail
    """
    gc = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=max(temperature, 1e-3),
        top_p=0.95,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        generation_config=gc,
        logits_processor=LogitsProcessorList([SanitizeLogits()]),
    )
    full_ids = out.sequences[0]
    prompt_len = inputs["input_ids"].shape[1]
    if full_ids.shape[0] <= prompt_len:
        return "", [], torch.empty((1, 0), dtype=torch.long, device=model.device)
    gen_ids = full_ids[prompt_len:].unsqueeze(0)
    # slice step logits for this (single) sample
    T = gen_ids.shape[1]
    scores_1 = [out.scores[t][0:1, :].contiguous() for t in range(T)]
    gen_text = tok.decode(gen_ids[0], skip_special_tokens=True)
    return gen_text, scores_1, gen_ids


@torch.inference_mode()
def gen_batch_texts(model, tok, prompts: List[str], max_new_tokens: int, temperature: float) -> List[str]:
    """
    Batched generation (no scores). Used for reflections (we don't need scores there).
    Returns list of decoded generated tails per prompt.
    """
    if not prompts:
        return []
    gc = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=max(temperature, 1e-3),
        top_p=0.95,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True,
        output_scores=False,
    )
    batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    attn = batch["attention_mask"]
    prompt_lens = attn.sum(dim=1)  # [B]
    out = model.generate(**batch, generation_config=gc, logits_processor=LogitsProcessorList([SanitizeLogits()]))
    seq = out.sequences  # [B, prompt_len + gen_len_max]
    texts = []
    for i in range(seq.size(0)):
        tail = seq[i, prompt_lens[i].item():]
        texts.append(tok.decode(tail, skip_special_tokens=True))
    return texts


@torch.inference_mode()
def gen_batch_with_scores(model, tok, prompts: List[str], max_new_tokens: int, temperature: float):
    """
    Batched generation that ALSO returns per-sample step logits and generated token ids.
    Returns:
      texts:   List[str] length B
      scores:  List[List[Tensor]] length B; each inner list has T_i tensors [1, V]
      gen_ids: List[Tensor] length B; each is [1, T_i]
    """
    if not prompts:
        return [], [], []
    gc = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=max(temperature, 1e-3),
        top_p=0.95,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        return_dict_in_generate=True,
        output_scores=True,
    )
    batch = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    attn = batch["attention_mask"]
    prompt_lens = attn.sum(dim=1)  # [B]
    out = model.generate(**batch, generation_config=gc, logits_processor=LogitsProcessorList([SanitizeLogits()]))

    seq = out.sequences
    B = seq.size(0)
    texts: List[str] = []
    per_sample_scores: List[List[torch.Tensor]] = []
    per_sample_gen_ids: List[torch.Tensor] = []

    # compute generated lengths per sample
    total_len = seq.size(1)
    gen_lens = [int(total_len - prompt_lens[i].item()) for i in range(B)]

    for i in range(B):
        start = prompt_lens[i].item()
        gen_ids_i = seq[i, start:].unsqueeze(0)  # [1, T_i]
        T_i = gen_lens[i]
        scores_i = []
        for t in range(T_i):
            # out.scores[t] is [B, V]; take row i, keep shape [1, V]
            scores_i.append(out.scores[t][i:i+1, :].contiguous())
        text_i = tok.decode(gen_ids_i[0], skip_special_tokens=True)
        texts.append(text_i)
        per_sample_scores.append(scores_i)
        per_sample_gen_ids.append(gen_ids_i)
    return texts, per_sample_scores, per_sample_gen_ids


def eval_code(code: str, tests):
    # Simple runner (not sandboxed). Prefer EvalPlus for robust/safe execution.
    score, raw = run_tests_simple(code, tests)
    return score, raw


@torch.inference_mode()
def mini_eval_student(model, tok, problems_subset, max_new_tokens=320):
    """
    Quick validation: greedy student (no reflection) on a small held-out subset.
    Returns pass rate in [0,1].
    """
    if not problems_subset:
        return 0.0
    model.set_adapter("default")
    hits = 0
    for pb in problems_subset:
        base_prompt = pb.prompt + "\n# Write the function above."
        out = model.generate(
            **tok(base_prompt, return_tensors="pt").to(model.device),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.pad_token_id,
        )
        code = tok.decode(out.sequences[0], skip_special_tokens=True)
        R, _ = eval_code(code, pb.tests)
        hits += 1 if R > 0 else 0
    return hits / len(problems_subset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--dataset", type=str, default="mbpp")
    ap.add_argument("--max_problems", type=int, default=200)
    ap.add_argument("--k_reflections", type=int, default=2)
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--device", type=str, default="cuda", help="cuda | mps | cpu")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bettertransformer", action="store_true", help="try FasterTransformer path if available")
    # instrumentation options
    ap.add_argument("--metrics_csv", type=str, default="./checkpoints/train_metrics.csv")
    ap.add_argument("--eval_every", type=int, default=10, help="mini-eval frequency (problems)")
    ap.add_argument("--val_size", type=int, default=20, help="held-out MBPP problems for quick validation")
    ap.add_argument("--show_inner_bar", action="store_true", help="show a nested bar for K reflections")
    args = ap.parse_args()

    # QoL: fewer tokenizer warnings & graceful MPS fallback
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    # Seed (best-effort)
    random.seed(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except Exception:
        pass
    torch.manual_seed(args.seed)

    cfg = TrainCfg(
        model_name=args.model,
        max_problems=args.max_problems,
        k_reflections=args.k_reflections,
        save_dir=args.save_dir,
        device=args.device,
    )
    os.makedirs(cfg.save_dir, exist_ok=True)

    # ---- Data ----
    if args.dataset != "mbpp":
        raise ValueError("Only 'mbpp' is supported in this minimal repo.")
    # Load a bit extra so we can carve a tiny validation slice
    problems_all = load_mbpp(limit=cfg.max_problems + max(args.val_size, 0))
    problems = problems_all[: cfg.max_problems]
    val_subset = problems_all[cfg.max_problems : cfg.max_problems + args.val_size] if args.val_size > 0 else []

    # ---- Model & adapters ----
    model, tok = load_model_tokenizer(cfg.model_name, device=cfg.device)
    # Try a faster inference path (safe fallback)
    if args.bettertransformer:
        try:
            model = model.to_bettertransformer()
            print("Using BetterTransformer fast path.")
        except Exception:
            pass

    model = add_lora_adapters(model)  # default adapter == student; also adds 'reflector'

    # Separate optimizers per adapter
    model.set_adapter("default")  # student
    opt_student = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr_student)
    model.set_adapter("reflector")
    opt_refl = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr_reflector)

    # ---- Metrics CSV ----
    os.makedirs(os.path.dirname(args.metrics_csv), exist_ok=True)
    write_header = not os.path.exists(args.metrics_csv)
    csv_f = open(args.metrics_csv, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if write_header:
        csv_w.writerow(
            ["epoch", "step", "pid", "R0", "R_best", "A_best", "KD", "refl_loss", "kd_loss", "sft_loss",
             "refl_len", "code_len", "secs"]
        )
        csv_f.flush()

    # Running stats
    kd_yes = 0
    refl_help = 0
    sum_A = 0.0
    sum_R0 = 0.0
    sum_Rbest = 0.0
    total_seen = 0
    last_val = None

    # ---- Training loop ----
    for epoch in range(args.epochs):
        if args.shuffle:
            random.shuffle(problems)

        pbar = tqdm(enumerate(problems), total=len(problems),
                    desc=f"Train e{epoch+1}/{args.epochs}")
        for idx, pb in pbar:
            t0 = time.time()

            # 1) Baseline attempt (student, no reflection)
            model.set_adapter("default")
            base_prompt = pb.prompt + "\n# Write the function above."
            base_code, _, _ = gen_with_scores(model, tok, base_prompt, cfg.max_new_tokens_code, cfg.temp_code)
            R0, _ = eval_code(base_code, pb.tests)

            # 2) K reflections (batched) + K retries (batched with scores)
            #    This is the main speedup vs. generating K times serially.
            model.set_adapter("reflector")
            refl_prompt = format_reflection_prompt(pb.prompt, R0)
            refl_prompts = [refl_prompt] * cfg.k_reflections
            reflections = gen_batch_texts(model, tok, refl_prompts, cfg.max_new_tokens_refl, cfg.temp_refl)

            model.set_adapter("default")
            retry_prompts = [format_retry_prompt(pb.prompt, r) for r in reflections]
            code_texts, teacher_scores_batch, gen_ids_batch = gen_batch_with_scores(
                model, tok, retry_prompts, cfg.max_new_tokens_code, cfg.temp_code
            )

            cand = []
            for rtxt, ctext, t_scores, g_ids in zip(reflections, code_texts, teacher_scores_batch, gen_ids_batch):
                R_i, _ = eval_code(ctext, pb.tests)
                A_i = R_i - R0
                cand.append((A_i, rtxt, ctext, t_scores, g_ids))

            # 3) RL on reflections (advantage-weighted NLL, labels masked to reflection tokens)
            model.set_adapter("reflector")
            opt_refl.zero_grad(set_to_none=True)
            loss_refl_val = None
            for (A_i, reflection, _code_text, _scores, _gen_ids) in cand:
                r_prompt = format_reflection_prompt(pb.prompt, R0)
                p_ids = tok(r_prompt, return_tensors="pt").to(model.device)["input_ids"]
                r_ids = tok(reflection, return_tensors="pt").to(model.device)["input_ids"]
                if r_ids.shape[1] == 0:
                    continue
                inp = torch.cat([p_ids, r_ids], dim=1)
                attn = torch.ones_like(inp)
                labels = inp.clone()
                labels[:, :p_ids.shape[1]] = -100
                out = model(input_ids=inp, attention_mask=attn, labels=labels)
                nll = out.loss
                w = max(min(float(A_i), 1.0), -1.0)  # clip
                loss_refl_val = nll * (-w) if loss_refl_val is None else loss_refl_val + nll * (-w)
            if loss_refl_val is not None:
                (loss_refl_val / max(len(cand), 1)).backward()
                opt_refl.step()

            # 4) KD distillation (teacher-with-reflection → student-without), gated by advantage
            best = max(cand, key=lambda x: x[0]) if cand else None
            did_kd = False
            kd_loss_val = None
            sft_loss_val = None
            if best and best[0] >= cfg.tau_gate:
                _A_best, _reflection, _code_text, teacher_scores, gen_ids = best
                if gen_ids is not None and gen_ids.numel() > 0 and len(teacher_scores) > 0:
                    teacher_topk = collect_teacher_topk(teacher_scores, topk=cfg.topk_kd)

                    model.set_adapter("default")
                    opt_student.zero_grad(set_to_none=True)

                    x_ids = tok(pb.prompt + "\n# Write the function above.", return_tensors="pt").to(model.device)
                    y_ids = gen_ids.to(model.device)

                    concat_ids = torch.cat([x_ids["input_ids"], y_ids], dim=1)
                    attn = torch.ones_like(concat_ids)
                    out = model(input_ids=concat_ids, attention_mask=attn, labels=concat_ids)
                    logits = out.logits[0, -y_ids.shape[1]:, :]

                    kd_loss = kd_kl_topk(logits.float(), teacher_topk)
                    sft_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_ids[0], ignore_index=-100)
                    total_loss = cfg.kd_weight * kd_loss + cfg.sft_weight * sft_loss
                    total_loss.backward()
                    opt_student.step()
                    did_kd = True
                    kd_loss_val = float(kd_loss.detach())
                    sft_loss_val = float(sft_loss.detach())

            # ---- Update running stats & log row ----
            total_seen += 1
            A_best = best[0] if best else 0.0
            R_best_num = max([c[0] + R0 for c in cand], default=R0)  # R_best numeric = R0 + max A
            if A_best > 0:
                refl_help += 1
            if did_kd:
                kd_yes += 1
            sum_A += A_best
            sum_R0 += R0
            sum_Rbest += R_best_num

            # CSV row
            refl_len = len(cand[0][1].split()) if cand else 0
            code_len = len((best[2] if best else base_code).split())
            secs = time.time() - t0
            csv_w.writerow([
                epoch, idx, pb.pid, f"{R0:.2f}", f"{R_best_num:.2f}", f"{A_best:.2f}",
                1 if did_kd else 0,
                f"{float(loss_refl_val.detach()):.4f}" if loss_refl_val is not None else "",
                f"{kd_loss_val:.4f}" if kd_loss_val is not None else "",
                f"{sft_loss_val:.4f}" if sft_loss_val is not None else "",
                refl_len, code_len, f"{secs:.2f}",
            ])
            csv_f.flush()

            # Periodic mini-eval on held-out set
            if args.eval_every > 0 and args.val_size > 0 and (idx + 1) % args.eval_every == 0:
                last_val = mini_eval_student(model, tok, val_subset, max_new_tokens=cfg.max_new_tokens_code)

            # Update main bar postfix
            avg_A = sum_A / max(1, total_seen)
            help_rate = 100.0 * refl_help / max(1, total_seen)
            kd_rate = 100.0 * kd_yes / max(1, total_seen)
            avg_R0 = sum_R0 / max(1, total_seen)
            avg_Rbest = sum_Rbest / max(1, total_seen)
            pbar.set_postfix_str(
                f"Aavg={avg_A:.2f} Help%={help_rate:.1f} KD%={kd_rate:.1f} "
                f"R0avg={avg_R0:.2f} Rbestavg={avg_Rbest:.2f}"
                + (f" Val@{args.val_size}={last_val:.2f}" if last_val is not None else "")
            )

    # Save adapters
    model.set_adapter("default")
    model.save_pretrained(os.path.join(cfg.save_dir, "student_adapter"))
    model.set_adapter("reflector")
    model.save_pretrained(os.path.join(cfg.save_dir, "reflector_adapter"))
    print("Saved adapters to", cfg.save_dir)

    # Close CSV
    csv_f.close()


if __name__ == "__main__":
    main()
