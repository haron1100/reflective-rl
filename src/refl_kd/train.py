# src/refl_kd/train.py
import argparse
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList
from tqdm import tqdm

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
        # Replace NaN/Inf and clamp extreme values to keep softmax stable
        scores = torch.nan_to_num(scores, neginf=-self.clamp, posinf=self.clamp)
        return scores.clamp(min=-self.clamp, max=self.clamp)


def gen_with_scores(model, tok, prompt: str, max_new_tokens: int, temperature: float):
    """
    Generate and return:
      - gen_text: decoded *generated* tail (no prompt)
      - scores:   list of per-step logits for each generated token
      - gen_ids:  [1, T] token ids of *generated* tail
    """
    gc = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=True if temperature > 0 else False,
        temperature=max(temperature, 1e-3),  # avoid degenerate 0 temp
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
        logits_processor=LogitsProcessorList([SanitizeLogits()])
    )

    full_ids = out.sequences[0]                       # [prompt_len + T]
    prompt_len = inputs["input_ids"].shape[1]
    if full_ids.shape[0] <= prompt_len:
        # No generation; return empty tail
        return "", [], torch.empty((1, 0), dtype=torch.long, device=model.device)

    gen_ids = full_ids[prompt_len:].unsqueeze(0)      # [1, T]
    gen_text = tok.decode(gen_ids[0], skip_special_tokens=True)
    return gen_text, out.scores, gen_ids


def eval_code(code: str, tests):
    # Simple runner (not sandboxed). Prefer EvalPlus for robust/safe execution.
    score, raw = run_tests_simple(code, tests)
    return score, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--dataset", type=str, default="mbpp")
    ap.add_argument("--max_problems", type=int, default=200)
    ap.add_argument("--k_reflections", type=int, default=2)
    ap.add_argument("--save_dir", type=str, default="./checkpoints")
    ap.add_argument("--device", type=str, default="cuda", help="cuda | mps | cpu")
    args = ap.parse_args()

    # QoL for Macs: fewer tokenizer warnings & graceful MPS fallback
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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
    problems = load_mbpp(limit=cfg.max_problems)

    # ---- Model & adapters ----
    model, tok = load_model_tokenizer(cfg.model_name, device=cfg.device)
    model = add_lora_adapters(model)  # default adapter == student; also adds 'reflector'
    # (Optional, but sometimes helps on MPS)
    # model.to(args.device)

    # Separate optimizers per adapter
    model.set_adapter("default")  # student
    opt_student = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr_student)

    model.set_adapter("reflector")
    opt_refl = AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr_reflector)

    # ---- Training loop ----
    pbar = tqdm(problems, desc="Train")
    for pb in pbar:
        # 1) Baseline attempt (student, no reflection)
        model.set_adapter("default")
        base_prompt = pb.prompt + "\n# Write the function above."
        base_code, _, _ = gen_with_scores(model, tok, base_prompt, cfg.max_new_tokens_code, cfg.temp_code)
        R0, _ = eval_code(base_code, pb.tests)

        # 2) Sample K reflections & retries (teacher runs with reflection in context)
        cand = []  # tuples: (A_i, reflection_text, code_text, teacher_scores, gen_ids)
        for _ in range(cfg.k_reflections):
            # Reflection (reflector adapter)
            model.set_adapter("reflector")
            refl_prompt = format_reflection_prompt(pb.prompt, R0)
            reflection, _, _ = gen_with_scores(model, tok, refl_prompt, cfg.max_new_tokens_refl, cfg.temp_refl)

            # Retry with reflection (teacher = same base weights, student adapter for generation)
            model.set_adapter("default")
            retry_prompt = format_retry_prompt(pb.prompt, reflection)
            code_text, teacher_scores, gen_ids = gen_with_scores(
                model, tok, retry_prompt, cfg.max_new_tokens_code, cfg.temp_code
            )

            # Evaluate
            R_i, _ = eval_code(code_text, pb.tests)
            A_i = R_i - R0
            cand.append((A_i, reflection, code_text, teacher_scores, gen_ids))

        # 3) RL on reflections (advantage-weighted NLL, labels masked to reflection tokens)
        model.set_adapter("reflector")
        opt_refl.zero_grad(set_to_none=True)
        loss_refl = None

        for (A_i, reflection, _code_text, _scores, _gen_ids) in cand:
            refl_prompt = format_reflection_prompt(pb.prompt, R0)
            p_ids = tok(refl_prompt, return_tensors="pt").to(model.device)["input_ids"]    # prompt tokens
            r_ids = tok(reflection, return_tensors="pt").to(model.device)["input_ids"]    # reflection tokens
            if r_ids.shape[1] == 0:
                continue  # nothing to learn from empty reflection
            inp = torch.cat([p_ids, r_ids], dim=1)                                       # [1, P+R]
            attn = torch.ones_like(inp)
            labels = inp.clone()
            labels[:, :p_ids.shape[1]] = -100                                            # only reflection tokens get loss

            out = model(input_ids=inp, attention_mask=attn, labels=labels)
            nll = out.loss
            w = max(min(float(A_i), 1.0), -1.0)                                          # clip advantage
            loss_refl = nll * (-w) if loss_refl is None else loss_refl + nll * (-w)

        if loss_refl is not None:
            (loss_refl / max(len(cand), 1)).backward()
            opt_refl.step()

        # 4) KD distillation (teacher-with-reflection → student-without), gated by advantage
        best = max(cand, key=lambda x: x[0]) if cand else None
        did_kd = False
        if best and best[0] >= cfg.tau_gate:
            _A_best, _reflection, code_text, teacher_scores, gen_ids = best
            # require generated tokens and teacher scores
            if gen_ids is not None and gen_ids.numel() > 0 and len(teacher_scores) > 0:
                teacher_topk = collect_teacher_topk(teacher_scores, topk=cfg.topk_kd)

                model.set_adapter("default")
                opt_student.zero_grad(set_to_none=True)

                # Student conditions on *original* prompt only (no reflection)
                x_ids = tok(pb.prompt + "\n# Write the function above.", return_tensors="pt").to(model.device)
                y_ids = gen_ids.to(model.device)  # generated tokens from teacher run [1, T]

                # Forward student on concat(x, y) to get logits over y
                concat_ids = torch.cat([x_ids["input_ids"], y_ids], dim=1)
                attn = torch.ones_like(concat_ids)
                out = model(input_ids=concat_ids, attention_mask=attn, labels=concat_ids)
                logits = out.logits[0, -y_ids.shape[1]:, :]                               # [T, V]

                # KD + optional SFT
                kd_loss = kd_kl_topk(logits.float(), teacher_topk)
                sft_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_ids[0], ignore_index=-100)
                total = cfg.kd_weight * kd_loss + cfg.sft_weight * sft_loss
                total.backward()
                opt_student.step()
                did_kd = True

        # progress bar
        bestA = best[0] if best else 0.0
        pbar.set_postfix_str(f"R0={R0:.2f} bestA={bestA:.2f} KD={'Y' if did_kd else 'N'}")

    # Save adapters
    model.set_adapter("default")
    model.save_pretrained(os.path.join(cfg.save_dir, "student_adapter"))
    model.set_adapter("reflector")
    model.save_pretrained(os.path.join(cfg.save_dir, "reflector_adapter"))
    print("Saved adapters to", cfg.save_dir)


if __name__ == "__main__":
    main()
