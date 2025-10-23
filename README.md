# refl-kd-coder

Minimal, modular code to try **reflection RL + distillation** for coding:
- **RL on reflection tokens** with advantage gating (good reflections ↑, bad ↓)
- **KD/SFT** of the core policy so it behaves **as if** it had the reflection, **without** needing one at test-time.

**Target model:** Qwen2.5-Coder-1.5B-Instruct (or any small Causal LM)  
**Train data:** MBPP (with unit tests) — do **not** train on HumanEval.  
**Eval:** HumanEval (pass@1), HumanEval+ / MBPP+ via EvalPlus.

## Quick start

```bash
# 1) Setup
conda create -n reflkd python=3.10 -y
conda activate reflkd
pip install -r requirements.txt

# 2) Optional: install EvalPlus (for HumanEval+/MBPP+ & sandboxed runs)
pip install -U "evalplus[vllm] @ git+https://github.com/evalplus/evalplus"

# 3) Baseline eval on HumanEval
python -m refl_kd.eval_humaneval --model Qwen/Qwen2.5-Coder-1.5B-Instruct --passk 1

# 4) Train (small run on MBPP)
python -m refl_kd.train   --model Qwen/Qwen2.5-Coder-1.5B-Instruct   --dataset mbpp   --max_problems 200   --k_reflections 2   --save_dir ./checkpoints

# 5) Eval the student (no reflection) on HumanEval(+)
python -m refl_kd.eval_humaneval --model Qwen/Qwen2.5-Coder-1.5B-Instruct --student_adapter ./checkpoints/student_adapter --passk 1
```

## Files

```
src/refl_kd/
  data.py         # MBPP loader (prompts + tests)
  evaluator.py    # run code against tests (simple); use EvalPlus for sandbox
  prompts.py      # tiny reflection & retry templates
  models.py       # load base model; add two LoRA adapters: default(student) + reflector
  kd.py           # top-k KD (teacher-with-reflection → student-without)
  train.py        # baseline → reflections → RL on refl → KD distill
  eval_humaneval.py  # evaluate pass@1 on HumanEval (+ EvalPlus if installed)
  utils.py        # small config dataclass
requirements.txt
```

## Notes
- This repo is intentionally **minimal**. It avoids heavy abstractions and keeps the training loop in a single file.
- The reflection RL is a **basic REINFORCE-style** update that weights the reflection NLL by advantage. For more advanced RL, consider TRL (GRPO/PPO).
- The evaluator here is **not sandboxed**. Prefer running tests via **EvalPlus** with Docker for safety and robustness.
- Respect licenses for models and datasets.
