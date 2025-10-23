from dataclasses import dataclass

@dataclass
class TrainCfg:
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    save_dir: str = "./checkpoints"
    max_problems: int = 200
    k_reflections: int = 2
    tau_gate: float = 0.0  # minimum advantage to accept for KD
    topk_kd: int = 64
    lr_student: float = 5e-6
    lr_reflector: float = 5e-6
    max_new_tokens_code: int = 320
    max_new_tokens_refl: int = 120
    temp_code: float = 0.2
    temp_refl: float = 0.7
    sft_weight: float = 0.5
    kd_weight: float = 1.0
    anchor_weight: float = 0.02
    device: str = "cuda"
