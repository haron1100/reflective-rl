import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def _pick_dtype():
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        return torch.float16
    return torch.float32

def load_model_tokenizer(model_name: str, device: str = "auto"):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    dtype = _pick_dtype()
    # load first (usually CPU); then move if requested
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")

    device = device.lower()
    if device in {"mps", "cpu"}:
        model.to(device)
    # for cuda, device_map="auto" already does the right thing

    return model, tok

def add_lora_adapters(model, r:int=16, alpha:int=16, dropout:float=0.05):
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, cfg)  # default adapter
    peft_model.print_trainable_parameters()
    peft_model.add_adapter("reflector", cfg)
    return peft_model
