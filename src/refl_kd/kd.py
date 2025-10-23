import torch
import torch.nn.functional as F
from typing import List, Dict

@torch.no_grad()
def collect_teacher_topk(scores: List[torch.Tensor], topk:int=64):
    # Convert generate(..., output_scores=True) 'scores' into per-step top-k distributions.
    out = []
    for step_logits in scores:  # each is [B, V]
        probs = F.softmax(step_logits[0].float(), dim=-1)
        vals, idx = torch.topk(probs, k=min(topk, probs.size(-1)))
        out.append({"ids": idx.cpu(), "probs": vals.cpu()})
    return out

def kd_kl_topk(student_logits: torch.Tensor, teacher_topk: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    # student_logits: [T, V]; teacher_topk: list length T with dicts of 'ids' [k] and 'probs' [k]
    T, V = student_logits.shape
    losses = []
    for t in range(min(T, len(teacher_topk))):
        info = teacher_topk[t]
        ids = info["ids"].to(student_logits.device)
        p = info["probs"].to(student_logits.device)
        s_logits = student_logits[t, ids]
        s = F.softmax(s_logits, dim=-1)
        p = p / (p.sum() + 1e-8)
        kl = (p * (p.add(1e-8).log() - s.add(1e-8).log())).sum()
        losses.append(kl)
    return torch.stack(losses).mean() if losses else student_logits.mean()*0
