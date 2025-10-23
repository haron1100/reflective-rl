import torch, torch.nn.functional as F

@torch.no_grad()
def collect_teacher_topk(scores, topk: int = 64):
    out = []
    for step_logits in scores:  # [B, V] on device
        probs = F.softmax(step_logits.float(), dim=-1)
        vals, idx = torch.topk(probs, k=min(topk, probs.size(-1)))
        out.append({"ids": idx, "probs": vals})  # stay on device
    return out

def kd_kl_topk(student_logits: torch.Tensor, teacher_topk):
    losses = []
    T = min(len(teacher_topk), student_logits.size(0))
    for t in range(T):
        info = teacher_topk[t]
        ids = info["ids"]                # on device
        p = info["probs"]                # on device
        s = F.softmax(student_logits[t, ids], dim=-1)
        p = p / (p.sum() + 1e-8)
        kl = (p * (p.add(1e-8).log() - s.add(1e-8).log())).sum()
        losses.append(kl)
    return torch.stack(losses).mean() if losses else student_logits.mean()*0