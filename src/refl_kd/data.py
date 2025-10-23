from datasets import load_dataset
from typing import List

class Problem:
    def __init__(self, pid: str, prompt: str, tests: list):
        self.pid = pid
        self.prompt = prompt
        self.tests = tests

def _try(repo: str, subset: str | None, split: str):
    kwargs = {"path": repo}
    if subset is not None:
        kwargs["name"] = subset
    return load_dataset(**kwargs, split=split)

def load_mbpp(limit: int | None = None) -> List[Problem]:
    """
    Tries common MBPP/MBPP+ mirrors that typically expose only a `test` split.
    Returns a list of Problem(prompt, tests).
    """
    tried = []
    candidates = [
        ("Muennighoff/mbpp", "sanitized", "test"),     # sanitized with tests
        ("google-research-datasets/mbpp", "sanitized", "test"),
        ("evalplus/mbppplus", None, "test"),           # MBPP+ (richer tests)
        ("RLAIF/mbpp", None, "test"),
        ("nlile/mbpp", None, "test"),
    ]
    ds = None
    for repo, subset, split in candidates:
        try:
            ds = _try(repo, subset, split)
            break
        except Exception as e:
            tried.append(f"{repo}/{subset or '-'}:{split}: {e}")
    if ds is None:
        raise RuntimeError("Could not load MBPP. Tried:\n" + "\n".join(tried))

    problems: List[Problem] = []
    for ex in ds:
        pid = str(ex.get("task_id", len(problems)))
        prompt = ex.get("text") or ex.get("prompt") or ex.get("question")
        tests = ex.get("test_list") or ex.get("tests") or ex.get("challenge_test_list") or []
        tests = [t if isinstance(t, str) else str(t) for t in tests]
        if prompt and tests:
            problems.append(Problem(pid=pid, prompt=prompt, tests=tests))
        if limit and len(problems) >= limit:
            break
    return problems
