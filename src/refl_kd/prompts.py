REFLECTION_SYSTEM = (
    "You are a concise code debugger. "
    "Given a Python function prompt and failing behavior, propose a minimal fix plan.\n"
    "Use this schema:\n"
    "1) failing_case\n2) hypothesis\n3) patch_plan\n"
    "Be specific, under 120 tokens."
)

def format_reflection_prompt(problem_prompt: str, baseline_score: float) -> str:
    return (
        f"{REFLECTION_SYSTEM}\n\n"
        f"PROBLEM:\n{problem_prompt}\n\n"
        f"BASELINE_SCORE: {baseline_score:.2f}\n"
        "Write your reflection:"
    )

def format_retry_prompt(problem_prompt: str, reflection: str) -> str:
    return (
        "You will write a single Python function that solves the task.\n"
        "Do not read input. Do not print. Keep helpers local.\n\n"
        f"PROBLEM:\n{problem_prompt}\n\n"
        f"REFLECTION:\n{reflection}\n\n"
        "Write the function now:\n"
    )
