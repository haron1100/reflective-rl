import subprocess, tempfile, os
from typing import List, Tuple

def run_tests_simple(code: str, tests: List[str], timeout_sec: int = 5) -> Tuple[float, str]:
    # Extremely simple (non-sandboxed) runner. Prefer EvalPlus for sandboxing/robustness.
    prog = code + "\n\n" + "\n".join(tests) + "\nprint('__RESULT__', 'OK')"
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prog.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(prog)
        try:
            out = subprocess.check_output(
                ["python", "-I", path],
                stderr=subprocess.STDOUT,
                timeout=timeout_sec
            ).decode("utf-8", errors="ignore")
        except subprocess.CalledProcessError as e:
            out = e.output.decode("utf-8", errors="ignore")
        except subprocess.TimeoutExpired:
            return 0.0, "TIMEOUT"
    failures = out.count("AssertionError")
    passed = 1.0 if failures == 0 and "__RESULT__ OK" in out else 0.0
    return passed, out
