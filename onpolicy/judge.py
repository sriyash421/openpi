"""VLM judge via OpenRouter: instruction + 8 frames of an attempt -> {success, feedback}.

Mirrors DiT's verifiers/open_router.py (retry with backoff, usage-based cost tracking, thread-safe totals) but uses
only the standard library so it runs inside the py3.8 LIBERO client. The judge sees the instruction and frames only.
"""
import base64
import io
import json
import os
import re
import ssl
import threading
import time
import urllib.request


URL = "https://openrouter.ai/api/v1/chat/completions"
# uv's standalone Python ships no CA bundle; use the host's (RHEL) or the container's (Ubuntu).
_CA = next((p for p in (os.environ.get("SSL_CERT_FILE"), "/etc/pki/tls/certs/ca-bundle.crt", "/etc/ssl/certs/ca-certificates.crt")
            if p and os.path.exists(p)), None)
_SSL = ssl.create_default_context(cafile=_CA)
MODEL = "google/gemini-3.8-flash"

SUCCESS_FEEDBACK = "The task was completed correctly."

# Hard cap on this project's judge spend (USD). Every paid call appends its cost to LEDGER (first line: the spend
# before the ledger existed); judges refuse to call once the total reaches BUDGET, and onpolicy/run.py stops.
LEDGER = os.environ.get("JUDGE_LEDGER", "/gscratch/scrubbed/sriyash/openpi_pt/judge_ledger.txt")
BUDGET = float(os.environ.get("JUDGE_BUDGET", "500"))


class BudgetExceeded(RuntimeError):
    pass


def spent(ledger=LEDGER):
    total = 0.0
    try:
        for line in open(ledger):
            try:
                total += float(line.split()[-1])
            except (ValueError, IndexError):
                pass
    except FileNotFoundError:
        pass
    return total

# Used when the simulator reports failure: the VLM only explains what went wrong.
FAILURE_PROMPT = (
    "You are giving feedback to a robot arm that FAILED a household manipulation task in simulation.\n"
    "Task instruction: \"{instruction}\"\n"
    "The images are {n} frames from the failed attempt in time order (first = start, last = end).\n"
    "Say what went wrong semantically (e.g. wrong object picked, wrong target location, object dropped, task stopped "
    "early) and what the robot should do instead, naming the objects as they appear in the instruction.\n"
    "Answer with JSON only: {{\"feedback\": \"<one sentence, at most 30 words>\"}}."
)

PROMPT = (
    "You are judging a robot arm attempting a household manipulation task in simulation.\n"
    "Task instruction: \"{instruction}\"\n"
    "The images are {n} frames from the attempt in time order (first = start, last = end).\n"
    "Decide whether the final state satisfies the instruction. If it does not, say what went wrong semantically "
    "(e.g. wrong object picked, wrong target location, object dropped, task stopped early) and what the robot should "
    "do instead, naming the objects as they appear in the instruction.\n"
    "Answer with JSON only: {{\"success\": true|false, \"feedback\": \"<one sentence, at most 30 words>\"}}. "
    "If the attempt succeeded, feedback should be \"The task was completed correctly.\""
)


class Judge:
    def __init__(self, model=MODEL, image_size=256, max_tokens=2048, reasoning_effort="low", retries=3, timeout=90):
        self.model, self.image_size, self.max_tokens, self.retries, self.timeout = model, image_size, max_tokens, retries, timeout
        self.reasoning = {"effort": reasoning_effort, "exclude": True}  # Gemini 3.8 Flash reasons; its tokens count toward max_tokens
        self.key = os.environ["OPENROUTER_API_KEY"]
        self.cost, self.calls, self.failures = 0.0, 0, 0
        self._lock = threading.Lock()
        self._spent, self._checked = 0.0, 0.0

    def _check_budget(self):
        if time.time() - self._checked > 30:
            self._spent, self._checked = spent(), time.time()
        if self._spent >= BUDGET:
            raise BudgetExceeded(f"judge spend ${self._spent:.2f} >= budget ${BUDGET:.0f} ({LEDGER})")

    def _encode(self, frame):
        from PIL import Image  # lazy: onpolicy/run.py imports this module (for spent()) from a venv without PIL

        buf = io.BytesIO()
        Image.fromarray(frame).resize((self.image_size, self.image_size)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    def __call__(self, instruction, frames, sim_success=None):
        """Returns {"success": bool|None, "feedback": str, "cost": float, "error": str|None}.

        With sim_success given, the simulator is the verdict: successes are labeled without a VLM call, and failures
        only ask the VLM to explain what went wrong. Without it, the VLM judges success itself (validation mode)."""
        if sim_success:
            return {"success": True, "feedback": SUCCESS_FEEDBACK, "cost": 0.0, "error": None}
        prompt = FAILURE_PROMPT if sim_success is False else PROMPT
        content = [{"type": "text", "text": prompt.format(instruction=instruction, n=len(frames))}]
        content += [{"type": "image_url", "image_url": {"url": self._encode(f)}} for f in frames]
        body = json.dumps({"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0,
                           "max_tokens": self.max_tokens, "reasoning": self.reasoning, "usage": {"include": True}}).encode()
        err = None
        for attempt in range(self.retries + 1):
            self._check_budget()
            try:
                req = urllib.request.Request(URL, data=body, headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout, context=_SSL) as r:
                    out = json.loads(r.read())
                cost = float(out.get("usage", {}).get("cost", 0.0) or 0.0)
                with open(LEDGER, "a") as f:  # O_APPEND: one short line per paid call, concurrent writers are safe
                    f.write(f"{self.model} {cost:.6f}\n")
                self._spent += cost
                text = out["choices"][0]["message"]["content"] or ""
                parsed = json.loads(re.search(r"\{.*\}", text, re.S).group(0))
                with self._lock:
                    self.cost += cost
                    self.calls += 1
                success = False if sim_success is False else bool(parsed["success"])
                return {"success": success, "feedback": str(parsed["feedback"]).strip(), "cost": cost, "error": None}
            except Exception as e:  # network, rate limit, malformed JSON: back off and retry
                err = repr(e)
                time.sleep(2 ** attempt)
        with self._lock:
            self.failures += 1
        return {"success": None, "feedback": "", "cost": 0.0, "error": err}
