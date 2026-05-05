"""
agent.py  —  AI English Tutor
==============================
3 stages, 2 attempts each.

FLOW (same for every stage):
  Attempt 1  → correct  → next stage
  Attempt 1  → wrong    → show why → Attempt 2 (different task, same skill)
  Attempt 2  → correct  → next stage
  Attempt 2  → wrong    → show correct answer → next stage

STAGES:
  1  Image Description   — student sees an image, writes 1-2 sentences  (voice ok)
  2  Sentence Correction — student fixes a broken sentence               (text only)
  3  Free Speaking       — student speaks/writes about a topic           (voice ok)

INPUT MODES:
  "voice"  →  mic recorder + editable text area shown in app.py
  "text"   →  plain text input only (no mic)

EVALUATION:
  llava  for stages with an image
  llama3 for text-only stages
  Both models are told to return strict JSON only.
"""

import base64, json, os, re
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# ── paths ─────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def img(name): return os.path.join(BASE_DIR, "images", name)

# ── curriculum ────────────────────────────────────────────────────
CURRICULUM = {
    1: {
        "main": {
            "title": "Image Description",
            "instruction": "Look at the image. Write 1 or 2 sentences describing what you see.",
            "example": "Example: A woman is reading a book in the park.",
            "input_mode": "voice",
            "image": img("img1.jpeg"),
            "expected": "1-2 grammatically correct sentences describing the image.",
        },
        "retry": {
            "title": "Image Description",
            "instruction": "Look at this new image. Write 1 or 2 sentences describing what you see.",
            "example": "Remember: subject + verb. Example: 'Two children are playing football.'",
            "input_mode": "voice",
            "image": img("img2.jpeg"),
            "expected": "1-2 grammatically correct sentences describing the image.",
        },
    },
    2: {
        "main": {
            "title": "Sentence Correction",
            "instruction": 'Fix this sentence and type the corrected version:\n\n"He go to school every day."',
            "example": "Hint: with he / she / it, the verb needs -s or -es.",
            "input_mode": "text",
            "image": None,
            "expected": "He goes to school every day.",
        },
        "retry": {
            "title": "Sentence Correction",
            "instruction": 'Fix this sentence and type the corrected version:\n\n"She don\'t like cold weather."',
            "example": "Hint: with she / he / it, use doesn't (not don't).",
            "input_mode": "text",
            "image": None,
            "expected": "She doesn't like cold weather.",
        },
    },
    3: {
        "main": {
            "title": "Free Speaking",
            "instruction": "Tell me about your daily routine. Speak or write 2–3 sentences.",
            "example": "Example: 'I wake up at 7. Then I have breakfast and go to work.'",
            "input_mode": "voice",
            "image": None,
            "expected": "2-3 clear sentences in present tense about daily routine.",
        },
        "retry": {
            "title": "Free Speaking",
            "instruction": "Answer this question in 2–3 sentences: What do you like to do on weekends?",
            "example": "Try to use: first, then, after that.",
            "input_mode": "voice",
            "image": None,
            "expected": "2-3 clear, connected sentences about weekends.",
        },
    },
}

MAX_STAGE = max(CURRICULUM)  # 3

# ── state ─────────────────────────────────────────────────────────
class State(TypedDict):
    stage: int
    retry: int          # 0 = first attempt, 1 = second attempt
    task: dict          # current task dict from CURRICULUM
    user_input: str
    result: str         # "pass" | "fail" | ""
    feedback: str       # why pass/fail
    correction: str     # corrected answer shown on 2nd fail
    done: bool

# ── models ────────────────────────────────────────────────────────
_text_llm   = ChatOllama(model="llama3", temperature=0)
_vision_llm = ChatOllama(model="llava",  temperature=0)

# ── helpers ───────────────────────────────────────────────────────
def _parse(raw: str) -> dict:
    """Extract JSON from model output. Falls back to keyword heuristic."""
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.I).strip().strip("`")
    try:
        d = json.loads(cleaned)
        if isinstance(d, dict):
            d.setdefault("result", "fail")
            d.setdefault("feedback", "")
            d.setdefault("correction", "")
            return d
    except Exception:
        pass
    m = re.search(r"\{[^{}]+\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group())
            if isinstance(d, dict):
                d.setdefault("result", "fail")
                d.setdefault("feedback", "")
                d.setdefault("correction", "")
                return d
        except Exception:
            pass
    # keyword fallback — look for the result field value explicitly
    # Search for  "result": "pass"  or  "result": "fail"  pattern first
    result_match = re.search(r'"result"\s*:\s*"(pass|fail)"', raw, re.I)
    if result_match:
        r = result_match.group(1).lower()
        return {"result": r, "feedback": raw[:200], "correction": ""}
    # Last resort: if the raw output starts with or is dominated by "pass"
    # Only treat as pass when the word "pass" appears and "fail" does NOT appear
    lower = raw.lower()
    if "pass" in lower and "fail" not in lower:
        return {"result": "pass", "feedback": raw[:200], "correction": ""}
    return {"result": "fail", "feedback": raw[:200], "correction": ""}


def _b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _prompt(task: dict, answer: str) -> str:
    title = task['title']

    # Extra strictness rules depending on task type
    if title == "Sentence Correction":
        strict_rules = (
            "STRICT RULES for Sentence Correction:\n"
            "  - The student must fix ALL grammar errors in the sentence.\n"
            "  - Any remaining grammar mistake (wrong verb form, wrong auxiliary, etc.) = 'fail'.\n"
            "  - Spelling mistakes = 'fail'.\n"
            "  - If the student submits the original broken sentence unchanged = 'fail'.\n"
            "  - Only mark 'pass' if the sentence is fully grammatically correct.\n"
            f"  - The expected correct answer is: {task['expected']}\n"
        )
    elif title == "Image Description":
        strict_rules = (
            "STRICT RULES for Image Description:\n"
            "  - The student must write at least 1 complete, grammatically correct sentence.\n"
            "  - Subject-verb agreement errors = 'fail'.\n"
            "  - Missing verb = 'fail'.\n"
            "  - Spelling mistakes = 'fail'.\n"
            "  - Only mark 'pass' if grammar is correct and the description relates to the image.\n"
        )
    else:  # Free Speaking
        strict_rules = (
            "STRICT RULES for Free Speaking:\n"
            "  - The student must write at least 2 sentences.\n"
            "  - Subject-verb agreement errors = 'fail'.\n"
            "  - Clearly wrong tense usage = 'fail'.\n"
            "  - Spelling mistakes = 'fail'.\n"
            "  - Only mark 'pass' if the sentences are grammatically correct and on topic.\n"
        )

    return (
        "You are a strict English teacher evaluating a student's answer.\n"
        "Your job is to catch grammar mistakes, not to be lenient.\n\n"
        f"Task: {title}\n"
        f"Instruction shown to student: {task['instruction']}\n"
        f"Expected quality: {task['expected']}\n"
        f"Student's answer: \"{answer}\"\n\n"
        f"{strict_rules}\n"
        "feedback: one short sentence explaining exactly why the answer is correct or incorrect.\n"
        "correction: if 'fail', write the fully corrected version and explain the grammar rule that was broken; "
        "if 'pass', leave this as an empty string.\n\n"
        "YOU MUST RESPOND WITH ONLY THIS JSON — absolutely no other text before or after:\n"
        '{"result": "pass", "feedback": "...", "correction": ""}\n'
        "or\n"
        '{"result": "fail", "feedback": "...", "correction": "..."}\n'
        "Do NOT include any explanation outside the JSON."
    )

# ── nodes ─────────────────────────────────────────────────────────
def load_task(state: State) -> dict:
    stage, retry = state.get("stage", 1), state.get("retry", 0)
    if stage > MAX_STAGE:
        return {"done": True, "task": {}}
    key   = "retry" if retry else "main"
    task  = CURRICULUM[stage][key]
    return {"task": task, "stage": stage, "retry": retry, "done": False}


def evaluate(state: State) -> dict:
    task  = state["task"]
    ans   = state.get("user_input", "").strip()
    if not ans:
        return {"result": "fail",
                "feedback": "No answer received — please try again.",
                "correction": ""}

    prompt = _prompt(task, ans)
    image  = task.get("image")

    if image and os.path.exists(image):
        msg = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/jpeg;base64,{_b64(image)}"}},
        ])
        raw = _vision_llm.invoke([msg]).content
    else:
        raw = _text_llm.invoke(prompt).content

    d = _parse(raw)
    # Be conservative: only pass if result field is exactly "pass"
    r = "pass" if d["result"].strip().lower() == "pass" else "fail"
    return {"result": r, "feedback": d["feedback"], "correction": d["correction"]}


def advance(state: State) -> dict:
    if state["result"] == "pass":
        return {"stage": state["stage"] + 1, "retry": 0}
    if state.get("retry", 0) == 0:
        return {"retry": 1}               # first fail → switch to retry task
    return {"stage": state["stage"] + 1, "retry": 0}  # second fail → move on


def _route(state: State) -> str:
    return END if state.get("done") else "evaluate"

# ── graph ─────────────────────────────────────────────────────────
wf = StateGraph(State)
wf.add_node("load_task", load_task)
wf.add_node("evaluate",  evaluate)
wf.add_node("advance",   advance)

wf.set_entry_point("load_task")
wf.add_conditional_edges("load_task", _route)
wf.add_edge("evaluate", "advance")
wf.add_edge("advance",  "load_task")

graph = wf.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["evaluate"],   # pause here so app.py can inject user_input
)