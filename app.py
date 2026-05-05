"""
app.py  —  AI English Tutor  (Streamlit frontend)
==================================================

HOW TRANSCRIPTION WORKS — the correct pattern:
  1. mic_recorder (streamlit-mic-recorder) returns {"bytes":..., "id":...}
     only when a NEW recording is complete. The "id" changes every recording.
  2. We compare audio["id"] to session_state.last_audio_id.
  3. If new  → transcribe → set st.session_state["answer_box"] = transcript
              (set the session_state KEY, not a separate variable)
  4. Render  st.text_area(key="answer_box") with NO value= param.
             Streamlit reads the key from session_state → text appears.
  5. User can then edit the text_area freely.
  6. On Submit → read st.session_state["answer_box"] for the final answer.

WHY THIS WORKS: Setting st.session_state["key"] before rendering a widget
with that key causes Streamlit to initialise the widget with that value.
This is the ONLY correct way to pre-populate a text_area from code.
Using value= alone is ignored on subsequent reruns for keyed widgets.

INSTALL: pip install streamlit streamlit-mic-recorder faster-whisper langgraph langchain-ollama
"""

import os, uuid, tempfile
import streamlit as st
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
from agent import graph, MAX_STAGE

# ── page config ───────────────────────────────────────────────────
st.set_page_config(page_title="AI English Tutor", page_icon="🎓", layout="centered")

# ── css ───────────────────────────────────────────────────────────
st.markdown("""
<style>
body { background:#f6f8ff; }

.task-card {
    background:#fff;
    border-radius:14px;
    padding:1.6rem 2rem;
    box-shadow:0 2px 12px rgba(0,0,0,.07);
    margin-bottom:1.2rem;
}
.pill {
    display:inline-block;
    color:#fff;
    border-radius:20px;
    padding:5px 20px;
    font-size:.83rem;
    font-weight:700;
    margin-bottom:.9rem;
}
.instruction {
    font-size:1.08rem;
    line-height:1.75;
    margin-bottom:.4rem;
}
.hint {
    color:#6b7280;
    font-size:.9rem;
    margin-bottom:1rem;
}
.fb-pass {
    background:#f0fdf4;border:1.5px solid #22c55e;
    border-radius:10px;padding:.9rem 1.1rem;margin-bottom:1rem;
}
.fb-fail {
    background:#fffbeb;border:1.5px solid #f59e0b;
    border-radius:10px;padding:.9rem 1.1rem;margin-bottom:.6rem;
}
.fb-correction {
    background:#eff6ff;border:1.5px solid #3b82f6;
    border-radius:10px;padding:.9rem 1.1rem;margin-bottom:1rem;
}
.mic-hint { font-size:.8rem;color:#9ca3af;margin-top:-.3rem;margin-bottom:.7rem; }
</style>
""", unsafe_allow_html=True)

# ── whisper ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading speech model…")
def _whisper():
    # device="cpu" + compute_type="int8" required on CPU-only machines
    return WhisperModel("tiny.en", device="cpu", compute_type="int8")

def transcribe(audio_bytes: bytes) -> str:
    """Write bytes to a .wav temp file, run faster-whisper, return text."""
    wm = _whisper()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        path = f.name
    try:
        segs, _ = wm.transcribe(path, language="en", beam_size=1)
        return " ".join(s.text.strip() for s in segs).strip()
    except Exception:
        return ""
    finally:
        os.unlink(path)

# ── session state defaults ────────────────────────────────────────
def _init():
    ss = st.session_state
    ss.setdefault("tid",            str(uuid.uuid4()))
    ss.setdefault("last_audio_id",  None)
    ss.setdefault("show_feedback",  False)
    ss.setdefault("prev_stage",     None)
    ss.setdefault("prev_retry",     None)
_init()

cfg = {"configurable": {"thread_id": st.session_state.tid}}

# ── bootstrap graph ───────────────────────────────────────────────
gs = graph.get_state(cfg)
if not gs.values:
    graph.invoke({"stage":1,"retry":0,"user_input":"",
                  "result":"","feedback":"","correction":"","done":False}, cfg)
    gs = graph.get_state(cfg)

v  = gs.values
task       = v.get("task", {})
stage      = v.get("stage", 1)
retry      = v.get("retry", 0)
done       = v.get("done", False)
result     = v.get("result", "")
feedback   = v.get("feedback", "")
correction = v.get("correction", "")

# ── detect stage/retry change → reset answer box ─────────────────
if (st.session_state.prev_stage, st.session_state.prev_retry) != (stage, retry):
    st.session_state.last_audio_id = None
    # Reset the answer box for the new task
    # Key pattern: "box_{stage}_{retry}" — fresh key per task
    # Setting it to "" here means the text_area will open empty
    st.session_state[f"box_{stage}_{retry}"] = ""
    st.session_state.prev_stage = stage
    st.session_state.prev_retry = retry

# ── stage meta ────────────────────────────────────────────────────
COLORS = {1: "#c0392b", 2: "#1976d2", 3: "#388e3c"}
ICONS  = {1: "🖼️", 2: "✏️", 3: "🗣️"}
color  = COLORS.get(min(stage, MAX_STAGE), "#555")
icon   = ICONS.get(min(stage, MAX_STAGE), "📚")

# ── header ────────────────────────────────────────────────────────
hc, rc = st.columns([5, 1])
with hc:
    st.markdown("## 🎓 AI English Tutor")
with rc:
    if st.button("↩ Restart"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

pct = min((stage - 1) / MAX_STAGE, 1.0) if not done else 1.0
st.progress(pct, text=f"Stage {min(stage,MAX_STAGE)} of {MAX_STAGE}")
st.divider()

# ── finished ──────────────────────────────────────────────────────
if done:
    st.markdown("""
    <div class="task-card" style="text-align:center;padding:2.5rem">
        <h1>🎉 Well done!</h1>
        <p style="font-size:1.1rem">You completed all 3 stages. Keep practising!</p>
    </div>""", unsafe_allow_html=True)
    st.balloons()
    st.stop()

# ── feedback (shown only once, right after submit) ────────────────
if st.session_state.show_feedback and result:
    if result == "pass":
        st.markdown(
            f'<div class="fb-pass">✅ <strong>Correct!</strong> {feedback}</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="fb-fail">💡 <strong>Not quite.</strong> {feedback}</div>',
            unsafe_allow_html=True)
        if correction:
            st.markdown(
                f'<div class="fb-correction">📝 <strong>Corrected:</strong> {correction}</div>',
                unsafe_allow_html=True)
    st.session_state.show_feedback = False   # show only once

# ── task card ─────────────────────────────────────────────────────
attempt = "🔄 Second Attempt" if retry else "📝 First Attempt"
st.markdown(
    f'<div class="pill" style="background:{color}">'
    f'{icon} {task.get("title","Stage "+str(stage))} &nbsp;|&nbsp; {attempt}'
    f'</div>', unsafe_allow_html=True)

image_path   = task.get("image")
instruction  = task.get("instruction", "")
hint         = task.get("example", "")
input_mode   = task.get("input_mode", "voice")

# Image (Stage 1 only) shown beside instruction so no scrolling needed
if image_path and os.path.exists(image_path):
    ic, tc = st.columns([1, 1.4], gap="large")
    with ic:
        st.image(image_path, use_container_width=True)
    with tc:
        st.markdown(f'<p class="instruction">{instruction}</p>', unsafe_allow_html=True)
        if hint:
            st.markdown(f'<p class="hint">{hint}</p>', unsafe_allow_html=True)
elif image_path:
    st.warning(f"Image not found: {image_path}")
    st.markdown(f'<p class="instruction">{instruction}</p>', unsafe_allow_html=True)
else:
    st.markdown(f'<p class="instruction">{instruction}</p>', unsafe_allow_html=True)
    if hint:
        st.markdown(f'<p class="hint">{hint}</p>', unsafe_allow_html=True)

st.divider()

# ── answer box key (unique per stage + retry) ─────────────────────
box_key = f"box_{stage}_{retry}"

# ── voice input ───────────────────────────────────────────────────
if input_mode == "voice":
    st.markdown("**🎙️ Record your answer**")
    st.markdown('<p class="mic-hint">Press Start → speak → press Stop → '
                'text appears in the box below → edit if needed → Submit</p>',
                unsafe_allow_html=True)

    audio = mic_recorder(
        start_prompt="⏺ Start Recording",
        stop_prompt="⏹ Stop Recording",
        just_once=True,             # returns audio dict once, then None — prevents re-transcription
        use_container_width=False,
        format="wav",               # wav → faster-whisper reads it without format detection issues
        key=f"mic_{stage}_{retry}", # fresh widget key per task
    )

    # ── THE CORRECT TRANSCRIPTION PATTERN ────────────────────────
    # When mic_recorder returns a new recording (new id),
    # we set st.session_state[box_key] = transcript BEFORE the text_area renders.
    # Streamlit will then initialise the text_area widget with this value.
    if audio and audio.get("id") != st.session_state.last_audio_id:
        st.session_state.last_audio_id = audio["id"]
        with st.spinner("Transcribing…"):
            text = transcribe(audio["bytes"])
        if text:
            st.session_state[box_key] = text   # ← KEY: set the widget's session_state key
        else:
            st.warning("Couldn't detect speech — please try again or type below.")

    # text_area reads from session_state[box_key] automatically
    st.text_area(
        "✍️ Your answer *(you can edit this before submitting)*",
        key=box_key,                # Streamlit reads st.session_state[box_key] as its value
        height=120,
        placeholder="Your transcribed speech appears here, or type directly…",
    )

else:
    # ── text-only input (Stage 2: sentence correction) ────────────
    # No mic. Simple text_area.
    st.text_area(
        "✍️ Type your corrected sentence",
        key=box_key,
        height=80,
        placeholder="Type the corrected sentence here…",
    )

# ── submit ────────────────────────────────────────────────────────
if st.button("✅ Submit", type="primary"):
    answer = st.session_state.get(box_key, "").strip()
    if not answer:
        st.error("Please record your voice or type an answer first.")
        st.stop()

    with st.spinner("Evaluating…"):
        graph.update_state(cfg, {"user_input": answer})
        graph.invoke(None, cfg)   # resume from interrupt_before["evaluate"]

    st.session_state.show_feedback = True
    st.rerun()