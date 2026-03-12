import streamlit as st
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from datetime import datetime

# ---------- App Config ----------
st.set_page_config(
    page_title="Summarizer Studio",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');

:root {
  --bg: #0b0e11;
  --bg-soft: #0f1418;
  --card: #151b20;
  --card-2: #1a232a;
  --text: #e9eef2;
  --muted: #9fb1bf;
  --accent: #7ad6f7;
  --accent-2: #ffb86b;
  --accent-3: #9b8cff;
  --success: #7ee0a3;
  --danger: #ff7b7b;
  --stroke: rgba(255,255,255,0.08);
  --glass: rgba(255,255,255,0.04);
}

html, body, [class*="css"]  {
  font-family: 'Sora', sans-serif;
  color: var(--text);
}

.stApp {
  background:
    radial-gradient(1200px 600px at 5% -10%, rgba(122,214,247,0.18), transparent),
    radial-gradient(900px 600px at 90% -20%, rgba(155,140,255,0.18), transparent),
    radial-gradient(800px 400px at 40% 120%, rgba(255,184,107,0.20), transparent),
    linear-gradient(180deg, #0a0d10 0%, #0b0e11 100%);
}

.block-container {
  padding-top: 1.8rem;
  padding-bottom: 2.2rem;
}

.topbar {
  background: linear-gradient(135deg, rgba(122,214,247,0.18), rgba(155,140,255,0.18));
  border: 1px solid var(--stroke);
  border-radius: 20px;
  padding: 18px 22px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
}

.brand-title {
  font-size: 30px;
  font-weight: 700;
  letter-spacing: 0.2px;
}

.brand-sub {
  color: var(--muted);
  margin-top: 4px;
}

.pill-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid var(--stroke);
  background: var(--glass);
  font-size: 12px;
  color: var(--muted);
}

.card {
  background: var(--card);
  border: 1px solid var(--stroke);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 12px 30px rgba(0,0,0,0.25);
}

.card-2 {
  background: var(--card-2);
  border: 1px solid var(--stroke);
  border-radius: 14px;
  padding: 14px;
}

.panel-title {
  font-size: 14px;
  letter-spacing: 0.2px;
  text-transform: uppercase;
  color: var(--muted);
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.metric-label {
  color: var(--muted);
  font-size: 12px;
}

.metric-value {
  font-size: 22px;
  font-weight: 600;
}

.chat-bubble {
  background: #0f1418;
  border: 1px solid rgba(122,214,247,0.35);
  border-radius: 18px;
  padding: 16px;
  line-height: 1.6;
}

.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  background: rgba(122,214,247,0.12);
  border: 1px solid rgba(122,214,247,0.4);
  color: var(--accent);
}

.mono {
  font-family: 'Space Mono', monospace;
  letter-spacing: 0.2px;
}

.stTextArea textarea {
  background: #0f1418;
  color: var(--text);
  border-radius: 14px;
  border: 1px solid var(--stroke);
}

.stButton>button {
  border-radius: 999px;
  padding: 0.55rem 1.2rem;
  border: 1px solid var(--stroke);
  background: linear-gradient(135deg, rgba(122,214,247,0.32), rgba(155,140,255,0.24));
  color: var(--text);
  font-weight: 600;
}

.stButton>button:hover {
  border-color: rgba(122,214,247,0.8);
}

.ghost {
  border: 1px dashed rgba(255,255,255,0.12);
  padding: 10px 12px;
  border-radius: 12px;
  color: var(--muted);
  font-size: 12px;
}

hr {
  border: none;
  border-top: 1px solid var(--stroke);
}

@media (max-width: 1200px) {
  .stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Utilities ----------

def ensure_nltk():
    nltk.download("punkt", quiet=True)
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass


def count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, truncation=False))


def split_into_chunks(tokenizer, text: str, max_tokens: int = 900):
    from nltk.tokenize import sent_tokenize

    sentences = sent_tokenize(text)
    chunks = []
    current = ""
    current_tokens = 0
    for sent in sentences:
        sent_tokens = count_tokens(tokenizer, sent)
        if current_tokens + sent_tokens <= max_tokens:
            current = f"{current} {sent}".strip()
            current_tokens += sent_tokens
        else:
            if current:
                chunks.append(current)
            current = sent
            current_tokens = sent_tokens
    if current:
        chunks.append(current)
    return chunks


@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        model = model.cuda()
    summarizer = pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    return summarizer, tokenizer, device


def summarize_text(text: str, summarizer, tokenizer, max_len: int, min_len: int, auto_chunk: bool):
    token_count = count_tokens(tokenizer, text)
    if auto_chunk and token_count > 900:
        chunks = split_into_chunks(tokenizer, text, max_tokens=900)
        chunk_summaries = []
        for chunk in chunks:
            out = summarizer(
                chunk,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
            chunk_summaries.append(out[0]["summary_text"])
        combined = " ".join(chunk_summaries)
        if count_tokens(tokenizer, combined) > 900:
            out = summarizer(
                combined,
                max_length=max_len * 2,
                min_length=min_len,
                do_sample=False,
                truncation=True,
            )
            return out[0]["summary_text"], token_count, len(chunks)
        return combined, token_count, len(chunks)

    out = summarizer(
        text,
        max_length=max_len,
        min_length=min_len,
        do_sample=False,
        truncation=True,
    )
    return out[0]["summary_text"], token_count, 1


def to_bullets(text: str):
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    return "\n".join([f"- {s}." for s in sentences])


def word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("<div class='badge'>Summarizer Studio</div>", unsafe_allow_html=True)
    st.write("")
    model_label = st.selectbox(
        "Model preset",
        options=[
            "Fast (distilbart-cnn-12-6)",
            "Balanced (bart-large-cnn)",
        ],
        index=0,
    )
    model_name = "sshleifer/distilbart-cnn-12-6" if "Fast" in model_label else "facebook/bart-large-cnn"

    max_len = st.slider("Max summary length", 80, 280, 160, 10)
    min_len = st.slider("Min summary length", 20, 120, 60, 5)
    auto_chunk = st.toggle("Auto chunk long docs", value=True)
    bullets = st.toggle("Format as bullet points", value=False)

    st.write("")
    st.caption("Model downloads happen on first run. CPU is supported.")
    st.markdown("<div class='ghost'>Tip: Use the Compare tab to preview two summary lengths side-by-side.</div>", unsafe_allow_html=True)

# ---------- Header ----------
preset_short = "FAST" if "Fast" in model_label else "BAL"
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {
        "input_words": 0,
        "summary_words": 0,
        "compression": "--",
        "chunks": 0,
    }

lm = st.session_state.last_metrics

top_left, top_right = st.columns([2.4, 1])
with top_left:
    st.markdown(
        f"""
<div class="topbar">
  <div>
    <div class="brand-title">Summarizer Studio</div>
    <div class="brand-sub">High-signal summaries with long-doc chunking, previews, and history.</div>
    <div class="pill-row" style="margin-top:10px;">
      <span class="pill">Mode: Summarize</span>
      <span class="pill">Auto Chunking</span>
      <span class="pill">Multi-Length Compare</span>
    </div>
  </div>
  <div class="card-2" style="min-width:180px;">
    <div class="metric-label">Session</div>
    <div class="metric-value">Ready</div>
    <div class="metric-label">Preset</div>
    <div class="metric-value mono">{preset_short}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
with top_right:
    st.markdown(
        f"""
<div class="card">
  <div class="panel-title">Last Run</div>
  <div class="stat-grid" style="margin-top:10px;">
    <div class="card-2"><div class="metric-label">Input words</div><div class="metric-value">{lm["input_words"]}</div></div>
    <div class="card-2"><div class="metric-label">Summary words</div><div class="metric-value">{lm["summary_words"]}</div></div>
    <div class="card-2"><div class="metric-label">Compression</div><div class="metric-value">{lm["compression"]}</div></div>
    <div class="card-2"><div class="metric-label">Chunks</div><div class="metric-value">{lm["chunks"]}</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.write("")

# ---------- Tabs ----------
summary_tab, compare_tab, history_tab = st.tabs(["Summarize", "Compare", "History"])

with summary_tab:
    left, right = st.columns([1.2, 1])
    with left:
        st.markdown("<div class='card'><div class='panel-title'>Input</div></div>", unsafe_allow_html=True)
        with st.form("summarize_form"):
            input_mode = st.radio("Input mode", ["Paste text", "Upload .txt"], horizontal=True)
            raw_text = ""
            if input_mode == "Paste text":
                raw_text = st.text_area("", height=260, placeholder="Paste your document here...")
            else:
                file = st.file_uploader("Upload a .txt file", type=["txt", "md"])
                if file:
                    raw_text = file.read().decode("utf-8", errors="ignore")
            submitted = st.form_submit_button("Generate Summary")
        if raw_text.strip():
            st.caption(f"Input words: {word_count(raw_text)}")

    with right:
        st.markdown("<div class='card'><div class='panel-title'>Output</div></div>", unsafe_allow_html=True)
        output_placeholder = st.empty()
        st.write("")
        st.markdown("<div class='card-2'><div class='panel-title'>Actions</div></div>", unsafe_allow_html=True)
        summary_download = st.session_state.get("last_summary", "")
        st.download_button(
            "Download last summary",
            data=summary_download,
            file_name="summary.txt",
            disabled=not bool(summary_download),
        )

    if submitted:
        if not raw_text.strip():
            st.error("No input provided.")
        else:
            ensure_nltk()
            with st.spinner("Loading model and summarizing..."):
                summarizer, tokenizer, device = load_model(model_name)
                summary, token_count, chunks = summarize_text(
                    raw_text,
                    summarizer,
                    tokenizer,
                    max_len,
                    min_len,
                    auto_chunk,
                )

            display_text = to_bullets(summary) if bullets else summary
            output_placeholder.markdown(
                f"<div class='chat-bubble'>{display_text}</div>",
                unsafe_allow_html=True,
            )
            st.session_state.last_summary = summary

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown("<div class='card-2'><div class='metric-label'>Tokens</div>"
                        f"<div class='metric-value'>{token_count}</div></div>",
                        unsafe_allow_html=True)
            c2.markdown("<div class='card-2'><div class='metric-label'>Chunks</div>"
                        f"<div class='metric-value'>{chunks}</div></div>",
                        unsafe_allow_html=True)
            c3.markdown("<div class='card-2'><div class='metric-label'>Model</div>"
                        f"<div class='metric-value'>{'GPU' if device==0 else 'CPU'}</div></div>",
                        unsafe_allow_html=True)
            ratio = max(1, len(summary.split()))
            c4.markdown("<div class='card-2'><div class='metric-label'>Summary words</div>"
                        f"<div class='metric-value'>{ratio}</div></div>",
                        unsafe_allow_html=True)

            input_words = word_count(raw_text)
            compression = f"{round((ratio / max(1, input_words)) * 100)}%"
            st.session_state.last_metrics = {
                "input_words": input_words,
                "summary_words": ratio,
                "compression": compression,
                "chunks": chunks,
            }

            # History
            if "history" not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, {
                "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
                "input_words": len(raw_text.split()),
                "summary": summary,
                "model": model_name,
            })

with compare_tab:
    st.markdown("<div class='card'><div class='panel-title'>Compare two summary lengths</div></div>", unsafe_allow_html=True)
    raw_text_compare = st.text_area("Input text", height=220, key="compare_text")
    c1, c2 = st.columns(2)
    with c1:
        max_a = st.slider("Short summary max", 60, 180, 110, 10)
        min_a = st.slider("Short summary min", 20, 80, 40, 5)
    with c2:
        max_b = st.slider("Detailed summary max", 140, 320, 220, 10)
        min_b = st.slider("Detailed summary min", 40, 140, 80, 5)

    if st.button("Run Comparison"):
        if not raw_text_compare.strip():
            st.error("No input provided.")
        else:
            ensure_nltk()
            with st.spinner("Summarizing..."):
                summarizer, tokenizer, _ = load_model(model_name)
                summary_a, _, _ = summarize_text(raw_text_compare, summarizer, tokenizer, max_a, min_a, auto_chunk)
                summary_b, _, _ = summarize_text(raw_text_compare, summarizer, tokenizer, max_b, min_b, auto_chunk)

            ca, cb = st.columns(2)
            with ca:
                st.markdown("<div class='card-2'><b>Short</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-bubble'>{summary_a}</div>", unsafe_allow_html=True)
            with cb:
                st.markdown("<div class='card-2'><b>Detailed</b></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='chat-bubble'>{summary_b}</div>", unsafe_allow_html=True)

with history_tab:
    st.markdown("<div class='card'><div class='panel-title'>History</div></div>", unsafe_allow_html=True)
    history = st.session_state.get("history", [])
    if not history:
        st.info("No summaries yet. Generate one to populate history.")
    else:
        for item in history[:10]:
            st.markdown(
                f"<div class='card-2'><div class='metric-label'>{item['time']}"
                f" | {item['input_words']} words | {item['model']}</div>"
                f"<div class='chat-bubble'>{item['summary']}</div></div>",
                unsafe_allow_html=True,
            )
