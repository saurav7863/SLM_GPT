import streamlit as st
import os
# Configuration loaded from environment variables (fallback to defaults)
DEFAULT_CTX: int = int(os.getenv("DEFAULT_CTX", 4000))
MODEL_PATH: str = os.getenv("MODEL_PATH", "models/Phi-3-mini-4k-instruct-q4.gguf")
LOGO_PATH: str = os.getenv("LOGO_PATH", "logo.WEBP")
WHISPER_CONFIG = {
    "model_size_or_path": os.getenv("WHISPER_MODEL_SIZE", "tiny"),
    "device": "cpu",
    "compute_type": "int8"
}
from agent import Agent
from faster_whisper import WhisperModel
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import soundfile as sf
import tempfile
import PyPDF2
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

@st.cache_resource
def load_whisper_model() -> WhisperModel:
    logger.info("Loading Whisper model with config: %s", WHISPER_CONFIG)
    return WhisperModel(**WHISPER_CONFIG)

@st.cache_resource
def get_agent(n_ctx: int) -> Agent:
    logger.info("Initializing Agent with context window %d", n_ctx)
    return Agent(model_path=MODEL_PATH, n_ctx=n_ctx, seed=42, keep_last=3)

def chat_tab(agent: Agent, whisper_model: WhisperModel) -> None:
    st.header("Chat")

    # Recording and input controls in columns for better layout
    col1, col2 = st.columns([1, 6], gap="small")
    with col1:
        if st.button("🎤 Record", key="record_btn"):
            st.session_state.recording = True
        if st.session_state.get('recording', False):
            st.info("Recording... speak then click Send.")
    with col2:
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Your message:", key="text_input")
            send = st.form_submit_button("Send", use_container_width=True)

    # Container for chat messages
    chat_container = st.container()
    # Display existing history
    with chat_container:
        for msg in agent.history:
            role = "user" if msg['role']=='user' else 'assistant'
            st.chat_message(role).write(msg['content'])

    # Handle new submission
    if send and (user_input or st.session_state.get('recording', False)):
        # Determine message text
        if user_input:
            text = user_input
        else:
            try:
                # Capture audio via WebRTC
                webrtc_ctx = webrtc_streamer(
                    key="mic",
                    mode=WebRtcMode.SENDONLY,
                    media_stream_constraints={"audio": True, "video": False},
                    audio_receiver_size=8192
                )
                frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                if not frames:
                    logger.warning("No audio captured.")
                    st.warning("No audio captured.")
                    st.stop()
                arrays = [f.to_ndarray() for f in frames]
                audio_data = np.concatenate(arrays, axis=0)
                if audio_data.dtype.kind == 'f':
                    audio_data = (audio_data * 32767).astype(np.int16)
                rate = frames[0].rate
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                sf.write(tmp.name, audio_data, rate, subtype='PCM_16')
                st.audio(tmp.name)
                segments, _ = whisper_model.transcribe(tmp.name)
                text = "".join(seg.text for seg in segments)
                st.session_state.recording = False
                st.markdown(f"**Transcribed:** {text}")
            except Exception:
                logger.error("Audio capture failed", exc_info=True)
                st.warning("Audio capture failed. Please try again.")
                return

        # Append user message
        agent.history.append({'role':'user','content':text})
        chat_container.chat_message("user").write(text)

        # Stream assistant response word-by-word into one placeholder
        placeholder = chat_container.empty()
        full_text = ""
        for token in agent.stream_chat(
            text,
            pdf_text=st.session_state.get('pdf_text')
        ):
            full_text += token
            placeholder.markdown(full_text)
        # After streaming, append final static chat message
        agent.history.append({'role':'assistant','content':full_text})
        chat_container.chat_message("assistant").write(full_text)
        placeholder.empty()

def pdf_tab(agent: Agent) -> None:
    st.header("PDF Analysis & Form-Filling")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")
    if pdf_file:
        try:
            path = f"/tmp/{pdf_file.name}"
            with open(path,"wb") as f:
                f.write(pdf_file.getbuffer())
            reader = PyPDF2.PdfReader(path)
            pdf_text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
            st.session_state.pdf_path = path
            st.session_state.pdf_text = pdf_text
            st.success("PDF loaded.")
        except Exception:
            logger.error("Failed to load or parse PDF", exc_info=True)
            st.warning("Failed to load or parse PDF. Please try another file.")
            return
    if st.session_state.get('pdf_text'):
        with st.expander("View extracted text"):
            st.write(st.session_state.pdf_text[:1000] + "... (truncated)")
        query = st.text_input("Ask PDF:", key="pdf_query")
        if st.button("Analyze PDF", key="analyze_pdf_btn") and query:
            with st.spinner("Analyzing..."):
                resp = "".join(
                    agent.analyze_pdf(query, st.session_state.pdf_text)
                )
            st.markdown("**Analysis:**")
            st.write(resp)

def settings_tab(agent: Agent) -> None:
    st.header("Settings & Tools")
    st.subheader("Commands")
    st.markdown(
        "`open safari`, `open app <name>`, `go to <url>`, `fetch data from <url>`,\n"
        "`fill pdf with field=val,...`, `schedule <task>`"
    )
    st.subheader("Model Parameters")
    default_ctx = st.session_state.get("settings_n_ctx", DEFAULT_CTX)
    new_ctx = st.slider("Context tokens (n_ctx)", 64, 2048, default_ctx, key="settings_n_ctx")
    if st.button("Apply Settings", key="apply_settings"):
        # Clamp and rebuild agent
        ctx = max(64, min(new_ctx, 2048))
        logger.info("Applying new context window setting: %d", ctx)
        old = agent
        new_agent = Agent(
            model_path=old.llm.model_path,
            n_ctx=ctx,
            seed=old.llm.seed,
            keep_last=old.keep_last
        )
        new_agent.history = old.history
        st.session_state.agent = new_agent
        st.success(f"Context window set to {ctx} tokens.")

def main() -> None:
    st.set_page_config(
        page_title="SLM Chat-Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Hide default Streamlit style and inject enhanced custom styling
    st.markdown(
        """
        <style>
        /* Hide default menu, header, footer */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* Adjust main content padding */
        .block-container {padding: 1rem 2rem !important;}

        /* Style form container */
        .stForm {background: #f9f9f9; padding: 1rem; border-radius: 8px;}

        /* Style buttons */
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 0.6rem 1rem;
            font-size: 1rem;
            border-radius: 5px;
            transition: background-color 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #45a049;
            cursor: pointer;
        }

        /* Style text inputs */
        .stTextInput > div > div > input {
            border-radius: 5px;
            padding: 0.5rem;
            border: 1px solid #ccc;
        }

        /* Improve expander header */
        .streamlit-expanderHeader {
            font-weight: bold;
            font-size: 1.1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    whisper_model = load_whisper_model()
    if 'agent' not in st.session_state:
        st.session_state.agent = get_agent(DEFAULT_CTX)
    agent = st.session_state.agent

    st.image(LOGO_PATH, width=120)
    st.title("🦙 Multi-Modal Assistant")

    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 PDF Tools", "⚙️ Settings"])

    with tab1:
        chat_tab(agent, whisper_model)
    with tab2:
        pdf_tab(agent)
    with tab3:
        settings_tab(agent)

if __name__ == "__main__":
    main()