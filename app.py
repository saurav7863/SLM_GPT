import streamlit as st
from agent import Agent
from faster_whisper import WhisperModel
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import numpy as np
import soundfile as sf
import tempfile
import PyPDF2

# Page configuration
st.set_page_config(
    page_title="Multi-Modal Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Whisper model for speech-to-text
@st.cache_resource
def load_whisper_model():
    return WhisperModel(
        model_size_or_path="tiny",
        device="cpu",
        compute_type="int8"
    )
whisper_model = load_whisper_model()

# Default context window
DEFAULT_CTX = 128

# Initialize Agent if not present
# Initialize Agent with specified model
if 'agent' not in st.session_state:
    st.session_state.agent = Agent(
        model_path="/Users/saurav/Desktop/GPT_LLM/models/Phi-3-mini-4k-instruct-q4.gguf",
        n_ctx=DEFAULT_CTX,
        seed=42,
        keep_last=3
    )

# Display logo
logo_path = "/Users/saurav/Desktop/GPT_LLM/logo.WEBP"
st.image(logo_path, width=120)

st.title("🦙 Multi-Modal Assistant")

# Create UI tabs
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 PDF Tools", "⚙️ Settings"])

# -------- Chat Tab --------
with tab1:
    st.header("Chat")

    # Recording control and input form at top
    if st.button("🎤 Record", key="record_btn"):
        st.session_state.recording = True
    if st.session_state.get('recording', False):
        st.info("Recording... speak then click Send.")

    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Your message:", key="text_input")
        send = st.form_submit_button("Send", use_container_width=True)

    # Container for chat messages
    chat_container = st.container()
    # Display existing history
    with chat_container:
        for msg in st.session_state.agent.history:
            role = "user" if msg['role']=='user' else 'assistant'
            st.chat_message(role).write(msg['content'])

    # Handle new submission
    if send and (user_input or st.session_state.get('recording', False)):
        # Determine message text
        if user_input:
            text = user_input
        else:
            # Capture audio via WebRTC
            webrtc_ctx = webrtc_streamer(
                key="mic",
                mode=WebRtcMode.SENDONLY,
                media_stream_constraints={"audio": True, "video": False},
                audio_receiver_size=8192
            )
            frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            if not frames:
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

        # Append user message
        st.session_state.agent.history.append({'role':'user','content':text})
        chat_container.chat_message("user").write(text)

                # Stream assistant response word-by-word into one placeholder
        placeholder = chat_container.empty()
        full_text = ""
        for token in st.session_state.agent.stream_chat(
            text,
            pdf_text=st.session_state.get('pdf_text')
        ):
            full_text += token
            placeholder.markdown(full_text)
        # After streaming, append final static chat message
        st.session_state.agent.history.append({'role':'assistant','content':full_text})
        chat_container.chat_message("assistant").write(full_text)
        placeholder.empty()

# ------ PDF Tools Tab ------
        st.session_state.agent.history.append({'role':'assistant','content':full_text})

# ------ PDF Tools Tab ------
with tab2:
    st.header("PDF Analysis & Form-Filling")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_upload")
    if pdf_file:
        path = f"/tmp/{pdf_file.name}"
        with open(path,"wb") as f:
            f.write(pdf_file.getbuffer())
        reader = PyPDF2.PdfReader(path)
        pdf_text = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        st.session_state.pdf_path = path
        st.session_state.pdf_text = pdf_text
        st.success("PDF loaded.")
    if st.session_state.get('pdf_text'):
        with st.expander("View extracted text"):
            st.write(st.session_state.pdf_text[:1000] + "... (truncated)")
        query = st.text_input("Ask PDF:", key="pdf_query")
        if st.button("Analyze PDF", key="analyze_pdf_btn") and query:
            with st.spinner("Analyzing..."):
                resp = "".join(
                    st.session_state.agent.analyze_pdf(query, st.session_state.pdf_text)
                )
            st.markdown("**Analysis:**")
            st.write(resp)

# ------ Settings Tab ------
with tab3:
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
        old = st.session_state.agent
        st.session_state.agent = Agent(
            model_path=old.llm.model_path,
            n_ctx=ctx,
            seed=old.llm.seed,
            keep_last=old.keep_last
        )
        st.session_state.agent.history = old.history
        st.success(f"Context window set to {ctx} tokens.")