import streamlit as st
from agent import Agent

# Page configuration
st.set_page_config(
    page_title="SLM-Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Default context window
DEFAULT_CTX = 2048

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

st.title("SLM-Assistant")

# Create UI tabs
tab1, tab2 = st.tabs(["💬 Chat", "⚙️ Settings"])

# -------- Chat Tab --------
with tab1:
    st.header("Chat")

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
    if send and user_input:
        # Determine message text
        text = user_input

        # Append user message
        st.session_state.agent.history.append({'role':'user','content':text})
        chat_container.chat_message("user").write(text)

                # Stream assistant response word-by-word into one placeholder
        placeholder = chat_container.empty()
        full_text = ""
        for token in st.session_state.agent.stream_chat(text):
            full_text += token
            placeholder.markdown(full_text)
        # After streaming, append final static chat message
        st.session_state.agent.history.append({'role':'assistant','content':full_text})
        chat_container.chat_message("assistant").write(full_text)
        placeholder.empty()

# ------ Settings Tab ------
with tab2:
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