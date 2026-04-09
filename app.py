import streamlit as st
import asyncio

import ingest
import search_agent
import logs

# --- CONFIG ---
REPO_OWNER = "huggingface"
REPO_NAME = "pytorch-image-models"

# --- CACHE INITIALIZATION (runs once) ---
@st.cache_resource
def initialize_system():
    st.write("🔄 Initializing index...")

    pytorch_img_index, pytorch_img_vindex = ingest.index_data(
        REPO_OWNER, REPO_NAME, chunk=True
    )

    agent = search_agent.init_agent(
        pytorch_img_index,
        pytorch_img_vindex,
        REPO_OWNER,
        REPO_NAME,
    )

    return agent


# --- INIT ---
st.set_page_config(page_title="AI Repo Assistant", layout="wide")
st.title(f"🤖 AI Assistant for {REPO_OWNER}/{REPO_NAME}")

agent = initialize_system()

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- DISPLAY CHAT HISTORY ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- USER INPUT ---
user_input = st.chat_input("Ask a question about the repo...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # --- MODEL RESPONSE ---
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        async def run_agent():
            # simple (non-streaming)
            result = await agent.run(user_prompt=user_input)
            return result

        # Run async
        result = asyncio.run(run_agent())

        response_text = result.output

        # Log interaction (your existing logic)
        logs.log_interaction_to_file(agent, result.new_messages())

        response_placeholder.markdown(response_text)

    # Save assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )