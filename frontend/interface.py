import streamlit as st
import requests

st.set_page_config(page_title="🧠 RAG Chatbot", layout="wide")

# — Sidebar: Upload & index documents —
st.sidebar.header("📄 Upload Documents")
uploaded_file = st.sidebar.file_uploader("Upload PDF", type="pdf")
if uploaded_file:
    with st.sidebar:
        with st.spinner("🔄 Embedding & indexing..."):
            res = requests.post(
                "http://127.0.0.1:8000/upload/",
                files={"file": uploaded_file.getvalue()},
            )
        if res.status_code == 200:
            st.success("✅ Document processed!")
        else:
            st.error("❌ Upload failed.")

# — Initialize chat history —
if "messages" not in st.session_state:
    # Each entry: {"role": "user"/"assistant", "content": str, "source": str (assistant only)}
    st.session_state.messages = []

# — Chat Input —
user_input = st.chat_input("Enter your question…")
if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call backend
    with st.spinner("🤔 Thinking…"):
        resp = requests.post("http://127.0.0.1:8000/ask/", params={"query": user_input})
        data = resp.json()
        answer = data.get("answer", "No answer.")
        source = data.get("source", "")
    # Append assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "source": source
    })

# — Render the chat messages in order —
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if msg.get("source"):
                st.caption(f"📄 Source: {msg['source']}")
