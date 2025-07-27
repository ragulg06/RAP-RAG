# app/interface.py
import streamlit as st
import requests
import time
import json
from typing import Dict, Any

# Page configuration
st.set_page_config(
    page_title="🧠 Advanced RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    text-align: center;
    margin-bottom: 2rem;
    background: linear-gradient(90deg, #4CAF50, #2196F3);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.stats-container {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 1rem 0;
}

.confidence-high { color: #4CAF50; font-weight: bold; }
.confidence-medium { color: #FF9800; font-weight: bold; }
.confidence-low { color: #f44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🧠 Advanced RAG Chatbot</h1>', unsafe_allow_html=True)

# Sidebar for document upload and settings
with st.sidebar:
    st.header("📄 Document Management")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Document", 
        type=["pdf", "docx"],
        help="Upload PDF or Word documents for the chatbot to use as knowledge base"
    )
    
    if uploaded_file:
        with st.spinner("🔄 Processing document..."):
            files = {"file": uploaded_file.getvalue()}
            response = requests.post("http://127.0.0.1:8000/upload/", files=files)
            
            if response.status_code == 200:
                data = response.json()
                st.success("✅ Document processed successfully!")
                st.info(f"📊 {data['chunks_count']} chunks indexed")
                
                # Show GPU usage if available
                if "gpu_usage" in data and data["gpu_usage"]["gpu_available"]:
                    with st.expander("🖥️ GPU Usage"):
                        gpu_data = data["gpu_usage"]
                        st.metric("Memory Allocated", f"{gpu_data['memory_allocated']:.1f} MB")
                        st.metric("Memory Cached", f"{gpu_data['memory_cached']:.1f} MB")
            else:
                st.error("❌ Failed to process document")
    
    st.divider()
    
    # Advanced settings
    st.header("⚙️ Settings")
    show_confidence = st.checkbox("Show Confidence Scores", value=True)
    show_timing = st.checkbox("Show Response Time", value=True)
    show_sources = st.checkbox("Show Source Details", value=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.conversation_stats = {
        "total_queries": 0,
        "avg_response_time": 0,
        "avg_confidence": 0
    }

# Main chat interface
st.header("💬 Chat Interface")

# Display conversation statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Queries", st.session_state.conversation_stats["total_queries"])
with col2:
    st.metric("Avg Response Time", f"{st.session_state.conversation_stats['avg_response_time']:.2f}s")
with col3:
    st.metric("Avg Confidence", f"{st.session_state.conversation_stats['avg_confidence']:.2f}")

# Chat input
user_input = st.chat_input("Ask me anything about the uploaded documents...")

if user_input:
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": time.time()
    })
    
    # Make API call
    with st.spinner("🤔 Thinking..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/ask/",
                json={"query": user_input},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "sources": data.get("sources", ""),
                    "confidence": data.get("confidence", 0),
                    "response_time": data.get("response_time", 0),
                    "timestamp": time.time()
                })
                
                # Update conversation statistics
                stats = st.session_state.conversation_stats
                stats["total_queries"] += 1
                stats["avg_response_time"] = (
                    (stats["avg_response_time"] * (stats["total_queries"] - 1) + data.get("response_time", 0)) 
                    / stats["total_queries"]
                )
                stats["avg_confidence"] = (
                    (stats["avg_confidence"] * (stats["total_queries"] - 1) + data.get("confidence", 0))
                    / stats["total_queries"]
                )
                
            else:
                st.error(f"Error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Please ensure the backend server is running.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            # Only show the answer
            st.markdown(f"**Answer:**\n{message['content']}")
            # Show sources if enabled
            if show_sources and message.get("sources"):
                st.caption("📄 **Sources:**")
                # Display sources as they come from backend (already deduplicated and formatted)
                st.caption(message["sources"])
            # Show confidence and timing if enabled
            if show_confidence:
                conf = message.get("confidence", 0)
                if conf >= 0.8:
                    conf_class = "confidence-high"
                elif conf >= 0.6:
                    conf_class = "confidence-medium"
                else:
                    conf_class = "confidence-low"
                st.markdown(f'<span class="{conf_class}">Confidence: {conf:.2f}</span>', unsafe_allow_html=True)
            if show_timing and message.get("response_time"):
                st.caption(f"⏱️ {message['response_time']:.2f}s")

# Footer with system information
with st.expander("🔧 System Information"):
    try:
        health_response = requests.get("http://127.0.0.1:8000/health/")
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.json(health_data)
        else:
            st.error("Backend health check failed")
    except:
        st.warning("Backend not accessible")
