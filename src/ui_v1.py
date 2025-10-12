"""UI Components"""
import streamlit as st
from config import APP_TITLE, APP_DESCRIPTION, LLM_MODEL

def setup_page():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="💬",
        layout="wide"
    )

def show_header():
    st.title(APP_TITLE)
    st.markdown(APP_DESCRIPTION)
    st.markdown("---")

def show_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        st.subheader("🤖 Model")
        st.info(f"Using: {LLM_MODEL}")
        st.subheader("📝 Options")
        clear_chat = st.button("🗑️ Clear Chat")
        st.markdown("---")
        st.markdown("""
        **Personal AI Assistant**
        
        - 🤖 Local LLM (Ollama)
        - 💾 Persistent Memory
        - 📚 RAG Support
        - 🔒 100% Private
        
        [GitHub](https://github.com/your-username/personal-ai-assistant)
        """)
        return clear_chat

def chat_interface(llm_handler):  # ← ต้องมี parameter นี้!
    """Main chat interface with real LLM"""
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("พิมพ์คำถาม..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response (ใช้ LLM จริงๆ!)
        with st.chat_message("assistant"):
            with st.spinner("🤔 กำลังคิด..."):
                response = llm_handler.generate(prompt)  # ← ใช้ LLM!
                st.markdown(response)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": response})

def clear_chat_history():
    """Clear chat history"""
    st.session_state.messages = []
    st.success("✅ ล้างประวัติการสนทนาแล้ว!")