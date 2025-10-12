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

def show_sidebar(memory_handler):
    """Show sidebar with document management"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model info
        st.subheader("🤖 Model")
        st.info(f"Using: {LLM_MODEL}")
        
        # Document stats
        st.subheader("📚 Knowledge Base")
        doc_count = memory_handler.count_documents()
        st.metric("Total Chunks", doc_count)
        
        # Show document list
        sources = memory_handler.get_all_sources()
        docs_to_delete = []  # Initialize here
        
        if sources:
            st.write(f"**Documents ({len(sources)}):**")
            
            # Create expander for document list
            with st.expander("📄 View Documents", expanded=False):
                for source in sources:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.text(f"📄 {source}")
                    with col2:
                        if st.button("🗑️", key=f"delete_{source}", help="ลบเอกสารนี้"):
                            docs_to_delete.append(source)
        else:
            st.info("ยังไม่มีเอกสาร")
        
        # File uploader
        st.subheader("📤 Upload Documents")
        uploaded_files = st.file_uploader(
            "เลือกไฟล์",
            type=['txt', 'pdf', 'docx', 'md'],
            accept_multiple_files=True,
            help="รองรับ: TXT, PDF, DOCX, MD"
        )
        
        # Options
        st.subheader("📝 Options")
        clear_chat = st.button("🗑️ Clear Chat")
        clear_memory = st.button("🗑️ Clear All Documents", 
                                help="ลบเอกสารทั้งหมด",
                                type="secondary")
        
        # About
        st.markdown("---")
        st.markdown("""
        **Personal AI Assistant**
        
        - 🤖 Local LLM (Ollama)
        - 💾 Persistent Memory
        - 📚 RAG Support
        - 🔒 100% Private
        
        [GitHub](https://github.com/6amdev/personal-ai-assistant)
        """)
        
        # Return 4 values!
        return clear_chat, clear_memory, uploaded_files, docs_to_delete

def chat_interface(llm_handler, memory_handler):
    """Main chat interface with RAG"""
    
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
        
        # Get AI response with RAG
        with st.chat_message("assistant"):
            with st.spinner("🤔 กำลังคิด..."):
                # Get context from documents
                context = memory_handler.get_context(prompt, k=3)
                
                # Create full prompt
                if context:
                    full_prompt = f"""{context}

คำถาม: {prompt}

กรุณาตอบโดยอิงจากข้อมูลที่เกี่ยวข้องข้างต้น (ถ้ามี) ถ้าข้อมูลไม่เกี่ยวข้อง ให้ตอบตามความรู้ทั่วไป"""
                else:
                    full_prompt = prompt
                
                # Generate response
                response = llm_handler.generate(full_prompt)
                st.markdown(response)
                
                # Show context used
                if context:
                    with st.expander("📚 ข้อมูลอ้างอิง"):
                        st.text(context)
        
        # Add to history
        st.session_state.messages.append({"role": "assistant", "content": response})

def clear_chat_history():
    """Clear chat history"""
    st.session_state.messages = []
    st.success("✅ ล้างประวัติการสนทนาแล้ว!")