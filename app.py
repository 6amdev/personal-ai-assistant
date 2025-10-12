"""Personal AI Assistant - Main App with RAG"""
import streamlit as st
import tempfile
import os
from pathlib import Path

from src.ui import setup_page, show_header, show_sidebar, chat_interface, clear_chat_history
from src.llm import LLMHandler
from src.memory import MemoryHandler
from src.document_processor import DocumentProcessor


def main():
    setup_page()
    
    @st.cache_resource
    def init_components():
        """Initialize LLM and Memory"""
        llm = LLMHandler()
        memory = MemoryHandler()
        return llm, memory
    
    try:
        llm, memory = init_components()
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        st.info("💡 ตรวจสอบว่า Ollama กำลังทำงานอยู่!")
        st.code("ollama list", language="bash")
        st.stop()
    
    show_header()
    
    # Sidebar
    clear_chat, clear_memory, uploaded_files, docs_to_delete = show_sidebar(memory)
    
    # Handle clear chat
    if clear_chat:
        clear_chat_history()
        memory.clear_conversation()
    
    # Handle delete specific documents
    if docs_to_delete:
        for doc_source in docs_to_delete:
            deleted_count = memory.delete_by_source(doc_source)
            if deleted_count > 0:
                st.success(f"✅ ลบ {doc_source} แล้ว ({deleted_count} chunks)")
                st.rerun()
    
    # Handle clear all documents
    if clear_memory:
        with st.spinner("🗑️ กำลังลบเอกสารทั้งหมด..."):
            success = memory.clear_all_documents()
            
            if success:
                st.success("✅ ลบเอกสารทั้งหมดแล้ว!")
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error("❌ เกิดข้อผิดพลาดในการลบเอกสาร")
    
    # Handle file uploads
    if uploaded_files:
        process_uploaded_files(uploaded_files, memory)
    
    # Main chat
    chat_interface(llm, memory)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Made with ❤️ using Ollama, LangChain & Streamlit
    </div>
    """, unsafe_allow_html=True)


def process_uploaded_files(uploaded_files, memory):
    """Process and store uploaded files"""
    
    # สร้าง progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    processor = DocumentProcessor()
    total_files = len(uploaded_files)
    total_chunks = 0
    all_chunks = []
    all_metadatas = []
    
    # Phase 1: อ่านไฟล์ทั้งหมด
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"📄 กำลังอ่าน {uploaded_file.name}... ({idx+1}/{total_files})")
            
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Process
            status_text.text(f"📝 แยกข้อความจาก {uploaded_file.name}...")
            chunks = processor.process_file(tmp_path)
            
            # รวม chunks
            all_chunks.extend(chunks)
            all_metadatas.extend([{'source': uploaded_file.name} for _ in chunks])
            
            total_chunks += len(chunks)
            
            # Cleanup
            os.unlink(tmp_path)
            
            # Update progress
            progress_bar.progress((idx + 1) / total_files * 0.5)
            
        except Exception as e:
            st.error(f"❌ Error: {uploaded_file.name}: {e}")
    
    # Phase 2: บันทึกทีเดียว (สำคัญ!) 🔥
    if all_chunks:
        status_text.text(f"🧠 กำลังบันทึก {total_chunks} chunks...")
        progress_bar.progress(0.6)
        
        try:
            # เรียกครั้งเดียว! ไม่ใช่ loop!
            memory.add_documents(all_chunks, all_metadatas)
            
            # เสร็จ
            progress_bar.progress(1.0)
            status_text.empty()
            
            st.success(f"✅ เรียนรู้เอกสารสำเร็จ! ({total_chunks} chunks จาก {total_files} ไฟล์)")
            st.balloons()
            st.rerun()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()