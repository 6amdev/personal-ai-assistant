"""Personal AI Assistant - Main App"""
import streamlit as st
from src.ui import setup_page, show_header, show_sidebar, chat_interface, clear_chat_history
from src.llm import LLMHandler
from src.memory import MemoryHandler

def main():
    setup_page()
    
    @st.cache_resource
    def init_components():
        """Initialize LLM and Memory (cached)"""
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
    clear_chat = show_sidebar()
    
    if clear_chat:
        clear_chat_history()
        memory.clear_conversation()
    
    # ส่ง llm_handler เข้าไปใน chat_interface! 🔥
    chat_interface(llm)  # ← เปลี่ยนตรงนี้!
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Made with ❤️ using Ollama, LangChain & Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()