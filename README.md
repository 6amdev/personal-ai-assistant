💬 Personal AI Assistant

AI Chatbot ส่วนตัวที่เรียนรู้จากเอกสารของคุณ - ฟรี 100% รันบนเครื่องเอง!

Show Image
Show Image
Show Image
Show Image
✨ Features

🤖 Local LLM - Llama 3.1 8B (via Ollama)
💾 Persistent Memory - จำบทสนทนาและเอกสารถาวร
📚 RAG Support - ตอบคำถามจากเอกสาร
🇹🇭 Thai Language - รองรับภาษาไทย
🔒 100% Private - ข้อมูลไม่ออกจากเครื่อง
🆓 Free - Open source ไม่มีค่าใช้จ่าย
📄 Multi-format - TXT, PDF, DOCX, MD

🚀 Quick Start
Installation
bashgit clone https://github.com/6amdev/personal-ai-assistant.git
cd personal-ai-assistant
pip install -r requirements.txt
ollama pull llama3.1:8b
streamlit run app.py
เปิด browser ที่ http://localhost:8501
📖 Usage
Chat: พิมพ์คำถาม → AI ตอบ
Upload: Sidebar → 📤 Upload Documents → เลือกไฟล์
Manage: 📄 View Documents → ลบได้ทีละไฟล์
🛠️ Tech Stack
Core Technologies
TechnologyPurposeWhy We Use ItStreamlitWeb UI Frameworkสร้าง web app ด้วย Python ง่ายๆ ไม่ต้องเขียน HTML/CSSLangChainLLM Frameworkเชื่อมต่อ LLM, Memory, และ Tools เข้าด้วยกันLlama 3.1 8BLanguage ModelOpen source LLM จาก Meta ฉลาดและรองรับภาษาไทยOllamaLLM Runtimeรัน LLM บนเครื่องเราได้ ไม่ต้องใช้ APIChromaDBVector Databaseเก็บ embeddings สำหรับค้นหาเอกสารที่เกี่ยวข้องPython 3.11Backend Languageภาษาหลักของโปรเจกต์
Document Processing
LibraryPurposepypdfอ่านไฟล์ PDFpython-docxอ่านไฟล์ DOCX (Word)unstructuredประมวลผลเอกสารหลายรูปแบบ
AI Components
RAG (Retrieval-Augmented Generation):

แปลงเอกสารเป็น embeddings (ตัวเลข)
ค้นหาข้อมูลที่เกี่ยวข้องกับคำถาม
ส่งข้อมูลให้ LLM อ่านและตอบ

Embeddings:

ใช้ Llama 3.1 แปลงข้อความเป็น vectors
เก็บใน ChromaDB สำหรับการค้นหาแบบ semantic

Memory:

Short-term: จำบทสนทนาปัจจุบัน (ConversationBufferMemory)
Long-term: จำเอกสารถาวร (ChromaDB)

🏗️ Architecture
User Interface (Streamlit)
    ↓
Application Layer (LangChain)
    ↓
┌───────────────┬──────────────┐
│               │              │
Memory      LLM (Llama 3.1)    │
(ChromaDB)   (via Ollama)      │

📊 Performance
TaskTimeChat Response3-8sUpload (1 file)5-10sUpload (5 files)15-30sSemantic Search<1s
Tested: RTX 4070 Ti 12GB
🐛 Troubleshooting
Ollama not found:
bashollama --version
Model not found:
bashollama pull llama3.1:8b
Import error:
bashpip install -r requirements.txt
Slow performance:

ใช้ GPU (ติดตั้ง CUDA drivers)
ลด chunk size ใน document_processor.py
ใช้ model เล็กกว่า เช่น llama3.1:3b

🤝 Contributing
Fork → Branch → Commit → Push → Pull Request
📝 License
MIT License
🙏 Acknowledgments
Special thanks to:

Meta AI for Llama 3.1
Ollama team for local LLM runtime
LangChain for the amazing framework
Streamlit for the beautiful UI
ChromaDB for vector storage

📧 Contact
6amdev • GitHub • Issues

Made with ❤️ by 6amdev
Give it a star if you like it! ⭐