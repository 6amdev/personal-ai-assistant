# 💬 Personal AI Assistant

> AI Chatbot ส่วนตัวที่เรียนรู้จากเอกสารของคุณ - ฟรี 100% รันบนเครื่องเอง!

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/ollama-required-orange.svg)](https://ollama.com)
[![Made by](https://img.shields.io/badge/made%20by-6amdev-red.svg)](https://github.com/6amdev)

## ✨ Features

- 🤖 **Local LLM** - Llama 3.1 8B (via Ollama)
- 💾 **Persistent Memory** - จำบทสนทนาและเอกสารถาวร
- 📚 **RAG Support** - ตอบคำถามจากเอกสาร
- 🇹🇭 **Thai Language** - รองรับภาษาไทย
- 🔒 **100% Private** - ข้อมูลไม่ออกจากเครื่อง
- 🆓 **Free** - Open source ไม่มีค่าใช้จ่าย
- 📄 **Multi-format** - TXT, PDF, DOCX, MD

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/6amdev/personal-ai-assistant.git
cd personal-ai-assistant
pip install -r requirements.txt
ollama pull llama3.1:8b
streamlit run app.py
```

เปิด browser ที่ `http://localhost:8501`

## 📖 Usage

**Chat:** พิมพ์คำถาม → AI ตอบ  
**Upload:** Sidebar → 📤 Upload Documents → เลือกไฟล์  
**Manage:** 📄 View Documents → ลบได้ทีละไฟล์

## 🛠️ Tech Stack

### Core Technologies

| Technology | Purpose | Why We Use It |
|-----------|---------|---------------|
| **[Streamlit](https://streamlit.io)** | Web UI Framework | สร้าง web app ด้วย Python ง่ายๆ ไม่ต้องเขียน HTML/CSS |
| **[LangChain](https://langchain.com)** | LLM Framework | เชื่อมต่อ LLM, Memory, และ Tools เข้าด้วยกัน |
| **[Llama 3.1 8B](https://ai.meta.com/llama/)** | Language Model | Open source LLM จาก Meta ฉลาดและรองรับภาษาไทย |
| **[Ollama](https://ollama.com)** | LLM Runtime | รัน LLM บนเครื่องเราได้ ไม่ต้องใช้ API |
| **[ChromaDB](https://trychroma.com)** | Vector Database | เก็บ embeddings สำหรับค้นหาเอกสารที่เกี่ยวข้อง |
| **Python 3.11** | Backend Language | ภาษาหลักของโปรเจกต์ |

### Document Processing

| Library | Purpose |
|---------|---------|
| **pypdf** | อ่านไฟล์ PDF |
| **python-docx** | อ่านไฟล์ DOCX (Word) |
| **unstructured** | ประมวลผลเอกสารหลายรูปแบบ |

### AI Components

**RAG (Retrieval-Augmented Generation):**
- แปลงเอกสารเป็น embeddings (ตัวเลข)
- ค้นหาข้อมูลที่เกี่ยวข้องกับคำถาม
- ส่งข้อมูลให้ LLM อ่านและตอบ

**Embeddings:**
- ใช้ Llama 3.1 แปลงข้อความเป็น vectors
- เก็บใน ChromaDB สำหรับการค้นหาแบบ semantic

**Memory:**
- Short-term: จำบทสนทนาปัจจุบัน (ConversationBufferMemory)
- Long-term: จำเอกสารถาวร (ChromaDB)

## 🏗️ Architecture

```
User Interface (Streamlit)
    ↓
Application Layer (LangChain)
    ↓
┌───────────────┬──────────────┐
│               │              │
 Memory      LLM (Llama 3.1)   │
 (ChromaDB)   (via Ollama)     │
```

## 📊 Performance

| Task | Time |
|------|------|
| Chat Response    | 3-8s   |
| Upload (1 file)  | 5-10s  |
| Upload (5 files) | 15-30s |
| Semantic Search  | <1s    |

*Tested: RTX 4070 Ti 12GB*

## 🐛 Troubleshooting

**Ollama not found:**
```bash
ollama --version
```

**Model not found:**
```bash
ollama pull llama3.1:8b
```

**Import error:**
```bash
pip install -r requirements.txt
```

**Slow performance:**
- ใช้ GPU (ติดตั้ง CUDA drivers)
- ลด chunk size ใน `document_processor.py`
- ใช้ model เล็กกว่า เช่น `llama3.1:3b`

## 🤝 Contributing

Fork → Branch → Commit → Push → Pull Request

## 📝 License

MIT License

## 🙏 Acknowledgments

Special thanks to:
- Meta AI for Llama 3.1
- Ollama team for local LLM runtime
- LangChain for the amazing framework
- Streamlit for the beautiful UI
- ChromaDB for vector storage

## 📧 Contact

**6amdev** • [GitHub](https://github.com/6amdev) • [Issues](https://github.com/6amdev/personal-ai-assistant/issues)

---

**Made with ❤️ by [6amdev](https://github.com/6amdev)**

*Give it a star if you like it!* ⭐