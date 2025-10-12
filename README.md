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

git clone https://github.com/6amdev/personal-ai-assistant.git
cd personal-ai-assistant
pip install -r requirements.txt
ollama pull llama3.1:8b
streamlit run app.py

เปิด browser ที่ `http://localhost:8501`

## 📖 Usage

**Chat:** พิมพ์คำถาม → AI ตอบ  
**Upload:** Sidebar → 📤 Upload Documents → เลือกไฟล์  
**Manage:** 📄 View Documents → ลบได้ทีละไฟล์

## 🛠️ Tech Stack

Streamlit • LangChain • Llama 3.1 8B • Ollama • ChromaDB • Python 3.11

## 📊 Performance

| Task | Time |
|------|------|
| Chat | 3-8s |
| Upload (1 file) | 5-10s |
| Search | <1s |

*Tested: RTX 4070 Ti 12GB*

## 🐛 Troubleshooting

**Ollama not found:**
ollama --version

**Model not found:**
ollama pull llama3.1:8b

**Import error:**
pip install -r requirements.txt

## 🤝 Contributing

Fork → Branch → Commit → Push → Pull Request

## 📝 License

MIT License

## 📧 Contact

**6amdev** • [GitHub](https://github.com/6amdev) • [Issues](https://github.com/6amdev/personal-ai-assistant/issues)

---

**Made with ❤️ by [6amdev](https://github.com/6amdev)**

*Give it a star if you like it!* ⭐