"""Document Processor - แยกข้อความจากเอกสาร"""
import os
import json
from typing import List
from pathlib import Path

# PDF
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# DOCX
try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


class DocumentProcessor:
    """ประมวลผลเอกสารหลายประเภท"""
    
    SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md', '.json']  # เพิ่ม .json
    
    @staticmethod
    def process_file(file_path: str) -> List[str]:
        """
        ประมวลผลไฟล์และแยกข้อความ
        
        Args:
            file_path: path ของไฟล์
            
        Returns:
            List ของข้อความที่แยกได้
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        if extension not in DocumentProcessor.SUPPORTED_EXTENSIONS:
            raise ValueError(f"ไม่รองรับไฟล์ {extension}")
        
        if extension == '.txt' or extension == '.md':
            return DocumentProcessor._process_txt(file_path)
        elif extension == '.pdf':
            return DocumentProcessor._process_pdf(file_path)
        elif extension == '.docx':
            return DocumentProcessor._process_docx(file_path)
        elif extension == '.json':  # เพิ่ม JSON
            return DocumentProcessor._process_json(file_path)
    
    @staticmethod
    def _process_txt(file_path: Path) -> List[str]:
        """อ่านไฟล์ text"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = DocumentProcessor._split_text(content, chunk_size=500)
        return chunks
    
    @staticmethod
    def _process_pdf(file_path: Path) -> List[str]:
        """อ่านไฟล์ PDF"""
        if not PdfReader:
            raise ImportError("ติดตั้ง pypdf: pip install pypdf")
        
        reader = PdfReader(file_path)
        text = ""
        
        for page in reader.pages:
            text += page.extract_text() + "\n"
        
        chunks = DocumentProcessor._split_text(text, chunk_size=500)
        return chunks
    
    @staticmethod
    def _process_docx(file_path: Path) -> List[str]:
        """อ่านไฟล์ DOCX"""
        if not DocxDocument:
            raise ImportError("ติดตั้ง python-docx: pip install python-docx")
        
        doc = DocxDocument(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        chunks = DocumentProcessor._split_text(text, chunk_size=500)
        return chunks
    
    @staticmethod
    def _process_json(file_path: Path) -> List[str]:
        """
        อ่านไฟล์ JSON และแปลงเป็น text ที่อ่านง่าย
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of text chunks
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # แปลง JSON เป็น text ที่อ่านง่าย
            text = DocumentProcessor._json_to_text(data)
            
            # แบ่งเป็น chunks
            chunks = DocumentProcessor._split_text(text, chunk_size=500)
            return chunks
            
        except Exception as e:
            print(f"Error reading JSON: {e}")
            return []
    
    @staticmethod
    def _json_to_text(data, indent=0) -> str:
        """
        แปลง JSON เป็น human-readable text
        
        Args:
            data: JSON data (dict/list/primitive)
            indent: ระดับการเยื้อง
            
        Returns:
            Formatted text
        """
        text = []
        prefix = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    text.append(f"{prefix}{key}:")
                    text.append(DocumentProcessor._json_to_text(value, indent + 1))
                else:
                    text.append(f"{prefix}{key}: {value}")
        
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, (dict, list)):
                    text.append(f"{prefix}[{i}]:")
                    text.append(DocumentProcessor._json_to_text(item, indent + 1))
                else:
                    text.append(f"{prefix}- {item}")
        
        else:
            text.append(f"{prefix}{data}")
        
        return "\n".join(text)
    
    @staticmethod
    def _split_text(text: str, chunk_size: int = 500) -> List[str]:
        """
        แบ่งข้อความเป็น chunks
        
        Args:
            text: ข้อความ
            chunk_size: ขนาดของแต่ละ chunk
            
        Returns:
            List of chunks
        """
        # ลบช่องว่างเกิน
        text = " ".join(text.split())
        
        if not text.strip():
            return []
        
        # แบ่งตาม sentence ก่อน
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # ถ้าเพิ่มประโยคนี้แล้วยาวเกิน chunk_size
            if len(current_chunk) + len(sentence) > chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        # เพิ่ม chunk สุดท้าย
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks


# ทดสอบ
if __name__ == "__main__":
    # ทดสอบ TXT
    test_txt = "test.txt"
    with open(test_txt, 'w', encoding='utf-8') as f:
        f.write("สวัสดีครับ! ผมชื่อ John Doe มีประสบการณ์ 5 ปี")
    
    # ทดสอบ JSON
    test_json = "test.json"
    with open(test_json, 'w', encoding='utf-8') as f:
        json.dump({
            "name": "John Doe",
            "skills": ["Python", "AI"],
            "experience": {"years": 5}
        }, f, ensure_ascii=False, indent=2)
    
    processor = DocumentProcessor()
    
    print("=== TXT ===")
    txt_chunks = processor.process_file(test_txt)
    print(f"Chunks: {len(txt_chunks)}")
    print(txt_chunks[0])
    
    print("\n=== JSON ===")
    json_chunks = processor.process_file(test_json)
    print(f"Chunks: {len(json_chunks)}")
    print(json_chunks[0])
    
    # ลบไฟล์
    os.remove(test_txt)
    os.remove(test_json)