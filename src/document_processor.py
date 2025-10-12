"""Document Processor - แยกข้อความจากเอกสาร"""
import os
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
    
    SUPPORTED_EXTENSIONS = ['.txt', '.pdf', '.docx', '.md']
    
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
    
    @staticmethod
    def _process_txt(file_path: Path) -> List[str]:
        """อ่านไฟล์ text"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # แบ่งเป็น chunks (แต่ละ chunk ~500 ตัวอักษร)
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
    # สร้างไฟล์ทดสอบ
    test_file = "test.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("""
        สวัสดีครับ! ผมชื่อ John Doe 
        ผมเป็นนักพัฒนาซอฟต์แวร์
        มีประสบการณ์ 5 ปี
        เชี่ยวชาญ Python, JavaScript, และ AI
        สนใจการพัฒนา AI applications
        """)
    
    processor = DocumentProcessor()
    chunks = processor.process_file(test_file)
    
    print(f"✅ พบ {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(chunk)
    
    # ลบไฟล์ทดสอบ
    os.remove(test_file)