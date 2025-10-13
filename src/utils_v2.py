"""Utilities - Helper functions"""
import re
import base64
from pathlib import Path
from io import BytesIO
from typing import List, Dict
from PIL import Image


class ImageHandler:
    """จัดการรูปภาพทุกรูปแบบ"""
    
    @staticmethod
    def extract_images_from_context(context: str) -> List[Dict]:
        """
        ดึงข้อมูลรูปภาพจาก context
        
        Args:
            context: ข้อความจาก RAG
            
        Returns:
            List of image info [{"type": "url/local/base64", "data": "...", "caption": "..."}]
        """
        images = []
        
        # 1. หา URL รูปภาพ (รองรับ query string และ Apple CDN)
        # Pattern 1: มี extension
        url_pattern1 = r'https?://[^\s\)]+\.(?:jpg|jpeg|png|gif|webp|svg)[^\s\)]*'
        urls = re.findall(url_pattern1, context, re.IGNORECASE)
        
        # Pattern 2: Apple CDN (fmt=jpeg)
        url_pattern2 = r'https?://[^\s\)]+fmt=(?:jpg|jpeg|png|gif|webp)[^\s\)]*'
        urls.extend(re.findall(url_pattern2, context, re.IGNORECASE))
        
        # Pattern 3: มี "image" หรือ "storeimages" ใน URL
        url_pattern3 = r'https?://[^\s\)]+(?:image|storeimages)[^\s\)]*'
        urls.extend(re.findall(url_pattern3, context, re.IGNORECASE))
        
        for url in urls:
            # ลบ punctuation ท้าย URL
            url = url.rstrip('.,;:!?)')
            images.append({
                "type": "url",
                "data": url.strip(),
                "caption": ImageHandler._extract_caption_near(context, url)
            })
        
        # 2. หา Local file paths
        local_pattern = r'(?:data/images/|./data/images/|images/)[^\s]+\.(?:jpg|jpeg|png|gif|webp)'
        paths = re.findall(local_pattern, context, re.IGNORECASE)
        for path in paths:
            if Path(path).exists():
                images.append({
                    "type": "local",
                    "data": path.strip(),
                    "caption": ImageHandler._extract_caption_near(context, path)
                })
        
        # 3. หา Base64 images
        base64_pattern = r'image_base64:\s*([A-Za-z0-9+/=]{100,})'
        base64_matches = re.findall(base64_pattern, context)
        for b64 in base64_matches:
            images.append({
                "type": "base64",
                "data": b64.strip(),
                "caption": "Base64 Image"
            })
        
        # 4. หา image_url field (JSON format)
        if 'image_url:' in context or 'image:' in context:
            lines = context.split('\n')
            for line in lines:
                if 'image_url:' in line.lower():
                    # หา URL หลัง image_url:
                    url_match = re.search(r'image_url:\s*(https?://[^\s\)]+)', line, re.IGNORECASE)
                    if url_match:
                        url = url_match.group(1).rstrip('.,;:!?)')
                        images.append({
                            "type": "url",
                            "data": url,
                            "caption": ImageHandler._extract_caption_from_line(line)
                        })
                elif 'image:' in line.lower() and not 'image_url' in line.lower():
                    path = line.split(':', 1)[1].strip()
                    if Path(path).exists():
                        images.append({
                            "type": "local",
                            "data": path,
                            "caption": ImageHandler._extract_caption_from_line(line)
                        })
        
        # ลบ duplicate
        seen = set()
        unique_images = []
        for img in images:
            key = f"{img['type']}:{img['data']}"
            if key not in seen:
                seen.add(key)
                unique_images.append(img)
        
        return unique_images
    
    @staticmethod
    def _extract_caption_near(context: str, image_ref: str, radius: int = 100) -> str:
        """หา caption ใกล้ๆ รูปภาพ"""
        try:
            idx = context.find(image_ref)
            if idx == -1:
                return ""
            
            # ดูข้อความข้างหน้า
            start = max(0, idx - radius)
            before = context[start:idx].strip()
            
            # หาบรรทัดที่มี name หรือ description
            lines = before.split('\n')
            for line in reversed(lines):
                if 'name:' in line.lower():
                    return line.split(':', 1)[1].strip()
                elif 'description:' in line.lower():
                    return line.split(':', 1)[1].strip()
            
            return ""
        except:
            return ""
    
    @staticmethod
    def _extract_caption_from_line(line: str) -> str:
        """หา caption จากบรรทัดเดียวกัน"""
        try:
            # ดูว่ามี name หรือ description ในบรรทัดเดียวกันไหม
            if 'name:' in line.lower():
                parts = line.split('name:', 1)
                if len(parts) > 1:
                    return parts[1].split(',')[0].strip()
            return ""
        except:
            return ""
    
    @staticmethod
    def image_to_base64(image_path: str) -> str:
        """
        แปลงรูปภาพเป็น base64 string
        
        Args:
            image_path: path to image file
            
        Returns:
            base64 encoded string
        """
        try:
            with open(image_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode()
        except Exception as e:
            print(f"Error encoding image: {e}")
            return ""
    
    @staticmethod
    def base64_to_image(base64_str: str):
        """
        แปลง base64 เป็น PIL Image
        
        Args:
            base64_str: base64 encoded string
            
        Returns:
            PIL Image object
        """
        try:
            image_data = base64.b64decode(base64_str)
            return Image.open(BytesIO(image_data))
        except Exception as e:
            print(f"Error decoding base64 image: {e}")
            return None
    
    @staticmethod
    def validate_image_url(url: str) -> bool:
        """
        เช็คว่า URL เป็นรูปภาพจริงไหม
        
        Args:
            url: image URL
            
        Returns:
            True if valid image URL
        """
        try:
            import requests
            response = requests.head(url, timeout=5)
            content_type = response.headers.get('content-type', '')
            return 'image' in content_type.lower()
        except:
            return False


# Helper function สำหรับใช้งานง่ายๆ
def extract_images(context: str) -> List[Dict]:
    """Shortcut function"""
    return ImageHandler.extract_images_from_context(context)