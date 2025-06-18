from PIL import Image
import pytesseract
from langchain.schema import Document

def load_images_with_ocr(path):
    img = Image.open(path)
    text = pytesseract.image_to_string(img)
    return [Document(page_content=text, metadata={"source": path})]
