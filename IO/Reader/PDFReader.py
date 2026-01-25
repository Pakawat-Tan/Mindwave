import pdfplumber
from .BaseReader import BaseReader

class PDFReader(BaseReader):

    def read(self, file_path: str):
        pages = []
        full_text = []

        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append({
                    "page": i,
                    "text": text
                })
                full_text.append(text)

        combined = "\n".join(full_text)

        return {
            "source": "file",
            "file_type": "pdf",
            "file_path": file_path,
            "content": combined,
            "structure": pages,
            "metadata": {
                "num_pages": len(pages),
                "length": len(combined)
            }
        }
