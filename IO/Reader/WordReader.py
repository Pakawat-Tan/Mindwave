from docx import Document
from .BaseReader import BaseReader

class WordReader(BaseReader):

    def read(self, file_path: str):
        doc = Document(file_path)

        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        full_text = "\n".join(paragraphs)

        return {
            "source": "file",
            "file_type": "word",
            "file_path": file_path,
            "content": full_text,
            "structure": [
                {"paragraph": i, "text": p}
                for i, p in enumerate(paragraphs)
            ],
            "metadata": {
                "num_paragraphs": len(paragraphs),
                "length": len(full_text)
            }
        }
