from .BaseReader import BaseReader

class TextReader(BaseReader):

    def read(self, file_path: str):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        lines = [l.strip() for l in content.splitlines() if l.strip()]

        return {
            "source": "file",
            "file_type": "text",
            "file_path": file_path,
            "content": content,
            "structure": [
                {"line": i, "text": l} for i, l in enumerate(lines)
            ],
            "metadata": {
                "num_lines": len(lines),
                "length": len(content)
            }
        }
