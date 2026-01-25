import re
from html.parser import HTMLParser
from langdetect import detect

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.texts = []

    def handle_data(self, data):
        text = data.strip()
        if text:
            self.texts.append(text)

class ContentParser:
    def parse(self, raw_html: str) -> dict:
        extractor = _TextExtractor()
        extractor.feed(raw_html)

        full_text = " ".join(extractor.texts)
        cleaned = self._clean_text(full_text)

        language = self._detect_language(cleaned)
        sections = self._split_sections(cleaned)

        return {
            "title": sections[0]["text"] if sections else "",
            "main_text": cleaned,
            "sections": sections,
            "language": language,
            "content_type": self._infer_type(cleaned),
            "signals": {
                "noise_ratio": self._estimate_noise(raw_html),
                "structure_score": len(sections) / max(len(cleaned.split()), 1),
                "length": len(cleaned)
            }
        }

    # -------------------------
    # Internal helpers
    # -------------------------

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"(cookie|privacy policy|subscribe)", "", text, flags=re.I)
        return text.strip()

    def _split_sections(self, text: str) -> list:
        paragraphs = text.split(". ")
        sections = []

        for i, p in enumerate(paragraphs):
            if len(p.split()) < 5:
                continue
            sections.append({
                "id": i,
                "text": p.strip()
            })
        return sections

    def _detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "unknown"

    def _infer_type(self, text: str) -> str:
        if "abstract" in text.lower():
            return "paper"
        if "blog" in text.lower():
            return "blog"
        return "article"

    def _estimate_noise(self, raw_html: str) -> float:
        html_len = len(raw_html)
        text_len = max(len(re.sub(r"<[^>]+>", "", raw_html)), 1)
        return 1.0 - (text_len / html_len)
