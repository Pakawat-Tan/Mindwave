class FactExtractor:
    def extract(self, text: str) -> dict:
        return {
            "facts": self._extract_facts(text),
            "concepts": self._extract_concepts(text),
            "relations": self._extract_relations(text),
            "source_type": "internet"
        }
