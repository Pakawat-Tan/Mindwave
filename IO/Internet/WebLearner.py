class WebLearner:
    def __init__(self, fetcher, parser, extractor, source_profiler):
        self.fetcher = fetcher
        self.parser = parser
        self.extractor = extractor
        self.source_profiler = source_profiler

    def fetch(self, query: str) -> list[dict]:
        pages = self.fetcher.search(query)

        docs = []
        for page in pages:
            text = self.parser.parse(page["html"])
            facts = self.extractor.extract(text)
            trust = self.source_profiler.score(page["url"])

            docs.append({
                "url": page["url"],
                "trust": trust,
                "knowledge": facts
            })

        return docs