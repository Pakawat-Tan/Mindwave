class InternetGateway:
    def __init__(self, web_learner, reviewer, rule_engine, memory_engine, learner):
        self.web_learner = web_learner
        self.reviewer = reviewer
        self.rule_engine = rule_engine
        self.memory_engine = memory_engine
        self.learner = learner

    def learn(self, query: str):
        raw_docs = self.web_learner.fetch(query)

        accepted_knowledge = []

        for doc in raw_docs:
            if not self.rule_engine.allow("internet_content", doc):
                continue

            score = self.reviewer.evaluate(doc)
            if score < 0.6:
                continue

            knowledge = doc["knowledge"]
            self.memory_engine.store_long_term(knowledge)
            accepted_knowledge.append(knowledge)

        self.learner.learn_from_external(accepted_knowledge)

        return {
            "query": query,
            "fetched": len(raw_docs),
            "accepted": len(accepted_knowledge)
        }
