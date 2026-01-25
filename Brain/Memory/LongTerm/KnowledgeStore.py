"""
KnowledgeStore.py
Stores structured knowledge - facts and concepts (semantic memory).
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import time


@dataclass
class Fact:
    """Represents a stored fact"""
    fact_id: str
    subject: str
    predicate: str
    obj: str
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.8
    source: str = "unknown"
    verified: bool = False
    usage_count: int = 0


@dataclass
class Concept:
    """Represents a stored concept"""
    concept_id: str
    name: str
    definition: str
    timestamp: float = field(default_factory=time.time)
    category: str = ""
    related_concepts: Set[str] = field(default_factory=set)
    confidence: float = 0.8
    usage_count: int = 0


class KnowledgeStore:
    """Stores structured knowledge, facts, and concepts."""
    
    def __init__(self, max_facts: int = 5000, max_concepts: int = 500):
        """Initialize knowledge store.
        
        Parameters
        ----------
        max_facts : int
            Maximum number of facts
        max_concepts : int
            Maximum number of concepts
        """
        self.max_facts = max_facts
        self.max_concepts = max_concepts
        self.facts: Dict[str, Fact] = {}
        self.concepts: Dict[str, Concept] = {}
        self.relationships: Dict[str, List[Tuple[str, str]]] = {}  # concept -> [(relation, concept)]
    
    def add_fact(self, fact_id: str, subject: str, predicate: str, obj: str,
                 confidence: float = 0.8, source: str = "unknown") -> bool:
        """Add a new fact.
        
        Parameters
        ----------
        fact_id : str
            Unique fact ID
        subject : str
            Subject of the fact
        predicate : str
            Predicate/relation
        obj : str
            Object of the fact
        confidence : float
            Confidence in the fact (0.0-1.0)
        source : str
            Source of the fact
            
        Returns
        -------
        bool
            Success
        """
        if len(self.facts) >= self.max_facts:
            self._evict_fact()
        
        fact = Fact(
            fact_id=fact_id,
            subject=subject,
            predicate=predicate,
            obj=obj,
            confidence=confidence,
            source=source
        )
        
        self.facts[fact_id] = fact
        
        return True
    
    def add_concept(self, concept_id: str, name: str, definition: str, 
                   category: str = "", confidence: float = 0.8) -> bool:
        """Add a new concept.
        
        Parameters
        ----------
        concept_id : str
            Unique concept ID
        name : str
            Concept name
        definition : str
            Concept definition
        category : str
            Concept category
        confidence : float
            Confidence in the concept (0.0-1.0)
            
        Returns
        -------
        bool
            Success
        """
        if len(self.concepts) >= self.max_concepts:
            self._evict_concept()
        
        concept = Concept(
            concept_id=concept_id,
            name=name,
            definition=definition,
            category=category,
            confidence=confidence
        )
        
        self.concepts[concept_id] = concept
        if concept_id not in self.relationships:
            self.relationships[concept_id] = []
        
        return True
    
    def relate_concepts(self, concept1_id: str, relation: str, concept2_id: str) -> bool:
        """Create a relationship between concepts.
        
        Parameters
        ----------
        concept1_id : str
            First concept ID
        relation : str
            Relationship type
        concept2_id : str
            Second concept ID
            
        Returns
        -------
        bool
            Success
        """
        if concept1_id not in self.relationships:
            self.relationships[concept1_id] = []
        if concept2_id not in self.relationships:
            self.relationships[concept2_id] = []
        
        if concept1_id in self.concepts:
            self.concepts[concept1_id].related_concepts.add(concept2_id)
        if concept2_id in self.concepts:
            self.concepts[concept2_id].related_concepts.add(concept1_id)
        
        self.relationships[concept1_id].append((relation, concept2_id))
        self.relationships[concept2_id].append((relation, concept1_id))
        
        return True
    
    def query_facts(self, subject: Optional[str] = None, 
                   predicate: Optional[str] = None, 
                   obj: Optional[str] = None) -> List[Fact]:
        """Query facts with optional filters.
        
        Parameters
        ----------
        subject : Optional[str]
            Filter by subject
        predicate : Optional[str]
            Filter by predicate
        obj : Optional[str]
            Filter by object
            
        Returns
        -------
        List[Fact]
            Matching facts
        """
        results = []
        
        for fact in self.facts.values():
            if subject and fact.subject != subject:
                continue
            if predicate and fact.predicate != predicate:
                continue
            if obj and fact.obj != obj:
                continue
            
            results.append(fact)
            fact.usage_count += 1
        
        return results
    
    def query_concepts(self, name: Optional[str] = None, 
                      category: Optional[str] = None) -> List[Concept]:
        """Query concepts.
        
        Parameters
        ----------
        name : Optional[str]
            Filter by name
        category : Optional[str]
            Filter by category
            
        Returns
        -------
        List[Concept]
            Matching concepts
        """
        results = []
        
        for concept in self.concepts.values():
            if name and concept.name != name:
                continue
            if category and concept.category != category:
                continue
            
            results.append(concept)
            concept.usage_count += 1
        
        return results
    
    def get_related_concepts(self, concept_id: str) -> List[Tuple[str, Concept]]:
        """Get concepts related to a given concept.
        
        Parameters
        ----------
        concept_id : str
            Concept ID
            
        Returns
        -------
        List[Tuple[str, Concept]]
            Tuples of (relation_type, concept)
        """
        if concept_id not in self.relationships:
            return []
        
        results = []
        for relation, related_id in self.relationships[concept_id]:
            if related_id in self.concepts:
                results.append((relation, self.concepts[related_id]))
        
        return results
    
    def verify_fact(self, fact_id: str) -> bool:
        """Mark a fact as verified.
        
        Parameters
        ----------
        fact_id : str
            Fact ID
            
        Returns
        -------
        bool
            Success
        """
        if fact_id in self.facts:
            self.facts[fact_id].verified = True
            self.facts[fact_id].confidence = min(1.0, self.facts[fact_id].confidence + 0.1)
            return True
        return False
    
    def update_confidence(self, fact_id: str, confidence: float) -> bool:
        """Update confidence in a fact.
        
        Parameters
        ----------
        fact_id : str
            Fact ID
        confidence : float
            New confidence (0.0-1.0)
            
        Returns
        -------
        bool
            Success
        """
        if fact_id in self.facts:
            self.facts[fact_id].confidence = max(0.0, min(1.0, confidence))
            return True
        return False
    
    def _evict_fact(self) -> Optional[str]:
        """Evict lowest priority fact.
        
        Returns
        -------
        Optional[str]
            Evicted fact ID
        """
        if not self.facts:
            return None
        
        evict_fact = min(
            self.facts.items(),
            key=lambda x: (x[1].verified, x[1].confidence, x[1].usage_count)
        )
        
        evicted_id = evict_fact[0]
        del self.facts[evicted_id]
        
        return evicted_id
    
    def _evict_concept(self) -> Optional[str]:
        """Evict lowest priority concept.
        
        Returns
        -------
        Optional[str]
            Evicted concept ID
        """
        if not self.concepts:
            return None
        
        evict_concept = min(
            self.concepts.items(),
            key=lambda x: (x[1].confidence, x[1].usage_count)
        )
        
        evicted_id = evict_concept[0]
        del self.concepts[evicted_id]
        if evicted_id in self.relationships:
            del self.relationships[evicted_id]
        
        return evicted_id
    
    def get_status(self) -> Dict[str, Any]:
        """Get knowledge store status.
        
        Returns
        -------
        Dict[str, Any]
            Status information
        """
        verified_facts = sum(1 for f in self.facts.values() if f.verified)
        
        return {
            "max_facts": self.max_facts,
            "current_facts": len(self.facts),
            "verified_facts": verified_facts,
            "facts_utilization_percent": (len(self.facts) / self.max_facts * 100) if self.max_facts > 0 else 0,
            "max_concepts": self.max_concepts,
            "current_concepts": len(self.concepts),
            "concepts_utilization_percent": (len(self.concepts) / self.max_concepts * 100) if self.max_concepts > 0 else 0,
            "relationships_count": sum(len(rels) for rels in self.relationships.values()),
            "average_fact_confidence": sum(f.confidence for f in self.facts.values()) / len(self.facts) if self.facts else 0.0,
            "average_concept_confidence": sum(c.confidence for c in self.concepts.values()) / len(self.concepts) if self.concepts else 0.0,
        }
