"""
Brain Module - Central cognitive system
รวม imports ทั้งหมดจาก Brain submodules
"""

# Meta - Self-awareness system
from .Meta.SelfModel import SelfModel
from .Meta.GoalTracker import GoalTracker
from .Meta.Introspection import Introspection

# Memory - Hierarchical memory system
from .Memory.MemoryEngine import MemoryEngine, ConsolidationConfig

# Memory - ShortTerm
from .Memory.ShortTerm.WorkingMemory import MemoryPriority, WorkingMemoryItem, WorkingMemory
from .Memory.ShortTerm.AttentionMap import AttentionType, AttentionFocus, AttentionMap

# Memory - MiddleTerm
from .Memory.MiddleTerm.ContextBuffer import ContextFrame, ContextBuffer
from .Memory.MiddleTerm.TopicContext import TopicContextData, TopicContext

# Memory - LongTerm
from .Memory.LongTerm.ExperienceStore import Experience, ExperienceStore
from .Memory.LongTerm.KnowledgeStore import Fact, Concept, KnowledgeStore
from .Memory.LongTerm.WeightArchive import WeightSnapshot, WeightArchive

# Memory - Emotions
from .Memory.Emotions.EmotionProfile import EmotionProfile, EmotionalTrait
from .Memory.Emotions.EmotionState import EmotionState, EmotionIntensity
from .Memory.Emotions.EmotionWeight import EmotionWeight, BiasProfile
from .Memory.Emotions.EmotionHistory import EmotionHistory
from .Memory.Emotions.EmotionEncoder import EmotionEncoder, EmotionVector

# Memory - Topics
from .Memory.Topics.TopicProfile import TopicProfile, TopicExpertise
from .Memory.Topics.TopicState import TopicState, TopicStateData
from .Memory.Topics.TopicWeight import TopicWeight, TopicWeightData
from .Memory.Topics.TopicHistory import TopicHistory
from .Memory.Topics.TopicEmbedding import TopicEmbedding
from .Memory.Topics.TopicRouter import TopicRouter, RoutingRule

# Neural - Functions
from .Neural.Functions.Activation import ActivationFunctions
from .Neural.Functions.Metrics import Metrics

# Neural - Weights
from .Neural.Weights.WeightSet import WeightSet
from .Neural.Weights.WeightStore import WeightStore
from .Neural.Weights.WeightLinker import WeightLinker
from .Neural.Weights.WeightStats import WeightStats

# Neural - Learning
from .Neural.Learning.LearningEngine import LearningEngine
from .Neural.Learning.GradientLearner import GradientLearner
from .Neural.Learning.SelfLearner import SelfLearner
from .Neural.Learning.AdvisorLearner import AdvisorLearner
from .Neural.Learning.EvolutionLearner import EvolutionLearner
from .Neural.Learning.ReplayLearner import ReplayLearner

# Neural - Core
from .Neural.Brain import BrainStructure
from .Neural.BrainController import BrainController

# Review - Quality assurance
from .Review.ApproveEngine import ApproveEngine
from .Review.ConfidenceScorer import ConfidenceScorer
from .Review.PerformanceMonitor import PerformanceMonitor

# Rules - Central rule engine
from .Rules.RuleEngine import RuleEngine

# Rules - Adaption
from .Rules.Adaption.AdaptionRule import AdaptionRule
from .Rules.Adaption.EmotionDrivenRule import EmotionDrivenRule
from .Rules.Adaption.StructureRule import StructureRule

# Rules - Learning
from .Rules.Learning.GradientRule import GradientRule
from .Rules.Learning.LearnRule import LearnRule

# Rules - Memory
from .Rules.Memory.EmotionRule import EmotionRule
from .Rules.Memory.MemoryRule import MemoryRule
from .Rules.Memory.TopicRule import TopicRule

# Rules - Routing
from .Rules.Routing.RoutingRule import RoutingRule

# Rules - Safety
from .Rules.Safety.SafetyRule import SafetyRule

# Rules - System
from .Rules.System.RuntimePolicy import RuntimePolicy
from .Rules.System.SystemRule import SystemRule

__all__ = [
    # Meta
    'SelfModel', 'GoalTracker', 'Introspection',
    
    # Memory - Core
    'MemoryEngine', 'ConsolidationConfig',
    
    # Memory - ShortTerm
    'MemoryPriority', 'WorkingMemoryItem', 'WorkingMemory',
    'AttentionType', 'AttentionFocus', 'AttentionMap',
    
    # Memory - MiddleTerm
    'ContextFrame', 'ContextBuffer',
    'TopicContextData', 'TopicContext',
    
    # Memory - LongTerm
    'Experience', 'ExperienceStore',
    'Fact', 'Concept', 'KnowledgeStore',
    'WeightSnapshot', 'WeightArchive',
    
    # Memory - Emotions
    'EmotionProfile', 'EmotionalTrait',
    'EmotionState', 'EmotionIntensity',
    'EmotionWeight', 'BiasProfile',
    'EmotionHistory',
    'EmotionEncoder', 'EmotionVector',
    
    # Memory - Topics
    'TopicProfile', 'TopicExpertise',
    'TopicState', 'TopicStateData',
    'TopicWeight', 'TopicWeightData',
    'TopicHistory',
    'TopicEmbedding',
    'TopicRouter', 'RoutingRule',
    
    # Neural - Functions
    'ActivationFunctions', 'Metrics',
    
    # Neural - Weights
    'WeightSet', 'WeightStore', 'WeightLinker', 'WeightStats',
    
    # Neural - Learning
    'LearningEngine', 'GradientLearner', 'SelfLearner', 'AdvisorLearner', 'EvolutionLearner', 'ReplayLearner',
    
    # Neural - Core
    'BrainStructure', 'BrainController',
    
    # Review
    'ApproveEngine', 'ConfidenceScorer', 'PerformanceMonitor',
    
    # Rules
    'RuleEngine',
    
    # Rules - Adaption
    'AdaptionRule', 'EmotionDrivenRule', 'StructureRule',
    
    # Rules - Learning
    'GradientRule', 'LearnRule',
    
    # Rules - Memory
    'EmotionRule', 'MemoryRule', 'TopicRule',
    
    # Rules - Routing
    'RoutingRule',
    
    # Rules - Safety
    'SafetyRule',
    
    # Rules - System
    'RuntimePolicy', 'SystemRule',
]