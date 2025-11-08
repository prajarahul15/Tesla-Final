"""
Shared Context for Multi-Agent Coordination
Enables agents to share insights, facts, and intermediate results
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

class FactConfidence(Enum):
    """Confidence levels for facts"""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.85
    VERY_HIGH = 0.95

@dataclass
class Fact:
    """A fact contributed by an agent"""
    key: str
    value: Any
    agent_id: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    supporting_data: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'key': self.key,
            'value': self.value,
            'agent_id': self.agent_id,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class Conflict:
    """A conflict between agent outputs"""
    key: str
    facts: List[Fact]
    severity: str  # 'low', 'medium', 'high'
    
    def to_dict(self) -> Dict:
        return {
            'key': self.key,
            'facts': [f.to_dict() for f in self.facts],
            'severity': self.severity
        }

class SharedContext:
    """
    Shared context store for multi-agent coordination
    Allows agents to share facts, insights, and intermediate results
    """
    
    def __init__(self, query: str = ""):
        self.query = query
        self.facts: Dict[str, List[Fact]] = {}  # key -> list of facts
        self.agent_outputs: Dict[str, Dict] = {}  # agent_id -> full output
        self.conflicts: List[Conflict] = []
        self.session_metadata: Dict[str, Any] = {}
        
    def add_fact(self, key: str, value: Any, agent_id: str, 
                 confidence: float = 0.8, supporting_data: Optional[Dict] = None):
        """Agent contributes a fact to shared context"""
        fact = Fact(
            key=key,
            value=value,
            agent_id=agent_id,
            confidence=confidence,
            supporting_data=supporting_data
        )
        
        if key not in self.facts:
            self.facts[key] = []
        
        self.facts[key].append(fact)
        
        # Check for conflicts
        self._detect_conflicts(key)
    
    def get_fact(self, key: str, prefer_agent: Optional[str] = None) -> Optional[Any]:
        """
        Retrieve a fact value, optionally preferring a specific agent
        If multiple facts exist, returns highest confidence value
        """
        if key not in self.facts:
            return None
        
        facts = self.facts[key]
        
        # Prefer specific agent if requested
        if prefer_agent:
            agent_facts = [f for f in facts if f.agent_id == prefer_agent]
            if agent_facts:
                return max(agent_facts, key=lambda f: f.confidence).value
        
        # Return highest confidence fact
        return max(facts, key=lambda f: f.confidence).value
    
    def get_all_facts_for_key(self, key: str) -> List[Fact]:
        """Get all facts for a specific key"""
        return self.facts.get(key, [])
    
    def get_facts_by_agent(self, agent_id: str) -> Dict[str, List[Fact]]:
        """Get all facts contributed by a specific agent"""
        agent_facts = {}
        for key, facts in self.facts.items():
            agent_specific = [f for f in facts if f.agent_id == agent_id]
            if agent_specific:
                agent_facts[key] = agent_specific
        return agent_facts
    
    def store_agent_output(self, agent_id: str, output: Dict):
        """Store complete agent output for reference by other agents"""
        self.agent_outputs[agent_id] = output
    
    def get_agent_output(self, agent_id: str) -> Optional[Dict]:
        """Retrieve another agent's output"""
        return self.agent_outputs.get(agent_id)
    
    def get_all_agent_outputs(self) -> Dict[str, Dict]:
        """Get all agent outputs"""
        return self.agent_outputs
    
    def _detect_conflicts(self, key: str):
        """Detect if there are conflicting facts for a key"""
        if key not in self.facts or len(self.facts[key]) < 2:
            return
        
        facts = self.facts[key]
        values = [f.value for f in facts]
        
        # Check if values are significantly different
        if isinstance(values[0], (int, float)):
            # For numeric values, check variance
            avg = sum(values) / len(values)
            variance = sum((v - avg) ** 2 for v in values) / len(values)
            relative_variance = (variance ** 0.5) / avg if avg != 0 else 0
            
            if relative_variance > 0.15:  # 15% variance threshold
                severity = 'high' if relative_variance > 0.3 else 'medium'
                conflict = Conflict(key=key, facts=facts, severity=severity)
                self.conflicts.append(conflict)
        else:
            # For non-numeric, check if values are different
            unique_values = set(str(v) for v in values)
            if len(unique_values) > 1:
                conflict = Conflict(key=key, facts=facts, severity='medium')
                self.conflicts.append(conflict)
    
    def get_conflicts(self) -> List[Conflict]:
        """Get all detected conflicts"""
        return self.conflicts
    
    def has_conflicts(self) -> bool:
        """Check if any conflicts exist"""
        return len(self.conflicts) > 0
    
    def get_summary(self) -> Dict:
        """Get a summary of the shared context"""
        return {
            'query': self.query,
            'total_facts': sum(len(facts) for facts in self.facts.values()),
            'fact_categories': list(self.facts.keys()),
            'agents_contributed': list(self.agent_outputs.keys()),
            'conflicts_detected': len(self.conflicts),
            'high_confidence_facts': sum(
                1 for facts in self.facts.values() 
                for f in facts if f.confidence > 0.8
            )
        }
    
    def to_dict(self) -> Dict:
        """Export context as dictionary"""
        return {
            'query': self.query,
            'facts': {
                key: [f.to_dict() for f in facts]
                for key, facts in self.facts.items()
            },
            'conflicts': [c.to_dict() for c in self.conflicts],
            'summary': self.get_summary()
        }

