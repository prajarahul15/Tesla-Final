"""
Task Decomposer for Financial Workflows
Breaks down complex financial queries into manageable sub-tasks
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
from openai import AsyncOpenAI
import os
import json
import re

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of financial tasks"""
    DATA_RETRIEVAL = "data_retrieval"
    CALCULATION = "calculation"
    ANALYSIS = "analysis"
    COMPARISON = "comparison"
    SIMULATION = "simulation"
    VISUALIZATION = "visualization"
    SYNTHESIS = "synthesis"

class TaskPriority(Enum):
    """Task execution priority"""
    HIGH = 1
    MEDIUM = 2
    LOW = 3

@dataclass
class SubTask:
    """Represents a single sub-task in a workflow"""
    id: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    dependencies: List[str]  # IDs of tasks that must complete first
    parameters: Dict[str, Any]
    agent: Optional[str] = None  # Which agent should handle this
    estimated_duration: int = 5  # seconds
    result: Optional[Dict[str, Any]] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    error: Optional[str] = None

@dataclass
class WorkflowPlan:
    """Complete workflow plan with all sub-tasks"""
    query: str
    tasks: List[SubTask]
    execution_order: List[str]  # Task IDs in execution order
    estimated_total_duration: int
    complexity_score: float  # 0-1, higher = more complex

class TaskDecomposer:
    """
    Intelligent task decomposer that breaks complex financial queries
    into manageable sub-tasks with proper dependencies and execution order
    """
    
    def __init__(self):
        self.model_name = "gpt-4o"
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key != 'your_openai_api_key_here':
            self.client = AsyncOpenAI(api_key=api_key, timeout=60.0)
        else:
            self.client = None
            logger.warning("OpenAI API key not configured for TaskDecomposer")
        
        # Define task patterns for common financial workflows
        self.workflow_patterns = {
            "valuation": ["data_retrieval", "calculation", "analysis"],
            "comparison": ["data_retrieval", "comparison", "synthesis"],
            "what_if": ["simulation", "analysis", "visualization"],
            "comprehensive": ["data_retrieval", "calculation", "analysis", "comparison", "synthesis"]
        }
    
    async def decompose_query(self, query: str, context: Optional[Dict] = None) -> WorkflowPlan:
        """
        Main entry point: Decompose a complex query into sub-tasks
        
        Args:
            query: User's financial query
            context: Optional context from session or previous queries
            
        Returns:
            WorkflowPlan with all sub-tasks and execution order
        """
        try:
            logger.info(f"🔧 Decomposing query: {query}")
            
            # Analyze query complexity
            complexity_score = self._assess_complexity(query)
            logger.info(f"📊 Query complexity: {complexity_score:.2f}")
            
            # If simple query, no decomposition needed
            if complexity_score < 0.3:
                logger.info("✅ Simple query - no decomposition needed")
                return self._create_single_task_plan(query, complexity_score)
            
            # Use AI to decompose if available
            if self.client:
                plan = await self._ai_decompose(query, context, complexity_score)
                if plan:
                    logger.info(f"✅ AI decomposition: {len(plan.tasks)} tasks")
                    return plan
            
            # Fallback to rule-based decomposition
            plan = self._rule_based_decompose(query, complexity_score)
            logger.info(f"✅ Rule-based decomposition: {len(plan.tasks)} tasks")
            return plan
            
        except Exception as e:
            logger.error(f"Error decomposing query: {str(e)}", exc_info=True)
            return self._create_single_task_plan(query, 0.5)
    
    def _assess_complexity(self, query: str) -> float:
        """
        Assess query complexity (0-1 scale)
        """
        query_lower = query.lower()
        score = 0.0
        
        # Check for multiple questions
        question_marks = query.count('?')
        score += min(question_marks * 0.1, 0.3)
        
        # Check for complex keywords
        complex_keywords = [
            'compare', 'analyze', 'evaluate', 'assess', 'comprehensive',
            'detailed', 'breakdown', 'step by step', 'multiple', 'various',
            'all', 'entire', 'complete', 'thorough', 'in-depth'
        ]
        found_keywords = sum(1 for kw in complex_keywords if kw in query_lower)
        score += min(found_keywords * 0.1, 0.3)
        
        # Check for multiple entities
        entities = [
            'revenue', 'margin', 'profit', 'cash flow', 'balance sheet',
            'income statement', 'scenario', 'valuation', 'dcf', 'forecast'
        ]
        found_entities = sum(1 for entity in entities if entity in query_lower)
        score += min(found_entities * 0.05, 0.2)
        
        # Check for temporal complexity (multiple time periods)
        temporal = ['historical', 'projected', 'forecast', 'trend', 'over time', 'years']
        if any(t in query_lower for t in temporal):
            score += 0.1
        
        # Check for multi-step indicators
        multi_step = ['first', 'then', 'after', 'finally', 'next', 'and then']
        if any(ms in query_lower for ms in multi_step):
            score += 0.15
        
        return min(score, 1.0)
    
    async def _ai_decompose(self, query: str, context: Optional[Dict], complexity_score: float) -> Optional[WorkflowPlan]:
        """
        Use GPT-4o to intelligently decompose the query
        """
        try:
            prompt = f"""
You are a financial workflow expert. Break down this financial analysis query into specific sub-tasks.

Query: "{query}"
Complexity Score: {complexity_score:.2f}

Create a workflow plan with sub-tasks. Each task should have:
1. A unique ID (task_1, task_2, etc.)
2. Clear description
3. Task type (data_retrieval, calculation, analysis, comparison, simulation, synthesis)
4. Priority (high, medium, low)
5. Dependencies (which tasks must complete first, use task IDs)
6. Parameters needed
7. Which agent should handle it (financial, market, whatif, cross_statement, or general)

Return JSON format:
{{
  "tasks": [
    {{
      "id": "task_1",
      "description": "Retrieve Tesla financial statements for 2025",
      "task_type": "data_retrieval",
      "priority": "high",
      "dependencies": [],
      "parameters": {{"scenario": "base", "year": 2025}},
      "agent": "financial",
      "estimated_duration": 5
    }},
    {{
      "id": "task_2",
      "description": "Calculate key financial metrics",
      "task_type": "calculation",
      "priority": "high",
      "dependencies": ["task_1"],
      "parameters": {{"metrics": ["revenue_growth", "margin"]}},
      "agent": "financial",
      "estimated_duration": 3
    }}
  ],
  "execution_order": ["task_1", "task_2"],
  "estimated_total_duration": 8
}}

IMPORTANT: 
- Keep tasks atomic and focused
- Ensure proper dependencies
- Parallel tasks should have no dependencies between them
- Return ONLY valid JSON, no markdown
"""
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial workflow decomposition expert. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean markdown
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\n', '', content)
                content = re.sub(r'\n```$', '', content)
            
            data = json.loads(content)
            
            # Convert to WorkflowPlan
            tasks = []
            for task_data in data.get('tasks', []):
                task = SubTask(
                    id=task_data['id'],
                    description=task_data['description'],
                    task_type=TaskType(task_data['task_type']),
                    priority=TaskPriority[task_data['priority'].upper()],
                    dependencies=task_data.get('dependencies', []),
                    parameters=task_data.get('parameters', {}),
                    agent=task_data.get('agent'),
                    estimated_duration=task_data.get('estimated_duration', 5)
                )
                tasks.append(task)
            
            plan = WorkflowPlan(
                query=query,
                tasks=tasks,
                execution_order=data.get('execution_order', [t.id for t in tasks]),
                estimated_total_duration=data.get('estimated_total_duration', sum(t.estimated_duration for t in tasks)),
                complexity_score=complexity_score
            )
            
            logger.info(f"✅ AI-generated workflow plan with {len(tasks)} tasks")
            return plan
            
        except Exception as e:
            logger.error(f"AI decomposition error: {str(e)}")
            return None
    
    def _rule_based_decompose(self, query: str, complexity_score: float) -> WorkflowPlan:
        """
        Fallback rule-based decomposition
        """
        query_lower = query.lower()
        tasks = []
        task_id_counter = 1
        
        # Identify workflow type
        workflow_type = self._identify_workflow_type(query_lower)
        
        # Data Retrieval (always first)
        tasks.append(SubTask(
            id=f"task_{task_id_counter}",
            description="Retrieve relevant financial data",
            task_type=TaskType.DATA_RETRIEVAL,
            priority=TaskPriority.HIGH,
            dependencies=[],
            parameters={"scope": "financial_statements", "scenario": "base"},
            agent="financial",
            estimated_duration=5
        ))
        prev_task_id = f"task_{task_id_counter}"
        task_id_counter += 1
        
        # Add workflow-specific tasks
        if "compare" in query_lower or "comparison" in query_lower:
            tasks.append(SubTask(
                id=f"task_{task_id_counter}",
                description="Compare financial metrics across scenarios/periods",
                task_type=TaskType.COMPARISON,
                priority=TaskPriority.HIGH,
                dependencies=[prev_task_id],
                parameters={"type": "scenario_comparison"},
                agent="financial",
                estimated_duration=8
            ))
            prev_task_id = f"task_{task_id_counter}"
            task_id_counter += 1
        
        if "what if" in query_lower or "simulate" in query_lower:
            tasks.append(SubTask(
                id=f"task_{task_id_counter}",
                description="Run simulation with specified parameters",
                task_type=TaskType.SIMULATION,
                priority=TaskPriority.HIGH,
                dependencies=[prev_task_id],
                parameters={"type": "what_if_simulation"},
                agent="whatif",
                estimated_duration=10
            ))
            prev_task_id = f"task_{task_id_counter}"
            task_id_counter += 1
        
        if any(word in query_lower for word in ['analyze', 'analysis', 'evaluate', 'assess']):
            tasks.append(SubTask(
                id=f"task_{task_id_counter}",
                description="Perform detailed financial analysis",
                task_type=TaskType.ANALYSIS,
                priority=TaskPriority.MEDIUM,
                dependencies=[prev_task_id],
                parameters={"depth": "comprehensive"},
                agent="financial",
                estimated_duration=7
            ))
            prev_task_id = f"task_{task_id_counter}"
            task_id_counter += 1
        
        # Synthesis (always last if multiple tasks)
        if len(tasks) > 1:
            tasks.append(SubTask(
                id=f"task_{task_id_counter}",
                description="Synthesize results into comprehensive response",
                task_type=TaskType.SYNTHESIS,
                priority=TaskPriority.LOW,
                dependencies=[prev_task_id],
                parameters={},
                agent="general",
                estimated_duration=3
            ))
        
        # Generate execution order (topological sort)
        execution_order = self._generate_execution_order(tasks)
        
        plan = WorkflowPlan(
            query=query,
            tasks=tasks,
            execution_order=execution_order,
            estimated_total_duration=sum(t.estimated_duration for t in tasks),
            complexity_score=complexity_score
        )
        
        return plan
    
    def _identify_workflow_type(self, query_lower: str) -> str:
        """Identify the type of financial workflow"""
        if "valuation" in query_lower or "dcf" in query_lower:
            return "valuation"
        elif "compare" in query_lower or "versus" in query_lower:
            return "comparison"
        elif "what if" in query_lower or "simulate" in query_lower:
            return "what_if"
        elif any(word in query_lower for word in ['comprehensive', 'complete', 'detailed', 'full']):
            return "comprehensive"
        else:
            return "standard"
    
    def _generate_execution_order(self, tasks: List[SubTask]) -> List[str]:
        """
        Generate execution order using topological sort based on dependencies
        """
        # Build adjacency list
        in_degree = {task.id: len(task.dependencies) for task in tasks}
        adj_list = {task.id: [] for task in tasks}
        
        for task in tasks:
            for dep in task.dependencies:
                adj_list[dep].append(task.id)
        
        # Topological sort using Kahn's algorithm
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        execution_order = []
        
        while queue:
            # Sort by priority for tasks at same level
            queue.sort(key=lambda tid: next(t.priority.value for t in tasks if t.id == tid))
            current = queue.pop(0)
            execution_order.append(current)
            
            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return execution_order
    
    def _create_single_task_plan(self, query: str, complexity_score: float) -> WorkflowPlan:
        """Create a simple single-task plan for non-complex queries"""
        task = SubTask(
            id="task_1",
            description=f"Process query: {query[:50]}...",
            task_type=TaskType.ANALYSIS,
            priority=TaskPriority.HIGH,
            dependencies=[],
            parameters={"query": query},
            agent="general",
            estimated_duration=5
        )
        
        return WorkflowPlan(
            query=query,
            tasks=[task],
            execution_order=["task_1"],
            estimated_total_duration=5,
            complexity_score=complexity_score
        )
    
    def visualize_workflow(self, plan: WorkflowPlan) -> str:
        """
        Create a text-based visualization of the workflow
        """
        lines = []
        lines.append(f"\n{'='*80}")
        lines.append(f"  WORKFLOW PLAN: {plan.query[:60]}...")
        lines.append(f"{'='*80}")
        lines.append(f"Complexity: {plan.complexity_score:.2f} | Total Duration: ~{plan.estimated_total_duration}s | Tasks: {len(plan.tasks)}")
        lines.append("")
        
        # Group tasks by dependency level
        levels = {}
        for task in plan.tasks:
            level = len(task.dependencies)
            if level not in levels:
                levels[level] = []
            levels[level].append(task)
        
        # Display by level
        for level in sorted(levels.keys()):
            lines.append(f"Level {level}: {'[Parallel Execution]' if len(levels[level]) > 1 else ''}")
            for task in levels[level]:
                deps = f" → Depends on: {', '.join(task.dependencies)}" if task.dependencies else ""
                lines.append(f"  {task.id}: {task.description}")
                lines.append(f"     Type: {task.task_type.value} | Priority: {task.priority.name} | Agent: {task.agent}")
                if task.dependencies:
                    lines.append(f"     Dependencies: {', '.join(task.dependencies)}")
                lines.append("")
        
        lines.append(f"Execution Order: {' → '.join(plan.execution_order)}")
        lines.append(f"{'='*80}\n")
        
        return "\n".join(lines)

# Singleton instance
task_decomposer = TaskDecomposer()

