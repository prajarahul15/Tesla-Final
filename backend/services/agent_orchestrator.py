"""
Agent Orchestrator for Tesla Financial Model
Coordinates multiple AI agents to handle complex queries with multi-step workflows
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from openai import OpenAI
from enum import Enum

# Agent types enum
class AgentType(Enum):
    PROACTIVE_INSIGHTS = "proactive_insights"
    PROPHET_FORECASTING = "prophet_forecasting"
    TESLA_AI_SIMULATOR = "tesla_ai_simulator"
    INCOME_STATEMENT = "income_statement"
    CROSS_STATEMENT = "cross_statement"
    TESLA_FA_CHAT = "tesla_fa_chat"
    METRIC_FORECASTING = "metric_forecasting"

# Task types
class TaskType(Enum):
    ANALYSIS = "analysis"
    FORECASTING = "forecasting"
    SIMULATION = "simulation"
    QUESTION_ANSWER = "question_answer"
    COMPARISON = "comparison"
    INSIGHT_GENERATION = "insight_generation"


class SharedContextStore:
    """
    Shared memory store for agent context and communication
    Enables agents to access results from other agents
    """
    
    def __init__(self):
        self.contexts = {}  # session_id -> context dict
        self.agent_results = {}  # session_id -> {agent_type: result}
        self.conversation_history = {}  # session_id -> messages list
    
    def create_session(self, session_id: str) -> None:
        """Create a new session"""
        self.contexts[session_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "scenario": "base",
            "year": None,
            "user_preferences": {}
        }
        self.agent_results[session_id] = {}
        self.conversation_history[session_id] = []
    
    def update_context(self, session_id: str, key: str, value: Any) -> None:
        """Update context for a session"""
        if session_id not in self.contexts:
            self.create_session(session_id)
        self.contexts[session_id][key] = value
    
    def get_context(self, session_id: str) -> Dict:
        """Get context for a session"""
        return self.contexts.get(session_id, {})
    
    def store_agent_result(self, session_id: str, agent_type: AgentType, result: Any) -> None:
        """Store result from an agent"""
        if session_id not in self.agent_results:
            self.agent_results[session_id] = {}
        self.agent_results[session_id][agent_type.value] = {
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_agent_result(self, session_id: str, agent_type: AgentType) -> Optional[Any]:
        """Get result from a specific agent"""
        if session_id in self.agent_results:
            agent_data = self.agent_results[session_id].get(agent_type.value)
            return agent_data.get("result") if agent_data else None
        return None
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add message to conversation history"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = []
        self.conversation_history[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get recent conversation history"""
        if session_id in self.conversation_history:
            return self.conversation_history[session_id][-limit:]
        return []
    
    def clear_session(self, session_id: str) -> None:
        """Clear all data for a session"""
        self.contexts.pop(session_id, None)
        self.agent_results.pop(session_id, None)
        self.conversation_history.pop(session_id, None)


class AgentOrchestrator:
    """
    Master orchestrator that coordinates multiple AI agents
    Routes queries, manages workflows, and aggregates results
    """
    
    def __init__(self):
        self.model_name = "gpt-4o-mini"
        self.client = None
        self.context_store = SharedContextStore()
        
        # Import agents lazily to avoid circular dependencies
        self.agents = {}
        self._load_agents()
    
    def _load_agents(self):
        """Load all available agents - lazy loading to avoid initialization errors"""
        # Don't load agents at orchestrator init - load them when needed
        # This avoids OpenAI client initialization errors at module import time
        self.agents_loaded = False
        print(f"[Orchestrator] Agent loading deferred until first use")
    
    def _ensure_agents_loaded(self):
        """Lazy load agents on first use"""
        if self.agents_loaded or len(self.agents) > 0:
            return
        
        try:
            from services.ai_agents import (
                proactive_insights_agent,
                prophet_forecasting_agent,
                tesla_ai_agent,
                income_statement_insights_agent,
                cross_statement_insights_agent
            )
            from services.tesla_fa_agent import tesla_fa_agent
            from services.metric_forecasting import metric_forecasting_service
            
            self.agents = {
                AgentType.PROACTIVE_INSIGHTS: proactive_insights_agent,
                AgentType.PROPHET_FORECASTING: prophet_forecasting_agent,
                AgentType.TESLA_AI_SIMULATOR: tesla_ai_agent,
                AgentType.INCOME_STATEMENT: income_statement_insights_agent,
                AgentType.CROSS_STATEMENT: cross_statement_insights_agent,
                AgentType.TESLA_FA_CHAT: tesla_fa_agent,
                AgentType.METRIC_FORECASTING: metric_forecasting_service
            }
            self.agents_loaded = True
            print(f"✅ Loaded {len(self.agents)} agents successfully")
        except Exception as e:
            print(f"⚠️  Error loading agents: {e}")
            self.agents = {}
    
    def _initialize_client(self):
        """Initialize OpenAI client for orchestration"""
        if self.client is None:
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key and api_key != 'your_openai_api_key_here':
                try:
                    self.client = OpenAI(api_key=api_key, timeout=30.0)
                    print("[Orchestrator] OpenAI client initialized")
                except Exception as e:
                    print(f"[Orchestrator] Failed to initialize client: {e}")
    
    def route_query(self, query: str, context: Dict = None) -> Tuple[List[AgentType], TaskType]:
        """
        Intelligent query routing - determines which agents should handle the query
        Returns: (list of agents to use, task type)
        """
        query_lower = query.lower()
        
        # Multi-agent scenarios
        if any(word in query_lower for word in ['forecast', 'predict', 'project']) and \
           any(word in query_lower for word in ['analyze', 'insight', 'risk']):
            # Forecasting + Analysis
            return [AgentType.PROPHET_FORECASTING, AgentType.PROACTIVE_INSIGHTS], TaskType.FORECASTING
        
        if any(word in query_lower for word in ['simulate', 'what if', 'change']) and \
           any(word in query_lower for word in ['impact', 'effect', 'result']):
            # Simulation + Impact Analysis
            return [AgentType.TESLA_AI_SIMULATOR, AgentType.CROSS_STATEMENT], TaskType.SIMULATION
        
        if 'income statement' in query_lower and any(word in query_lower for word in ['compare', 'analyze', 'impact']):
            return [AgentType.INCOME_STATEMENT], TaskType.COMPARISON
        
        if any(word in query_lower for word in ['all statements', 'balance sheet', 'cash flow', 'cross-statement']):
            return [AgentType.CROSS_STATEMENT], TaskType.ANALYSIS
        
        if any(word in query_lower for word in ['forecast', 'predict', 'trend']):
            return [AgentType.PROPHET_FORECASTING], TaskType.FORECASTING
        
        if any(word in query_lower for word in ['revenue', 'metric', 'cogs', 'r&d', 'sga']):
            return [AgentType.METRIC_FORECASTING], TaskType.FORECASTING
        
        # Default: conversational query
        return [AgentType.TESLA_FA_CHAT], TaskType.QUESTION_ANSWER
    
    def decompose_task(self, query: str, context: Dict = None) -> List[Dict]:
        """
        Decompose complex query into sub-tasks for different agents
        Returns list of tasks with agent assignments
        """
        self._initialize_client()
        
        if self.client is None:
            # Fallback: simple routing
            agents, task_type = self.route_query(query, context)
            return [{
                "agent": agents[0],
                "task": query,
                "task_type": task_type,
                "dependencies": []
            }]
        
        try:
            # Use LLM to decompose complex queries
            decomposition_prompt = f"""
You are a task planner for a financial modeling AI system. Break down the following query into specific sub-tasks.

Available agents:
- proactive_insights: Strategic analysis and insights
- prophet_forecasting: Time series forecasting
- tesla_ai_simulator: Interactive parameter simulation
- income_statement: Income statement analysis
- cross_statement: Integrated 3-statement analysis
- tesla_fa_chat: General Q&A and explanations
- metric_forecasting: Metric-specific forecasting

Query: {query}

Return a JSON array of tasks in this format:
[{{
  "agent": "agent_name",
  "task_description": "what this agent should do",
  "dependencies": ["task_0", "task_1"] or [],
  "priority": 1-10
}}]

Rules:
- Keep tasks atomic and specific
- Identify dependencies between tasks
- Higher priority = execute first
- Maximum 5 tasks
"""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": decomposition_prompt}],
                temperature=0.3,
                max_tokens=500,
                timeout=15.0
            )
            
            content = response.choices[0].message.content.strip()
            # Parse JSON
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            tasks = json.loads(content)
            return tasks
            
        except Exception as e:
            print(f"[Orchestrator] Task decomposition failed: {e}")
            # Fallback to simple routing
            agents, task_type = self.route_query(query, context)
            return [{
                "agent": agents[0].value,
                "task_description": query,
                "task_type": task_type.value,
                "dependencies": [],
                "priority": 5
            }]
    
    async def execute_workflow(
        self, 
        query: str, 
        session_id: str = None,
        context: Dict = None
    ) -> Dict:
        """
        Execute a complete workflow with multi-agent coordination
        
        Args:
            query: User's query or request
            session_id: Unique session identifier
            context: Additional context (scenario, year, etc.)
            
        Returns:
            Aggregated results from all agents
        """
        # Ensure agents are loaded
        self._ensure_agents_loaded()
        
        # Generate session ID if not provided
        if session_id is None:
            session_id = f"session_{datetime.utcnow().timestamp()}"
        
        # Create session in context store
        if session_id not in self.context_store.contexts:
            self.context_store.create_session(session_id)
        
        # Update context
        if context:
            for key, value in context.items():
                self.context_store.update_context(session_id, key, value)
        
        # Store user query
        self.context_store.add_message(session_id, "user", query)
        
        print(f"\n{'='*60}")
        print(f"[Orchestrator] Processing query: {query[:100]}...")
        print(f"[Orchestrator] Session: {session_id}")
        print(f"{'='*60}\n")
        
        # Step 1: Route query to appropriate agents
        agents_to_use, task_type = self.route_query(query, context)
        print(f"[Orchestrator] Selected agents: {[a.value for a in agents_to_use]}")
        print(f"[Orchestrator] Task type: {task_type.value}")
        
        # Step 2: Decompose into sub-tasks if complex
        tasks = self.decompose_task(query, context)
        print(f"[Orchestrator] Decomposed into {len(tasks)} task(s)")
        
        # Step 3: Execute tasks in priority order with dependency management
        task_results = {}
        sorted_tasks = sorted(tasks, key=lambda t: t.get('priority', 5), reverse=True)
        
        for i, task in enumerate(sorted_tasks):
            task_id = f"task_{i}"
            agent_name = task.get('agent')
            task_desc = task.get('task_description', query)
            dependencies = task.get('dependencies', [])
            
            print(f"\n[Orchestrator] Executing {task_id}: {agent_name}")
            
            # Check dependencies
            if dependencies:
                missing_deps = [dep for dep in dependencies if dep not in task_results]
                if missing_deps:
                    print(f"[Orchestrator] Skipping {task_id} - missing dependencies: {missing_deps}")
                    continue
            
            # Get dependency results for context
            dep_context = {dep: task_results[dep] for dep in dependencies if dep in task_results}
            
            # Execute task with appropriate agent
            try:
                agent_type = self._get_agent_type(agent_name)
                if agent_type:
                    result = await self._execute_agent_task(
                        agent_type, 
                        task_desc, 
                        session_id,
                        dep_context
                    )
                    task_results[task_id] = result
                    
                    # Store in shared context
                    self.context_store.store_agent_result(session_id, agent_type, result)
                    print(f"[Orchestrator] ✅ {task_id} completed successfully")
                else:
                    print(f"[Orchestrator] ⚠️  Unknown agent: {agent_name}")
                    
            except Exception as e:
                print(f"[Orchestrator] ❌ {task_id} failed: {e}")
                task_results[task_id] = {"error": str(e)}
        
        # Step 4: Aggregate results
        aggregated_result = self._aggregate_results(task_results, query, agents_to_use)
        
        # Store final result
        self.context_store.add_message(session_id, "assistant", json.dumps(aggregated_result))
        
        print(f"\n[Orchestrator] ✅ Workflow completed")
        print(f"{'='*60}\n")
        
        return {
            "success": True,
            "session_id": session_id,
            "query": query,
            "task_type": task_type.value,
            "agents_used": [a.value for a in agents_to_use],
            "tasks_executed": len(task_results),
            "result": aggregated_result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _get_agent_type(self, agent_name: str) -> Optional[AgentType]:
        """Convert agent name to AgentType enum"""
        agent_map = {
            "proactive_insights": AgentType.PROACTIVE_INSIGHTS,
            "prophet_forecasting": AgentType.PROPHET_FORECASTING,
            "tesla_ai_simulator": AgentType.TESLA_AI_SIMULATOR,
            "income_statement": AgentType.INCOME_STATEMENT,
            "cross_statement": AgentType.CROSS_STATEMENT,
            "tesla_fa_chat": AgentType.TESLA_FA_CHAT,
            "metric_forecasting": AgentType.METRIC_FORECASTING
        }
        return agent_map.get(agent_name)
    
    async def _execute_agent_task(
        self, 
        agent_type: AgentType, 
        task: str, 
        session_id: str,
        dependencies: Dict = None
    ) -> Any:
        """Execute a specific task with the appropriate agent"""
        agent = self.agents.get(agent_type)
        if not agent:
            raise ValueError(f"Agent {agent_type.value} not available")
        
        context = self.context_store.get_context(session_id)
        
        print(f"[Orchestrator] → Calling {agent_type.value} agent...")
        
        # Route to appropriate agent method based on type
        if agent_type == AgentType.TESLA_FA_CHAT:
            result = agent.generate_response(task, "orchestrated_query")
            return {"response": result, "agent": agent_type.value}
        
        elif agent_type == AgentType.PROACTIVE_INSIGHTS:
            # Need model data - get from context or previous agent results
            model_data = dependencies.get("simulation_result") if dependencies else None
            if not model_data:
                # Generate basic model data
                from data.tesla_data import TESLA_BASE_YEAR_DATA
                model_data = {"income_statements": [TESLA_BASE_YEAR_DATA]}
            
            scenario = context.get("scenario", "base")
            result = agent.analyze_financial_model(model_data, scenario)
            return {"insights": result, "agent": agent_type.value}
        
        elif agent_type == AgentType.CROSS_STATEMENT:
            # This needs original and updated statements
            # For now, return a placeholder
            return {"message": "Cross-statement analysis requires simulation data", "agent": agent_type.value}
        
        elif agent_type == AgentType.METRIC_FORECASTING:
            # Load data if needed
            if agent.monthly_data is None:
                agent.load_data()
            
            # Default univariate forecast for revenue
            result = agent.univariate_forecast("revenue_millions", horizon=12, test_months=6)
            return {"forecast": result, "agent": agent_type.value}
        
        else:
            return {"message": f"Agent {agent_type.value} executed", "agent": agent_type.value}
    
    def _aggregate_results(self, task_results: Dict, query: str, agents_used: List[AgentType]) -> Dict:
        """Aggregate results from multiple agents into a cohesive response"""
        self._initialize_client()
        
        if self.client is None or not task_results:
            # Simple aggregation without AI
            return {
                "summary": f"Processed query with {len(task_results)} task(s)",
                "results": task_results
            }
        
        try:
            # Use LLM to synthesize results
            synthesis_prompt = f"""
You are synthesizing results from multiple AI agents that analyzed this query: "{query}"

Agent results:
{json.dumps(task_results, indent=2, default=str)}

Provide a cohesive summary in JSON format:
{{
  "executive_summary": "2-3 sentence overview of key findings",
  "key_insights": ["insight 1", "insight 2", "insight 3"],
  "recommendations": ["recommendation 1", "recommendation 2"],
  "detailed_findings": {{"agent_name": "specific findings"}},
  "next_steps": ["suggested next step 1", "suggested next step 2"]
}}

Focus on:
- Integrating insights from different agents
- Highlighting contradictions or alignments
- Providing actionable recommendations
- Suggesting logical next steps
"""
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": synthesis_prompt}],
                temperature=0.4,
                max_tokens=800,
                timeout=20.0
            )
            
            content = response.choices[0].message.content.strip()
            # Parse JSON
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            aggregated = json.loads(content)
            aggregated["raw_results"] = task_results
            return aggregated
            
        except Exception as e:
            print(f"[Orchestrator] Result aggregation failed: {e}")
            return {
                "summary": f"Executed {len(task_results)} task(s) successfully",
                "results": task_results,
                "agents_used": [a.value for a in agents_used]
            }
    
    def ask(self, query: str, session_id: str = None, context: Dict = None) -> Dict:
        """
        Synchronous wrapper for execute_workflow
        Main entry point for orchestrated queries
        """
        import asyncio
        
        # Create event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.execute_workflow(query, session_id, context))
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary of a session including all agent interactions"""
        context = self.context_store.get_context(session_id)
        agent_results = self.context_store.agent_results.get(session_id, {})
        conversation = self.context_store.get_conversation_history(session_id)
        
        return {
            "session_id": session_id,
            "context": context,
            "agents_called": list(agent_results.keys()),
            "total_interactions": len(conversation),
            "conversation_preview": conversation[-5:] if conversation else []
        }
    
    def list_available_agents(self) -> List[Dict]:
        """List all available agents and their capabilities"""
        return [
            {
                "type": AgentType.PROACTIVE_INSIGHTS.value,
                "name": "Proactive Insights Agent",
                "description": "Generates strategic insights from financial models",
                "capabilities": ["scenario analysis", "risk identification", "strategic recommendations"]
            },
            {
                "type": AgentType.PROPHET_FORECASTING.value,
                "name": "Prophet Forecasting Agent",
                "description": "Time-series forecasting with AI insights",
                "capabilities": ["revenue forecasting", "trend analysis", "confidence intervals"]
            },
            {
                "type": AgentType.TESLA_AI_SIMULATOR.value,
                "name": "Tesla AI Simulator",
                "description": "Interactive simulation of pricing, cost, and delivery changes",
                "capabilities": ["parameter simulation", "impact analysis", "real-time insights"]
            },
            {
                "type": AgentType.INCOME_STATEMENT.value,
                "name": "Income Statement Analyst",
                "description": "Deep analysis of income statement changes",
                "capabilities": ["revenue analysis", "margin analysis", "opex analysis"]
            },
            {
                "type": AgentType.CROSS_STATEMENT.value,
                "name": "Cross-Statement Analyst",
                "description": "Integrated analysis across all 3 financial statements",
                "capabilities": ["balance sheet analysis", "cash flow analysis", "integrated metrics"]
            },
            {
                "type": AgentType.TESLA_FA_CHAT.value,
                "name": "Tesla Financial Assistant",
                "description": "Conversational AI for financial modeling questions",
                "capabilities": ["Q&A", "explanations", "guidance"]
            },
            {
                "type": AgentType.METRIC_FORECASTING.value,
                "name": "Metric Forecasting Service",
                "description": "Machine learning forecasting for financial metrics",
                "capabilities": ["univariate forecasting", "multivariate forecasting", "feature importance"]
            }
        ]


# Global orchestrator instance
agent_orchestrator = AgentOrchestrator()

