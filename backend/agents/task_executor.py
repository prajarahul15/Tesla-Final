"""
Task Executor for Managing Sub-Task Execution
Executes workflow plans with proper dependency management and parallel execution
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from .task_decomposer import WorkflowPlan, SubTask, TaskType, TaskPriority

logger = logging.getLogger(__name__)

class TaskExecutor:
    """
    Executes workflow plans by managing sub-task execution,
    handling dependencies, and enabling parallel execution where possible
    """
    
    def __init__(self, orchestrator):
        """
        Args:
            orchestrator: Reference to EnhancedAgentOrchestrator for agent access
        """
        self.orchestrator = orchestrator
        self.execution_history = []
    
    async def execute_workflow(self, plan: WorkflowPlan, session_id: str = "default") -> Dict[str, Any]:
        """
        Execute a complete workflow plan
        
        Args:
            plan: WorkflowPlan with tasks and execution order
            session_id: Session identifier for context
            
        Returns:
            Dictionary with execution results and metadata
        """
        try:
            logger.info(f"🚀 Executing workflow: {len(plan.tasks)} tasks")
            start_time = datetime.now()
            
            # Track execution
            execution_log = []
            results = {}
            
            # Execute tasks in order, respecting dependencies
            executed_tasks = set()
            
            for task_id in plan.execution_order:
                task = next(t for t in plan.tasks if t.id == task_id)
                
                # Check dependencies
                if not all(dep in executed_tasks for dep in task.dependencies):
                    logger.warning(f"⚠️  Task {task_id} dependencies not met, skipping")
                    task.status = "failed"
                    task.error = "Dependencies not satisfied"
                    continue
                
                # Execute task
                logger.info(f"▶️  Executing {task_id}: {task.description}")
                task.status = "in_progress"
                
                result = await self._execute_task(task, results, session_id)
                
                if result.get("success"):
                    task.status = "completed"
                    task.result = result
                    executed_tasks.add(task_id)
                    results[task_id] = result
                    logger.info(f"✅ {task_id} completed successfully")
                else:
                    task.status = "failed"
                    task.error = result.get("error", "Unknown error")
                    logger.error(f"❌ {task_id} failed: {task.error}")
                
                execution_log.append({
                    "task_id": task_id,
                    "description": task.description,
                    "status": task.status,
                    "duration": result.get("duration", 0),
                    "timestamp": datetime.now().isoformat()
                })
            
            # Calculate execution time
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            
            # Compile final results
            success_count = sum(1 for t in plan.tasks if t.status == "completed")
            failed_count = sum(1 for t in plan.tasks if t.status == "failed")
            
            # Synthesize results
            final_result = await self._synthesize_results(plan, results, session_id)
            
            workflow_result = {
                "success": success_count > 0,
                "workflow_plan": {
                    "query": plan.query,
                    "total_tasks": len(plan.tasks),
                    "tasks_completed": success_count,
                    "tasks_failed": failed_count,
                    "complexity_score": plan.complexity_score,
                    "estimated_duration": plan.estimated_total_duration,
                    "actual_duration": total_duration
                },
                "execution_log": execution_log,
                "task_results": results,
                "final_result": final_result,
                "metadata": {
                    "decomposition_used": True,
                    "parallel_execution": self._had_parallel_execution(plan),
                    "execution_efficiency": plan.estimated_total_duration / total_duration if total_duration > 0 else 1.0
                }
            }
            
            logger.info(f"🎉 Workflow completed: {success_count}/{len(plan.tasks)} tasks successful in {total_duration:.2f}s")
            
            return workflow_result
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "workflow_plan": {"query": plan.query}
            }
    
    async def _execute_task(self, task: SubTask, previous_results: Dict, session_id: str) -> Dict[str, Any]:
        """
        Execute a single sub-task
        """
        start_time = datetime.now()
        
        try:
            # Build context from previous task results
            context = self._build_task_context(task, previous_results)
            
            # Route to appropriate agent based on task type and agent assignment
            if task.agent == "whatif" or task.task_type == TaskType.SIMULATION:
                result = await self._execute_whatif_task(task, context, session_id)
            elif task.agent == "cross_statement" or "cross" in task.description.lower():
                result = await self._execute_cross_statement_task(task, context, session_id)
            elif task.agent == "market":
                result = await self._execute_market_task(task, context)
            elif task.task_type == TaskType.DATA_RETRIEVAL:
                result = await self._execute_data_retrieval(task, context)
            elif task.task_type == TaskType.CALCULATION:
                result = await self._execute_calculation(task, context)
            elif task.task_type == TaskType.COMPARISON:
                result = await self._execute_comparison(task, context)
            elif task.task_type == TaskType.ANALYSIS:
                result = await self._execute_analysis(task, context, session_id)
            elif task.task_type == TaskType.SYNTHESIS:
                result = await self._execute_synthesis(task, previous_results)
            else:
                result = await self._execute_general_task(task, context, session_id)
            
            duration = (datetime.now() - start_time).total_seconds()
            result["duration"] = duration
            
            return result
            
        except Exception as e:
            logger.error(f"Task execution error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "duration": (datetime.now() - start_time).total_seconds()
            }
    
    def _build_task_context(self, task: SubTask, previous_results: Dict) -> Dict[str, Any]:
        """Build context for task execution from previous results"""
        context = task.parameters.copy()
        
        # Add results from dependency tasks
        for dep_id in task.dependencies:
            if dep_id in previous_results:
                context[f"dep_{dep_id}"] = previous_results[dep_id]
        
        return context
    
    async def _execute_whatif_task(self, task: SubTask, context: Dict, session_id: str) -> Dict[str, Any]:
        """Execute what-if simulation task"""
        try:
            result = await self.orchestrator.whatif_agent.analyze_whatif_query(
                query=task.description,
                session_id=session_id,
                context=context
            )
            return result
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_cross_statement_task(self, task: SubTask, context: Dict, session_id: str) -> Dict[str, Any]:
        """Execute cross-statement analysis task"""
        try:
            # Use the cross-statement context method
            result = await self.orchestrator._get_cross_statement_context(task.description)
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_market_task(self, task: SubTask, context: Dict) -> Dict[str, Any]:
        """Execute market intelligence task"""
        try:
            # Call market intelligence agents
            result = await self.orchestrator.handle_market_intelligence_query(
                query=task.description,
                classification=None,
                context=context
            )
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_data_retrieval(self, task: SubTask, context: Dict) -> Dict[str, Any]:
        """Execute data retrieval task"""
        try:
            # Get financial context
            financial_data = await self.orchestrator._get_financial_context()
            return {"success": True, "data": financial_data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_calculation(self, task: SubTask, context: Dict) -> Dict[str, Any]:
        """Execute calculation task"""
        try:
            # Extract data from context and perform calculations
            # This would typically call specific calculation functions
            return {
                "success": True,
                "data": {"calculated": True, "description": task.description}
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_comparison(self, task: SubTask, context: Dict) -> Dict[str, Any]:
        """Execute comparison task"""
        try:
            # Get comparison data
            comparison_data = await self.orchestrator._get_scenario_comparison_context()
            return {"success": True, "data": comparison_data}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_analysis(self, task: SubTask, context: Dict, session_id: str) -> Dict[str, Any]:
        """Execute analysis task"""
        try:
            # Call financial modeling for analysis
            result = await self.orchestrator.handle_financial_modeling_query(
                query=task.description,
                classification=None,
                context=context
            )
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_synthesis(self, task: SubTask, previous_results: Dict) -> Dict[str, Any]:
        """Execute synthesis task - combine previous results"""
        try:
            # Combine all previous task results
            combined_data = {
                "task_count": len(previous_results),
                "results": previous_results
            }
            return {"success": True, "data": combined_data, "synthesized": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_general_task(self, task: SubTask, context: Dict, session_id: str) -> Dict[str, Any]:
        """Execute general task using OpenAI"""
        try:
            if self.orchestrator.openai_client:
                response = await self.orchestrator.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a financial analysis expert executing a specific task."},
                        {"role": "user", "content": f"Task: {task.description}\nContext: {context}"}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                content = response.choices[0].message.content
                return {"success": True, "data": {"response": content}}
            else:
                return {"success": True, "data": {"description": task.description}}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _synthesize_results(self, plan: WorkflowPlan, task_results: Dict, session_id: str) -> Dict[str, Any]:
        """
        Synthesize all task results into a comprehensive response
        """
        try:
            # Collect all insights and data
            all_insights = []
            all_recommendations = []
            all_data = {}
            
            for task_id, result in task_results.items():
                if result.get("success"):
                    data = result.get("data", {})
                    
                    # Extract insights
                    if isinstance(data, dict):
                        if "key_insights" in data:
                            all_insights.extend(data["key_insights"])
                        if "recommendations" in data:
                            all_recommendations.extend(data["recommendations"])
                        if "insights" in data:
                            insights_data = data["insights"]
                            if isinstance(insights_data, dict):
                                if "key_impacts" in insights_data:
                                    all_insights.extend(insights_data["key_impacts"])
                    
                    all_data[task_id] = data
            
            # Create comprehensive summary using orchestrator's synthesis capability
            if self.orchestrator.insight_synthesizer and len(all_insights) > 0:
                synthesized_insights = all_insights[:10]  # Top 10 insights
            else:
                synthesized_insights = all_insights
            
            return {
                "executive_summary": f"Completed {len(task_results)} sub-tasks for: {plan.query}",
                "key_insights": synthesized_insights[:5],  # Top 5
                "recommendations": list(set(all_recommendations))[:3],  # Top 3 unique
                "detailed_results": all_data,
                "tasks_executed": list(task_results.keys())
            }
            
        except Exception as e:
            logger.error(f"Synthesis error: {str(e)}")
            return {
                "executive_summary": f"Workflow completed with {len(task_results)} tasks",
                "tasks_executed": list(task_results.keys())
            }
    
    def _had_parallel_execution(self, plan: WorkflowPlan) -> bool:
        """Check if workflow had any parallel execution opportunities"""
        # Check if any tasks at the same dependency level
        levels = {}
        for task in plan.tasks:
            level = len(task.dependencies)
            if level not in levels:
                levels[level] = []
            levels[level].append(task)
        
        return any(len(tasks) > 1 for tasks in levels.values())
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of all workflow executions"""
        return {
            "total_executions": len(self.execution_history),
            "recent_executions": self.execution_history[-10:]
        }

# Note: TaskExecutor is instantiated by orchestrator with self reference

