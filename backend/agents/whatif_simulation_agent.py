"""
What-If Simulation Agent
Handles interactive scenario analysis with iterative refinement
"""

import logging
import re
from typing import Dict, List, Optional, Any
from openai import AsyncOpenAI
import os
import json

logger = logging.getLogger(__name__)

class WhatIfSimulationAgent:
    """
    AI-powered What-If Simulation Agent for Tesla Financial Analysis
    
    Capabilities:
    - Parse natural language "what-if" queries
    - Translate to simulation parameters
    - Execute simulations using cross-statement API
    - Provide iterative refinement
    - Generate insights and recommendations
    """
    
    def __init__(self):
        self.model_name = "gpt-4o"
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key and api_key != 'your_openai_api_key_here':
            self.client = AsyncOpenAI(api_key=api_key, timeout=60.0)
        else:
            self.client = None
            logger.warning("OpenAI API key not configured for What-If agent")
        
        # Session state for iterative refinement
        self.session_history = {}
    
    async def analyze_whatif_query(self, query: str, session_id: str = "default", context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main entry point for what-if analysis
        
        Args:
            query: Natural language query (e.g., "What if revenue grows 25%?")
            session_id: Session identifier for maintaining conversation context
            context: Optional additional context from previous interactions
        
        Returns:
            Dictionary with simulation parameters, results, insights, and follow-up suggestions
        """
        try:
            logger.info(f"🔍 What-If Query: {query}")
            
            # Initialize session if needed
            if session_id not in self.session_history:
                self.session_history[session_id] = {
                    "queries": [],
                    "simulations": [],
                    "learnings": []
                }
            
            # Parse query to extract simulation parameters
            parameters = await self._parse_query_to_parameters(query, session_id, context)
            
            if not parameters.get("success"):
                return {
                    "success": False,
                    "error": parameters.get("error", "Failed to parse query"),
                    "suggestion": "Try rephrasing your question. Example: 'What if revenue grows 25%?'"
                }
            
            # Execute simulation
            simulation_result = await self._execute_simulation(parameters, session_id)
            
            # Generate insights
            insights = await self._generate_insights(query, parameters, simulation_result, session_id)
            
            # Generate follow-up suggestions for iterative refinement
            follow_ups = await self._generate_follow_ups(query, parameters, simulation_result, session_id)
            
            # Store in session history
            self.session_history[session_id]["queries"].append(query)
            self.session_history[session_id]["simulations"].append({
                "query": query,
                "parameters": parameters,
                "result": simulation_result
            })
            
            return {
                "success": True,
                "query": query,
                "parameters_detected": parameters,
                "simulation_result": simulation_result,
                "insights": insights,
                "follow_up_suggestions": follow_ups,
                "session_id": session_id,
                "can_refine": True
            }
            
        except Exception as e:
            logger.error(f"What-If analysis error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "suggestion": "Please try a simpler query or contact support"
            }
    
    async def _parse_query_to_parameters(self, query: str, session_id: str, context: Optional[Dict]) -> Dict[str, Any]:
        """
        Parse natural language query into simulation parameters using GPT-4o
        """
        try:
            if not self.client:
                # Fallback: rule-based parsing
                return self._rule_based_parsing(query)
            
            # Build prompt with context from previous simulations
            session_context = ""
            if session_id in self.session_history:
                prev_sims = self.session_history[session_id]["simulations"]
                if prev_sims:
                    session_context = f"\n\nPrevious simulations in this session:\n"
                    for sim in prev_sims[-3:]:  # Last 3 simulations
                        session_context += f"- Query: {sim['query']}\n"
                        session_context += f"  Parameters: {sim['parameters'].get('extracted_params', {})}\n"
            
            prompt = f"""
You are a financial simulation parameter extractor. Parse the user's what-if query and extract simulation parameters.

User Query: "{query}"
{session_context}

Extract the following parameters if mentioned:
- revenue_growth: percentage change in revenue (e.g., 0.25 for 25% growth)
- automotive_revenue_growth: specific to automotive revenue
- services_revenue_growth: specific to services/other revenue
- gross_margin: new gross margin target (e.g., 0.22 for 22%)
- gross_margin_automotive: automotive segment gross margin
- operating_margin: operating margin target
- r&d_as_percent_revenue: R&D spending as % of revenue
- capex_as_percent_revenue: Capital expenditure as % of revenue
- days_sales_outstanding: DSO (accounts receivable days)
- days_inventory_outstanding: DIO (inventory days)
- days_payable_outstanding: DPO (accounts payable days)
- tax_rate: effective tax rate
- scenario: which scenario to use as baseline (best/base/worst)

Return a JSON object with:
{{
  "success": true,
  "extracted_params": {{parameter_name: value}},
  "scenario": "base",
  "confidence": 0.9,
  "interpretation": "Brief explanation of what will be simulated",
  "missing_params": ["list of params that might be needed but weren't specified"]
}}

If the query is unclear or not a what-if question, return:
{{
  "success": false,
  "error": "Could not parse query as a what-if simulation",
  "suggestion": "Example query format"
}}

IMPORTANT: Return ONLY valid JSON, no markdown formatting or code blocks.
"""
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial parameter extraction expert. Always return valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up markdown code blocks if present
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\n', '', content)
                content = re.sub(r'\n```$', '', content)
            
            result = json.loads(content)
            logger.info(f"✅ Parsed parameters: {result.get('extracted_params', {})}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing query with AI: {str(e)}")
            # Fallback to rule-based parsing
            return self._rule_based_parsing(query)
    
    def _rule_based_parsing(self, query: str) -> Dict[str, Any]:
        """
        Fallback rule-based parameter extraction
        """
        query_lower = query.lower()
        params = {}
        
        # Revenue growth patterns
        revenue_match = re.search(r'revenue.*?(\d+)%', query_lower)
        if revenue_match:
            params['automotive_revenue_growth'] = float(revenue_match.group(1)) / 100
        
        # Margin patterns
        margin_match = re.search(r'(?:gross )?margin.*?(\d+)%', query_lower)
        if margin_match:
            params['gross_margin_automotive'] = float(margin_match.group(1)) / 100
        
        # CapEx patterns
        capex_match = re.search(r'capex.*?(\d+)%', query_lower)
        if capex_match:
            params['capex_as_percent_revenue'] = float(capex_match.group(1)) / 100
        
        # Working capital patterns
        dso_match = re.search(r'dso.*?(\d+)', query_lower)
        if dso_match:
            params['days_sales_outstanding'] = float(dso_match.group(1))
        
        dio_match = re.search(r'dio.*?(\d+)', query_lower)
        if dio_match:
            params['days_inventory_outstanding'] = float(dio_match.group(1))
        
        dpo_match = re.search(r'dpo.*?(\d+)', query_lower)
        if dpo_match:
            params['days_payable_outstanding'] = float(dpo_match.group(1))
        
        if params:
            return {
                "success": True,
                "extracted_params": params,
                "scenario": "base",
                "confidence": 0.7,
                "interpretation": f"Simulating changes: {', '.join(params.keys())}",
                "method": "rule_based"
            }
        else:
            return {
                "success": False,
                "error": "Could not extract simulation parameters from query",
                "suggestion": "Try: 'What if revenue grows 25%?' or 'What if gross margin improves to 22%?'"
            }
    
    async def _execute_simulation(self, parameters: Dict, session_id: str) -> Dict[str, Any]:
        """
        Execute the simulation using the cross-statement API
        """
        try:
            import aiohttp
            
            scenario = parameters.get("scenario", "base")
            extracted_params = parameters.get("extracted_params", {})
            
            # Build simulation request
            sim_request = {
                **extracted_params,
                "include_ai_insights": True
            }
            
            # Call the simulation API
            url = f"http://127.0.0.1:8002/api/tesla/model/{scenario}/simulate-all-statements"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=sim_request, timeout=aiohttp.ClientTimeout(total=60)) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info("✅ Simulation executed successfully")
                        return {
                            "success": True,
                            "data": data,
                            "scenario": scenario
                        }
                    else:
                        error_text = await response.text()
                        logger.error(f"Simulation API error: {response.status} - {error_text}")
                        return {
                            "success": False,
                            "error": f"Simulation failed: {error_text[:200]}"
                        }
            
        except Exception as e:
            logger.error(f"Simulation execution error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _generate_insights(self, query: str, parameters: Dict, simulation_result: Dict, session_id: str) -> Dict[str, Any]:
        """
        Generate AI insights from simulation results
        """
        try:
            if not self.client or not simulation_result.get("success"):
                return self._fallback_insights(parameters, simulation_result)
            
            sim_data = simulation_result.get("data", {})
            deltas = sim_data.get("deltas", {})
            key_metrics = sim_data.get("key_metrics", {})
            ai_insights = sim_data.get("ai_insights", {})
            
            # Build comprehensive prompt
            prompt = f"""
Analyze this Tesla financial simulation and provide actionable insights.

User Query: "{query}"
Parameters Changed: {parameters.get('extracted_params', {})}

Key Results:
- Revenue CAGR: {key_metrics.get('revenue_cagr', 'N/A')}%
- FCF Margin: {key_metrics.get('free_cash_flow_margin', 'N/A')}%
- ROIC: {key_metrics.get('roic', 'N/A')}%

Income Statement Changes (2029):
- Revenue Change: ${deltas.get('income_statement', {}).get('2029', {}).get('total_revenue', 0) / 1e9:.2f}B
- Operating Income Change: ${deltas.get('income_statement', {}).get('2029', {}).get('operating_income', 0) / 1e9:.2f}B
- Net Income Change: ${deltas.get('income_statement', {}).get('2029', {}).get('net_income', 0) / 1e9:.2f}B

AI Analysis from Simulation: {ai_insights.get('executive_summary', 'N/A')[:300]}

Provide:
1. Executive Summary (2-3 sentences)
2. Key Impacts (3-5 bullet points with numbers)
3. Strategic Implications (2-3 points)
4. Risks & Considerations (2-3 points)
5. Recommendation (1-2 sentences)

Return as JSON:
{{
  "executive_summary": "...",
  "key_impacts": ["impact 1", "impact 2", ...],
  "strategic_implications": ["implication 1", ...],
  "risks": ["risk 1", ...],
  "recommendation": "..."
}}
"""
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst providing simulation insights. Be specific and quantitative."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1200
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean up markdown
            if content.startswith("```"):
                content = re.sub(r'^```(?:json)?\n', '', content)
                content = re.sub(r'\n```$', '', content)
            
            insights = json.loads(content)
            logger.info("✅ Generated AI insights")
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return self._fallback_insights(parameters, simulation_result)
    
    def _fallback_insights(self, parameters: Dict, simulation_result: Dict) -> Dict[str, Any]:
        """Fallback insights when AI is unavailable"""
        params = parameters.get('extracted_params', {})
        return {
            "executive_summary": f"Simulation applied {len(params)} parameter changes to Tesla's financial model.",
            "key_impacts": [
                f"Changed parameters: {', '.join(params.keys())}",
                "Impact calculated across all financial statements",
                "Results available in simulation_result"
            ],
            "strategic_implications": [
                "Review detailed financial statements for full impact",
                "Consider sensitivity to key assumptions"
            ],
            "risks": [
                "Assumptions may not reflect real market conditions",
                "External factors not captured in model"
            ],
            "recommendation": "Review simulation results and adjust parameters as needed."
        }
    
    async def _generate_follow_ups(self, query: str, parameters: Dict, simulation_result: Dict, session_id: str) -> List[str]:
        """
        Generate follow-up suggestions for iterative refinement
        """
        try:
            params = parameters.get('extracted_params', {})
            
            follow_ups = []
            
            # Based on what was changed, suggest related analyses
            if any('revenue' in k for k in params.keys()):
                follow_ups.append("What if we also increase operating expenses by 10%?")
                follow_ups.append("What if margin improves to 22% with this revenue growth?")
            
            if any('margin' in k for k in params.keys()):
                follow_ups.append("What if we achieve this margin while growing revenue 20%?")
                follow_ups.append("How would this margin affect free cash flow?")
            
            if any('capex' in k for k in params.keys()):
                follow_ups.append("What if we reduce CapEx but maintain growth?")
                follow_ups.append("How does this CapEx level affect cash position?")
            
            if any('days_' in k for k in params.keys()):
                follow_ups.append("What if we optimize all working capital metrics together?")
                follow_ups.append("How does working capital efficiency affect cash flow?")
            
            # Default suggestions
            if not follow_ups:
                follow_ups = [
                    "What if we increase revenue by 25%?",
                    "What if gross margin improves to 22%?",
                    "What if we reduce CapEx to 7% of revenue?"
                ]
            
            return follow_ups[:3]  # Return top 3
            
        except Exception as e:
            logger.error(f"Error generating follow-ups: {str(e)}")
            return [
                "What if revenue grows 20%?",
                "What if margins improve?",
                "What if CapEx decreases?"
            ]
    
    def clear_session(self, session_id: str):
        """Clear session history for fresh start"""
        if session_id in self.session_history:
            del self.session_history[session_id]
            logger.info(f"Cleared session: {session_id}")
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of session history"""
        if session_id not in self.session_history:
            return {"session_id": session_id, "queries": 0, "simulations": 0}
        
        history = self.session_history[session_id]
        return {
            "session_id": session_id,
            "queries": len(history["queries"]),
            "simulations": len(history["simulations"]),
            "recent_queries": history["queries"][-5:],
            "learnings": history.get("learnings", [])
        }

# Singleton instance
whatif_simulation_agent = WhatIfSimulationAgent()

