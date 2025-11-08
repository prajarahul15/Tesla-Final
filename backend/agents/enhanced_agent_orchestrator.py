"""
Enhanced Agent Orchestrator with Market Intelligence Integration
Routes queries to appropriate agents (financial modeling vs market intelligence)
Now with Multi-Agent Coordination for shared context and intelligent synthesis
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os

from .enhanced_query_classifier import EnhancedQueryClassifier, QueryType
from .stock_data_agent import StockDataAgent
from .market_sentiment_agent import MarketSentimentAgent
from .competitor_analysis_agent import CompetitorAnalysisAgent
from .risk_monitoring_agent import RiskMonitoringAgent
from .shared_context import SharedContext, FactConfidence
from .insight_synthesizer import InsightSynthesizer
from .whatif_simulation_agent import whatif_simulation_agent

# Import Cross-Statement Insights Agent from services
import sys
from pathlib import Path
backend_path = Path(__file__).parent.parent
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))
from services.ai_agents import cross_statement_insights_agent

# Import Task Decomposition components
from .task_decomposer import task_decomposer, WorkflowPlan
from .task_executor import TaskExecutor

logger = logging.getLogger(__name__)

class EnhancedAgentOrchestrator:
    """
    Enhanced Agent Orchestrator that can handle both financial modeling
    and market intelligence queries through intelligent routing
    """
    
    def __init__(self):
        # Core components
        self.query_classifier = EnhancedQueryClassifier()
        self.stock_data_agent = StockDataAgent()
        
        # Multi-Agent Coordination components (NEW!)
        self.shared_context = None  # Created per query
        self.insight_synthesizer = InsightSynthesizer()
        
        # Import existing agents (these should already exist in your system)
        try:
            from .financial_modeling_agent import FinancialModelingAgent
            self.financial_modeling_agent = FinancialModelingAgent()
        except ImportError:
            logger.warning("FinancialModelingAgent not found, using mock")
            self.financial_modeling_agent = None
        
        # Market intelligence agents (Phase 2 - NOW IMPLEMENTED!)
        self.market_sentiment_agent = MarketSentimentAgent()
        self.competitor_analysis_agent = CompetitorAnalysisAgent()
        self.risk_monitoring_agent = RiskMonitoringAgent() # Phase 3 - NOW IMPLEMENTED!
        
        # What-If Simulation Agent (NEW!)
        self.whatif_agent = whatif_simulation_agent
        
        # Cross-Statement Insights Agent (NOW INTEGRATED!)
        self.cross_statement_agent = cross_statement_insights_agent
        
        # Task Decomposition System (NEW!)
        self.task_decomposer = task_decomposer
        self.task_executor = TaskExecutor(self)  # Pass self for agent access
        self.enable_task_decomposition = True  # Can be toggled
        
        # RAG Service (NEW!)
        try:
            from services.rag import RAGService
            self.rag_service = RAGService()
            logger.info("✅ RAG Service initialized")
        except Exception as e:
            logger.warning(f"⚠️ RAG Service not available: {e}")
            self.rag_service = None
        
        # OpenAI client for response generation
        try:
            from openai import AsyncOpenAI
            self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        except ImportError:
            logger.warning("OpenAI client not available")
            self.openai_client = None
    
    async def execute_workflow(self, query: str, session_id: str = None, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for processing user queries
        Routes to appropriate agents based on query classification
        NOW WITH MULTI-AGENT COORDINATION!
        """
        try:
            logger.info(f"🚀 Processing query with Multi-Agent Coordination: {query}")
            
            # 0. Initialize shared context for this query
            self.shared_context = SharedContext(query=query)
            logger.info("✅ Shared context initialized")
            
            # 0.1. Check if task decomposition should be used
            if self.enable_task_decomposition:
                workflow_plan = await self.task_decomposer.decompose_query(query, context)
                
                # If complex query needing decomposition
                if workflow_plan.complexity_score >= 0.4:
                    logger.info(f"🔧 Using task decomposition (complexity: {workflow_plan.complexity_score:.2f})")
                    logger.info(self.task_decomposer.visualize_workflow(workflow_plan))
                    
                    # Execute workflow with task decomposition
                    workflow_result = await self.task_executor.execute_workflow(workflow_plan, session_id or "default")
                    
                    # Format result to match standard response structure
                    return self._format_workflow_result(workflow_result, query, classification=None)
            
            # 0.5. Check if this is a What-If query (takes precedence)
            if self._is_whatif_query(query):
                logger.info("🎯 Detected What-If query - routing to WhatIfSimulationAgent")
                result = await self.handle_whatif_query(query, session_id, context or {})
                return result
            
            # 1. Classify the query
            classification = await self.query_classifier.classify_query(query)
            
            # 2. Check for document summarization queries FIRST (before routing)
            # These should go to financial modeling handler for RAG access
            query_lower_for_doc_check = query.lower()
            is_doc_summary_query = any(word in query_lower_for_doc_check for word in ['summarize', 'summary', 'summarise']) and any(word in query_lower_for_doc_check for word in ['report', 'document', 'impact report', '10-k', '10k', 'annual report'])
            
            if is_doc_summary_query:
                logger.info(f"📄 Document summarization query detected - routing to financial modeling handler")
                # Override classification to ensure it goes to financial modeling
                result = await self.handle_financial_modeling_query(query, classification, context or {})
            # 2. Route to appropriate handler based on classification
            elif classification.query_type == QueryType.MARKET_INTELLIGENCE:
                result = await self.handle_market_intelligence_query(query, classification, context or {})
            elif classification.query_type == QueryType.FINANCIAL_MODELING:
                result = await self.handle_financial_modeling_query(query, classification, context or {})
            elif classification.query_type == QueryType.HYBRID:
                result = await self.handle_hybrid_query(query, classification, context or {})
            else:
                result = await self.handle_general_query(query, classification, context or {})
            
            # 3. Enhance result with metadata and coordination info
            coordination_summary = self.shared_context.get_summary() if self.shared_context else {}
            
            result.update({
                "query_type": classification.query_type.value,
                "confidence": classification.confidence,
                "detected_categories": classification.detected_categories,
                "agents_used": result.get("agents_used", []),
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "coordination": {
                    "enabled": True,
                    "facts_shared": coordination_summary.get("total_facts", 0),
                    "conflicts_detected": coordination_summary.get("conflicts_detected", 0),
                    "agents_coordinated": len(coordination_summary.get("agents_contributed", []))
                }
            })
            
            logger.info(f"✅ Query processed with {coordination_summary.get('total_facts', 0)} shared facts")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in enhanced orchestrator workflow: {str(e)}")
            return {
                "error": f"Failed to process query: {str(e)}",
                "query_type": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    async def handle_market_intelligence_query(self, query: str, classification, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle market intelligence related queries
        """
        try:
            logger.info(f"Handling market intelligence query: {query}")
            logger.info(f"Detected categories: {classification.detected_categories}")
            logger.info(f"Checking for market_market_sentiment...")
            
            agents_used = []
            analysis_results = {}
            
            # Route to specific market intelligence agents based on detected categories
            if "market_stock_performance" in classification.detected_categories:
                stock_analysis = await self.stock_data_agent.analyze_query(query, context)
                analysis_results["stock_analysis"] = stock_analysis
                agents_used.append("StockDataAgent")
            
            # Phase 2: Market Sentiment Agent
            if "market_market_sentiment" in classification.detected_categories:
                try:
                    logger.info("Routing to Market Sentiment Agent for news/sentiment analysis")
                    sentiment_analysis = await self.market_sentiment_agent.analyze_query(query, context)
                    logger.info(f"Market Sentiment Agent response: {sentiment_analysis}")
                    analysis_results["sentiment_analysis"] = sentiment_analysis
                    agents_used.append("MarketSentimentAgent")
                except Exception as e:
                    logger.error(f"Error in Market Sentiment Agent: {str(e)}")
                    import traceback
                    traceback.print_exc()
            
            # Phase 2: Competitor Analysis Agent
            if "market_competitor_analysis" in classification.detected_categories:
                competitor_analysis = await self.competitor_analysis_agent.analyze_query(query, context)
                analysis_results["competitor_analysis"] = competitor_analysis
                agents_used.append("CompetitorAnalysisAgent")
            
            # Phase 3: Risk Monitoring Agent
            query_lower = query.lower()
            if any(keyword in query_lower for keyword in ["risk", "alert", "threat", "danger", "volatility", "monitor"]):
                risk_analysis = await self.risk_monitoring_agent.analyze_query(query, context)
                analysis_results["risk_monitoring"] = risk_analysis
                agents_used.append("RiskMonitoringAgent")
            
            # Generate comprehensive response
            response = await self.generate_market_intelligence_response(query, analysis_results, classification)
            
            return {
                "executive_summary": response["executive_summary"],
                "analysis_sections": response.get("analysis_sections", []),
                "key_insights": response.get("key_insights", []),
                "recommendations": response.get("recommendations", []),
                "data_sources": response.get("data_sources", ["Yahoo Finance", "Alpha Vantage"]),
                "agents_used": agents_used,
                "tasks_executed": ["market_intelligence_analysis"],
                "market_data": analysis_results
            }
            
        except Exception as e:
            logger.error(f"Error handling market intelligence query: {str(e)}")
            return {
                "error": f"Failed to process market intelligence query: {str(e)}",
                "agents_used": [],
                "executive_summary": "Unable to process market intelligence query at this time."
            }
    
    async def handle_financial_modeling_query(self, query: str, classification, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle financial modeling related queries
        Uses OpenAI to answer based on available financial data
        """
        try:
            logger.info(f"Handling financial modeling query: {query}")
            
            # Use existing financial modeling agent if available
            if self.financial_modeling_agent:
                result = await self.financial_modeling_agent.process_query(query, context)
                return {
                    **result,
                    "agents_used": ["FinancialModelingAgent"],
                    "tasks_executed": ["financial_modeling_analysis"]
                }
            
            # Use OpenAI to answer financial questions with context
            if self.openai_client:
                logger.info("Using OpenAI for financial modeling analysis")
                
                # Check if this is a user-uploaded document query (FRA)
                is_user_document_query = context.get("user_document", False)
                
                # For user document queries, ONLY use the document content (no Tesla KB or financial model data)
                if is_user_document_query:
                    logger.info("📄 User document query detected - using ONLY uploaded document, no Tesla KB or financial model data")
                    financial_context = {
                        "user_document_only": True,
                        "file_id": context.get("file_id"),
                        "user_id": context.get("user_id")
                    }
                    # Note: Document chunks are already in the query text from FRA endpoint
                    query_lower = query.lower()  # Still need this for system prompt check later
                    is_document_summary_query = False  # Not applicable for user documents
                else:
                    # Check for document summarization queries FIRST
                    query_lower = query.lower()
                    is_document_summary_query = any(word in query_lower for word in ['summarize', 'summary', 'summarise']) and any(word in query_lower for word in ['report', 'document', 'impact report', '10-k', '10k', 'annual report'])
                    
                    # Build context from available APIs
                    # For document summarization queries, we still get context but will prioritize RAG documents
                    financial_context = await self._get_financial_context(query=query)
                    logger.info(f"Financial context keys: {list(financial_context.keys())}")
                
                # Phase 3: Detect temporal reasoning (skip for user documents)
                if not is_user_document_query and self.rag_service:
                    temporal_info = self.rag_service.temporal_reasoning.detect_temporal_query(query)
                    if temporal_info.get("has_temporal"):
                        financial_context['temporal_info'] = temporal_info
                        logger.info(f"🕐 Temporal query detected: {temporal_info['type']}, years: {temporal_info.get('years', [])}")
                
                # For document summarization queries, mark it so prompt builder can prioritize RAG documents
                if not is_user_document_query and is_document_summary_query:
                    financial_context['_is_document_summary'] = True
                if not is_user_document_query and is_document_summary_query:
                    logger.info(f"📄 Detected document summarization query - retrieving more chunks")
                    # Retrieve more chunks for comprehensive summarization (increase top_k)
                    if self.rag_service and 'rag_documents' in financial_context:
                        # Get source filename from query or retrieved documents
                        retrieved_docs = financial_context['rag_documents']
                        if retrieved_docs:
                            # Try to find matching document name in query
                            query_words = query_lower.split()
                            for doc in retrieved_docs[:3]:  # Check top 3
                                source_name = doc.get('metadata', {}).get('source', '').lower()
                                if any(word in source_name for word in query_words):
                                    logger.info(f"📚 Found matching document: {source_name} - retrieving more chunks")
                                    # Retrieve more chunks from this specific document
                                    enhanced_query = f"{query} {source_name}"
                                    additional_docs = self.rag_service.retrieve_documents(
                                        query=enhanced_query,
                                        top_k=15  # Get more chunks for summarization
                                    )
                                    if additional_docs:
                                        # Filter to prioritize the matching document
                                        matching_docs = [d for d in additional_docs if source_name in d.get('metadata', {}).get('source', '').lower()]
                                        if matching_docs:
                                            financial_context['rag_documents'] = matching_docs[:10] + retrieved_docs[:5]
                                            logger.info(f"✅ Enhanced RAG retrieval: {len(financial_context['rag_documents'])} total chunks")
                
                # Build prompt based on query type
                is_user_document_query = financial_context.get('user_document_only', False)
                
                if is_user_document_query:
                    # For user document queries, use isolated prompt (no Tesla KB or financial model data)
                    prompt = self._build_user_document_prompt(query=query)
                    logger.info("📄 Using user document-only prompt (isolated from Tesla KB)")
                else:
                    # For regular queries, use full financial prompt
                    is_document_summary_query = financial_context.get('_is_document_summary', False)
                    prompt = self._build_financial_prompt(
                        query=query,
                        financial_context=financial_context,
                        classification=classification,
                        is_document_summary_query=is_document_summary_query
                    )
                
                # Skip these for user document queries (they would pull Tesla KB or financial model data)
                if not is_user_document_query:
                    # Check for cross-statement simulation queries
                    if any(word in query_lower for word in ['cross-statement', 'cross statement', 'integrated', 'all statements', 'balance sheet impact', 'cash flow impact']):
                        logger.info(f"🔗 Query contains cross-statement keywords - fetching cross-statement data")
                        try:
                            financial_context['cross_statement_data'] = await self._get_cross_statement_context(query)
                            logger.info("✅ Cross-statement data added successfully")
                        except Exception as e:
                            logger.error(f"❌ Failed to fetch cross-statement data: {e}", exc_info=True)
                    
                    # Check for scenario comparison queries
                    if any(word in query_lower for word in ['compare scenarios', 'best vs base vs worst', 'scenario comparison', 'compare best', 'compare base', 'compare worst']):
                        logger.info(f"📊 Query contains scenario comparison keywords - fetching comparison data")
                        try:
                            financial_context['scenario_comparison'] = await self._get_scenario_comparison_context()
                            logger.info("✅ Scenario comparison data added successfully")
                        except Exception as e:
                            logger.error(f"❌ Failed to fetch scenario comparison data: {e}", exc_info=True)
                    
                    # For simulation queries, pre-calculate and inject results
                    if any(word in query.lower() for word in ['simulate', 'growth', 'increase', 'impact']):
                        logger.info(f"🎯 Query contains simulation keywords - calling _add_simulation_calculations")
                        try:
                            prompt = self._add_simulation_calculations(prompt, query, financial_context)
                            logger.info("✅ Simulation calculations added successfully")
                        except Exception as e:
                            logger.error(f"❌ Failed to add simulation calculations: {e}", exc_info=True)
                
                # Enhanced system prompt - different for user documents vs regular queries
                if is_user_document_query:
                    system_prompt = """You are a Financial Document Analyst. Analyze user-uploaded documents ONLY.

CRITICAL INSTRUCTIONS:
1. You MUST base your response ONLY on the document content provided in the user's query
2. DO NOT use any Tesla Knowledge Base documents, financial model forecasts, or external data
3. DO NOT reference Tesla's 10-K reports, Impact Reports, or any pre-existing documents
4. DO NOT use vehicle forecast data, financial statements, or any model projections
5. If the document doesn't contain the information, explicitly state: "This information is not available in the uploaded document"
6. Answer based SOLELY on what is provided in the document excerpts

REQUIRED FORMAT:

**Executive Summary**
[2-3 complete sentences based ONLY on the uploaded document]

**Financial Impact** (if applicable)
[Use only data from the uploaded document]

**Key Insights**
• [Insight 1 - from document only]
• [Insight 2 - from document only]
• [Insight 3 - from document only]

**Recommendations**
• [Recommendation 1 - based on document content]
• [Recommendation 2 - based on document content]

Keep response under 600 words."""
                else:
                    system_prompt = """You are Tesla's Financial Analyst. Provide a COMPLETE, CONCISE response.

CRITICAL INSTRUCTION:
If the prompt contains "PRE-CALCULATED SIMULATION RESULTS", you MUST:
1. COPY the table EXACTLY as shown
2. Do NOT recalculate any values
3. Do NOT change any numbers
4. Just add Executive Summary, Insights, and Recommendations around it

REQUIRED FORMAT:

**Executive Summary**
[2-3 complete sentences]

**Financial Impact**
[COPY the pre-calculated table if provided, otherwise create your own with COGS calculated]

**Key Insights**
• [Complete insight 1]
• [Complete insight 2]
• [Complete insight 3]

**Recommendations**
• [Complete recommendation 1]
• [Complete recommendation 2]

Keep under 600 words total."""

                # Get response from OpenAI
                logger.info(f"Calling GPT-4o with prompt length: {len(prompt)} chars")
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o",  # Best balance of speed, cost, and quality
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,  # Low for more consistent analysis
                    max_tokens=1500,  # Moderate length for balanced responses
                    timeout=60.0
                )
                
                answer = response.choices[0].message.content
                logger.info(f"GPT-4o response received - length: {len(answer) if answer else 0} chars")
                
                if not answer or len(answer.strip()) == 0:
                    logger.error("GPT-4o returned empty response!")
                    raise ValueError("AI model returned empty response")
                
                # Phase 3: Extract and verify citations
                citations_result = {}
                if self.rag_service and 'rag_documents' in financial_context:
                    retrieved_docs = financial_context.get('rag_documents', [])
                    if answer:  # Ensure answer is not None or empty
                        citations_result = self.rag_service.extract_and_verify_citations(
                            str(answer),  # Ensure it's a string
                            retrieved_docs
                        )
                        logger.info(f"📝 Citation extraction: {citations_result.get('verified_count', 0)} verified")
                    else:
                        logger.warning("⚠️ Empty answer, skipping citation extraction")
                
                # Parse response into structured format
                result = {
                    "executive_summary": answer,
                    "key_insights": self._extract_insights_from_response(answer),
                    "recommendations": self._extract_recommendations_from_response(answer),
                    "agents_used": ["OpenAI Financial Analyst"],
                    "tasks_executed": ["financial_analysis_with_llm"],
                    "data_sources": ["Tesla Financial API", "Vehicle Data", "Financial Statements"]
                }
                
                # Add Phase 3 features to response
                if citations_result:
                    result["citations"] = citations_result
                if 'temporal_info' in financial_context:
                    result["temporal_context"] = financial_context['temporal_info']
                
                return result
            else:
                # Fallback if OpenAI is not available
                logger.warning("OpenAI not available for financial modeling")
                return {
                    "executive_summary": "Financial analysis requires OpenAI API. Please configure OPENAI_API_KEY in .env file.",
                    "key_insights": [
                        "OpenAI API key not configured",
                        "Financial data is available via API endpoints",
                        "Configure API key to enable AI-powered financial analysis"
                    ],
                    "recommendations": [
                        "Add OPENAI_API_KEY to backend/.env file",
                        "Use financial statement pages for detailed analysis",
                        "Try market intelligence questions"
                    ],
                    "agents_used": [],
                    "tasks_executed": ["fallback_response"]
                }
                
        except Exception as e:
            logger.error(f"Error handling financial modeling query: {type(e).__name__}: {str(e)}", exc_info=True)
            import traceback
            error_details = traceback.format_exc()
            logger.error(f"Full traceback: {error_details}")
            return {
                "error": f"Failed to process financial modeling query: {type(e).__name__}: {str(e)}",
                "executive_summary": f"Unable to process financial modeling query. Error: {type(e).__name__}. Please try rephrasing your question or check the backend logs for details."
            }
    
    async def handle_hybrid_query(self, query: str, classification, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle queries that span both financial modeling and market intelligence
        """
        try:
            logger.info(f"Handling hybrid query: {query}")
            
            # Execute both financial and market intelligence analysis in parallel
            tasks = []
            
            # Market intelligence analysis
            market_task = self.handle_market_intelligence_query(query, classification, context)
            tasks.append(market_task)
            
            # Financial modeling analysis
            financial_task = self.handle_financial_modeling_query(query, classification, context)
            tasks.append(financial_task)
            
            # Wait for both analyses to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            market_result = results[0] if not isinstance(results[0], Exception) else {}
            financial_result = results[1] if not isinstance(results[1], Exception) else {}
            
            # Combine results
            combined_result = await self.combine_hybrid_results(query, market_result, financial_result, classification)
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error handling hybrid query: {str(e)}")
            return {
                "error": f"Failed to process hybrid query: {str(e)}",
                "executive_summary": "Unable to process hybrid query at this time."
            }
    
    async def handle_general_query(self, query: str, classification, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle general queries that don't clearly fit into financial or market categories
        """
        try:
            logger.info(f"Handling general query: {query}")
            
            # Try to provide helpful suggestions
            suggestions = self.query_classifier.get_all_suggestions()
            
            return {
                "executive_summary": "I can help you with Tesla's financial modeling and market intelligence analysis. Here are some examples of what you can ask:",
                "key_insights": [
                    "Financial Analysis: Revenue, costs, profitability, forecasting",
                    "Market Intelligence: Stock performance, sentiment, competitor analysis",
                    "Hybrid Analysis: How market factors affect financial projections"
                ],
                "recommendations": [
                    "Try asking about Tesla's stock performance",
                    "Ask about financial metrics like revenue or margins",
                    "Request competitor analysis or market sentiment"
                ],
                "suggestions": suggestions,
                "agents_used": [],
                "tasks_executed": ["general_guidance"]
            }
            
        except Exception as e:
            logger.error(f"Error handling general query: {str(e)}")
            return {
                "error": f"Failed to process general query: {str(e)}",
                "executive_summary": "Unable to process query at this time."
            }
    
    def _is_whatif_query(self, query: str) -> bool:
        """Detect if query is a what-if simulation request"""
        query_lower = query.lower()
        
        whatif_patterns = [
            "what if", "what-if", "suppose", "imagine", "simulate",
            "if we", "if i", "how would", "what would happen",
            "what happens if", "impact of", "effect of"
        ]
        
        has_whatif = any(pattern in query_lower for pattern in whatif_patterns)
        has_param = any(keyword in query_lower for keyword in [
            "revenue", "margin", "growth", "capex", "working capital",
            "%", "percent", "increase", "decrease", "improve", "reduce"
        ])
        
        return has_whatif and has_param
    
    async def handle_whatif_query(self, query: str, session_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle what-if simulation queries"""
        try:
            logger.info(f"🎯 Processing What-If query: {query}")
            
            whatif_result = await self.whatif_agent.analyze_whatif_query(
                query=query,
                session_id=session_id or "default",
                context=context
            )
            
            if not whatif_result.get("success"):
                return {
                    "executive_summary": whatif_result.get("error", "Unable to process what-if query"),
                    "key_insights": [whatif_result.get("suggestion", "")],
                    "recommendations": ["Try rephrasing your question", "Example: 'What if revenue grows 25%?'"],
                    "agents_used": ["WhatIfSimulationAgent"],
                    "tasks_executed": ["whatif_analysis_failed"],
                    "query_type": "whatif"
                }
            
            insights = whatif_result.get("insights", {})
            sim_data = whatif_result.get("simulation_result", {}).get("data", {})
            
            key_insights = []
            if insights.get("key_impacts"):
                key_insights.extend(insights["key_impacts"][:5])
            
            recommendations = []
            if insights.get("recommendation"):
                recommendations.append(insights["recommendation"])
            if insights.get("strategic_implications"):
                recommendations.extend(insights["strategic_implications"][:2])
            
            follow_ups = whatif_result.get("follow_up_suggestions", [])
            
            return {
                "executive_summary": insights.get("executive_summary", "What-if simulation completed"),
                "key_insights": key_insights,
                "recommendations": recommendations,
                "analysis_sections": [
                    {
                        "title": "What-If Simulation Results",
                        "content": insights.get("executive_summary", ""),
                        "data_type": "simulation"
                    },
                    {
                        "title": "Strategic Implications",
                        "content": "\n".join(f"• {impl}" for impl in insights.get("strategic_implications", [])),
                        "data_type": "strategy"
                    },
                    {
                        "title": "Risks & Considerations",
                        "content": "\n".join(f"• {risk}" for risk in insights.get("risks", [])),
                        "data_type": "risk"
                    }
                ],
                "simulation_data": {
                    "parameters": whatif_result.get("parameters_detected", {}),
                    "key_metrics": sim_data.get("key_metrics", {}),
                    "can_refine": True,
                    "follow_up_suggestions": follow_ups
                },
                "agents_used": ["WhatIfSimulationAgent"],
                "tasks_executed": ["whatif_parameter_extraction", "simulation_execution", "insight_generation"],
                "query_type": "whatif",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error handling what-if query: {str(e)}", exc_info=True)
            return {
                "error": f"Failed to process what-if query: {str(e)}",
                "executive_summary": "Unable to complete what-if simulation.",
                "query_type": "whatif"
            }
    
    async def generate_market_intelligence_response(self, query: str, analysis_results: Dict[str, Any], classification) -> Dict[str, Any]:
        """
        Generate a comprehensive response for market intelligence queries
        """
        try:
            if self.openai_client:
                # Use OpenAI to generate a comprehensive response
                prompt = self._build_market_intelligence_prompt(query, analysis_results, classification)
                
                response = await self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a Tesla market intelligence expert. Provide detailed, data-driven insights about Tesla's stock performance, market sentiment, and competitive position. Be concise but informative."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=1000
                )
                
                content = response.choices[0].message.content
                
                # Parse the response to extract structured information
                return self._parse_market_intelligence_response(content, analysis_results)
            else:
                # Fallback response without OpenAI
                return self._generate_fallback_market_response(analysis_results)
                
        except Exception as e:
            logger.error(f"Error generating market intelligence response: {str(e)}")
            return self._generate_fallback_market_response(analysis_results)
    
    def _build_market_intelligence_prompt(self, query: str, analysis_results: Dict[str, Any], classification) -> str:
        """Build a prompt for OpenAI to generate market intelligence response"""
        
        prompt_parts = [
            f"User Query: {query}",
            "",
            "Based on the following Tesla market data, provide a comprehensive analysis:"
        ]
        
        # Add stock analysis data
        if "stock_analysis" in analysis_results:
            stock_data = analysis_results["stock_analysis"]
            prompt_parts.extend([
                "",
                "Stock Data:",
                f"- Current Price: ${stock_data['analysis']['stock_data']['current_price']:.2f}",
                f"- Daily Change: {stock_data['analysis']['stock_data']['daily_change']:.2f} ({stock_data['analysis']['stock_data']['daily_change_percent']:.1f}%)",
                f"- Volume: {stock_data['analysis']['stock_data']['volume']:,}",
                f"- Market Cap: ${stock_data['analysis']['stock_data']['market_cap']:,.0f}",
                f"- 52-week High: ${stock_data['analysis']['fifty_two_week_analysis']['high']:.2f}",
                f"- 52-week Low: ${stock_data['analysis']['fifty_two_week_analysis']['low']:.2f}"
            ])
            
            if stock_data['analysis']['technical_indicators']:
                tech = stock_data['analysis']['technical_indicators']
                prompt_parts.extend([
                    "",
                    "Technical Indicators:",
                    f"- RSI (14): {tech['rsi_14']:.1f}",
                    f"- MACD: {tech['macd']:.2f}",
                    f"- 50-day SMA: ${tech['sma_50']:.2f}",
                    f"- 200-day SMA: ${tech['sma_200']:.2f}"
                ])
        
        # Add sentiment analysis data
        if "sentiment_analysis" in analysis_results:
            sentiment_data = analysis_results["sentiment_analysis"]
            if sentiment_data.get("sentiment"):
                sentiment = sentiment_data["sentiment"]
                prompt_parts.extend([
                    "",
                    "Market Sentiment:",
                    f"- Overall Sentiment: {sentiment.get('sentiment_data', {}).get('overall_sentiment', 'N/A')}",
                    f"- News Sentiment Score: {sentiment.get('news_analysis', {}).get('score', 0):.2f}",
                    f"- Positive Articles: {sentiment.get('news_analysis', {}).get('positive_count', 0)}",
                    f"- Negative Articles: {sentiment.get('news_analysis', {}).get('negative_count', 0)}",
                    f"- Neutral Articles: {sentiment.get('news_analysis', {}).get('neutral_count', 0)}"
                ])
                
                # Add recent news headlines
                news_articles = sentiment.get('news_analysis', {}).get('recent_news', [])
                if news_articles:
                    prompt_parts.extend(["", "Recent News Headlines:"])
                    for i, article in enumerate(news_articles[:5], 1):
                        prompt_parts.append(f"{i}. {article.get('title', 'N/A')} ({article.get('sentiment', 'N/A')})")
        
        # Add competitor analysis data
        if "competitor_analysis" in analysis_results:
            comp_data = analysis_results["competitor_analysis"]
            if comp_data.get("competitors"):
                prompt_parts.extend(["", "Competitor Landscape:"])
                for comp in comp_data["competitors"][:3]:  # Top 3 competitors
                    prompt_parts.append(f"- {comp.get('name', 'N/A')}: Market Cap ${comp.get('market_cap', 0):,.0f}, YoY Growth {comp.get('yoy_growth', 0):.1f}%")
        
        prompt_parts.extend([
            "",
            "Please provide:",
            "1. Executive summary of Tesla's current market position",
            "2. Key insights about stock performance and market dynamics",
            "3. Recommendations for monitoring Tesla's market performance",
            "",
            "Format your response as structured analysis with clear sections."
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_market_intelligence_response(self, content: str, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Parse OpenAI response into structured format"""
        
        # Simple parsing - in production, you'd want more sophisticated parsing
        lines = content.split('\n')
        
        executive_summary = ""
        key_insights = []
        recommendations = []
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if "executive summary" in line.lower() or "summary" in line.lower():
                current_section = "summary"
            elif "insights" in line.lower() or "key points" in line.lower():
                current_section = "insights"
            elif "recommendations" in line.lower() or "recommend" in line.lower():
                current_section = "recommendations"
            elif line.startswith(('-', '•', '1.', '2.', '3.')):
                if current_section == "insights":
                    key_insights.append(line.lstrip('-•123456789. '))
                elif current_section == "recommendations":
                    recommendations.append(line.lstrip('-•123456789. '))
            elif current_section == "summary" and not line.startswith(('executive', 'summary')):
                executive_summary += line + " "
        
        return {
            "executive_summary": executive_summary.strip() or content[:200] + "...",
            "key_insights": key_insights[:5],  # Limit to 5 insights
            "recommendations": recommendations[:3],  # Limit to 3 recommendations
            "data_sources": ["Yahoo Finance", "Alpha Vantage"]
        }
    
    def _generate_fallback_market_response(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback response when OpenAI is not available"""
        
        insights = []
        recommendations = []
        
        if "stock_analysis" in analysis_results:
            stock_data = analysis_results["stock_analysis"]
            price = stock_data['analysis']['stock_data']['current_price']
            change_pct = stock_data['analysis']['stock_data']['daily_change_percent']
            
            insights.extend([
                f"Tesla is currently trading at ${price:.2f}",
                f"Daily performance: {change_pct:+.1f}%",
                f"Market cap: ${stock_data['analysis']['stock_data']['market_cap']:,.0f}"
            ])
            
            if change_pct > 0:
                recommendations.append("Monitor for continued upward momentum")
            else:
                recommendations.append("Watch for potential support levels")
        
        return {
            "executive_summary": "Tesla market analysis based on current stock data and technical indicators.",
            "key_insights": insights,
            "recommendations": recommendations,
            "data_sources": ["Yahoo Finance", "Alpha Vantage"]
        }
    
    async def combine_hybrid_results(self, query: str, market_result: Dict[str, Any], financial_result: Dict[str, Any], classification) -> Dict[str, Any]:
        """
        Combine results from both market intelligence and financial modeling analysis
        NOW WITH INTELLIGENT SYNTHESIS (not just concatenation!)
        """
        
        try:
            logger.info("🔗 Starting intelligent synthesis of market + financial insights")
            
            # Store agent outputs in shared context
            if self.shared_context:
                self.shared_context.store_agent_output("market_intelligence", market_result)
                self.shared_context.store_agent_output("financial_modeling", financial_result)
                
                # Extract and store key facts from market analysis
                if market_result.get("market_data"):
                    market_data = market_result["market_data"]
                    if "stock_analysis" in market_data:
                        stock_info = market_data["stock_analysis"].get("analysis", {}).get("stock_data", {})
                        self.shared_context.add_fact(
                            "current_stock_price",
                            stock_info.get("current_price"),
                            "market_intelligence",
                            confidence=0.95
                        )
                        self.shared_context.add_fact(
                            "market_sentiment",
                            stock_info.get("daily_change_percent"),
                            "market_intelligence",
                            confidence=0.85
                        )
            
            agents_used = []
            if market_result.get("agents_used"):
                agents_used.extend(market_result["agents_used"])
            if financial_result.get("agents_used"):
                agents_used.extend(financial_result["agents_used"])
            
            # Get insights from both agents
            market_insights = market_result.get("key_insights", [])
            financial_insights = financial_result.get("key_insights", [])
            
            # Use InsightSynthesizer for intelligent combining
            synthesized_insights = self.insight_synthesizer.synthesize(
                market_insights=market_insights,
                financial_insights=financial_insights,
                query_context=query
            )
            
            logger.info(f"✅ Synthesized {len(synthesized_insights)} integrated insights from {len(market_insights)} market + {len(financial_insights)} financial insights")
            
            # Generate integrated recommendations
            market_recs = market_result.get("recommendations", [])
            financial_recs = financial_result.get("recommendations", [])
            
            integrated_recommendations = self.insight_synthesizer.generate_integrated_recommendations(
                market_recs=market_recs,
                financial_recs=financial_recs
            )
            
            # Create integrated executive summary
            market_summary = market_result.get("executive_summary", "Market analysis unavailable")
            financial_summary = financial_result.get("executive_summary", "Financial analysis unavailable")
            
            # Find synergies for summary
            synergies = self.insight_synthesizer._find_synergies(
                self.insight_synthesizer._extract_themes(market_insights, self.insight_synthesizer.market_keywords),
                self.insight_synthesizer._extract_themes(financial_insights, self.insight_synthesizer.financial_keywords)
            )
            
            integrated_summary = self.insight_synthesizer.create_executive_summary(
                market_summary=market_summary,
                financial_summary=financial_summary,
                synergies=synergies
            )
            
            # Check for conflicts in shared context
            conflicts_info = []
            if self.shared_context and self.shared_context.has_conflicts():
                conflicts = self.shared_context.get_conflicts()
                logger.warning(f"⚠️  Detected {len(conflicts)} conflicts between agent outputs")
                for conflict in conflicts:
                    conflicts_info.append({
                        "key": conflict.key,
                        "severity": conflict.severity,
                        "agents": [f.agent_id for f in conflict.facts]
                    })
            
            result = {
                "executive_summary": integrated_summary,
                "analysis_sections": [
                    {
                        "title": "Market Intelligence Analysis",
                        "content": market_summary,
                        "data_type": "market"
                    },
                    {
                        "title": "Financial Modeling Analysis", 
                        "content": financial_summary,
                        "data_type": "financial"
                    }
                ],
                "key_insights": synthesized_insights,  # SYNTHESIZED, not concatenated!
                "recommendations": integrated_recommendations,  # INTEGRATED, not just combined!
                "agents_used": list(set(agents_used)),
                "tasks_executed": ["market_intelligence_analysis", "financial_modeling_analysis", "intelligent_synthesis"],
                "market_data": market_result.get("market_data", {}),
                "financial_data": financial_result.get("financial_data", {}),
                "synthesis_metadata": {
                    "synergies_found": len(synergies),
                    "conflicts_detected": len(conflicts_info),
                    "insights_synthesized": len(synthesized_insights),
                    "original_market_insights": len(market_insights),
                    "original_financial_insights": len(financial_insights)
                }
            }
            
            if conflicts_info:
                result["conflicts"] = conflicts_info
            
            logger.info("✅ Intelligent synthesis complete")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in intelligent synthesis: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fallback to simple combination
            return {
                "executive_summary": f"Combined Analysis: Market analysis indicates current conditions while financial modeling shows fundamental performance.",
                "key_insights": (market_result.get("key_insights", [])[:3] + 
                               financial_result.get("key_insights", [])[:3]),
                "recommendations": (market_result.get("recommendations", [])[:2] + 
                                  financial_result.get("recommendations", [])[:2]),
                "agents_used": list(set(agents_used)),
                "error": f"Synthesis error (using fallback): {str(e)}"
            }
    
    async def _get_financial_context(self, query: str = "") -> Dict[str, Any]:
        """
        Fetch financial context from available API endpoints
        This provides data for OpenAI to generate informed responses
        
        Args:
            query: User query (optional, used for RAG retrieval)
        """
        context = {}
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Try to get vehicle summary for current year (2025)
                try:
                    url = "http://127.0.0.1:8002/api/vehicles/summary/2025"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            context['vehicle_summary_2025'] = data.get('summary', {})
                            logger.info("Fetched vehicle summary for financial context")
                except Exception as e:
                    logger.warning(f"Could not fetch vehicle summary: {str(e)}")
                
                # Try to get vehicle forecast data from autonomous agent
                try:
                    url = "http://127.0.0.1:8002/api/vehicles/forecast/cached"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('success'):
                                context['vehicle_forecasts_cached'] = data
                                logger.info(f"Fetched {data.get('count', 0)} cached vehicle forecasts")
                except Exception as e:
                    logger.warning(f"Could not fetch cached forecasts: {str(e)}")
                
                # Try to get vehicle forecast agent status
                try:
                    url = "http://127.0.0.1:8002/api/vehicles/forecast/agent-status"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            if data.get('success'):
                                context['forecast_agent_status'] = data.get('agent_status', {})
                                logger.info("Fetched vehicle forecast agent status")
                except Exception as e:
                    logger.warning(f"Could not fetch agent status: {str(e)}")
                
                # Try to get financial statements (base case) - all years
                try:
                    url = "http://127.0.0.1:8002/api/tesla/model/base"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            context['financial_statements_base'] = data
                            logger.info("Fetched base case financial statements for financial context")
                except Exception as e:
                    logger.warning(f"Could not fetch base case financial statements: {str(e)}")
                
                # Try to get best case scenario
                try:
                    url = "http://127.0.0.1:8002/api/tesla/model/best"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            context['financial_statements_best'] = data
                            logger.info("Fetched best case financial statements for financial context")
                except Exception as e:
                    logger.warning(f"Could not fetch best case financial statements: {str(e)}")
                
                # Try to get worst case scenario
                try:
                    url = "http://127.0.0.1:8002/api/tesla/model/worst"
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                        if response.status == 200:
                            data = await response.json()
                            context['financial_statements_worst'] = data
                            logger.info("Fetched worst case financial statements for financial context")
                except Exception as e:
                    logger.warning(f"Could not fetch worst case financial statements: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error fetching financial context: {str(e)}")
        
        # Retrieve relevant documents using RAG (NEW!)
        if query and self.rag_service:
            try:
                rag_documents = self.rag_service.retrieve_documents(
                    query=query,
                    top_k=5
                )
                if rag_documents:
                    context['rag_documents'] = rag_documents
                    logger.info(f"📚 Retrieved {len(rag_documents)} relevant documents from RAG")
            except Exception as e:
                logger.warning(f"⚠️ Error retrieving RAG documents: {e}")
        
        return context
    
    async def _get_scenario_comparison_context(self) -> Dict[str, Any]:
        """
        Fetch scenario comparison data from the comparison API
        """
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                url = "http://127.0.0.1:8002/api/tesla/comparison"
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info("✅ Fetched scenario comparison data")
                        return data
                    else:
                        logger.warning(f"Failed to fetch scenario comparison data: {response.status}")
                        return {}
        except Exception as e:
            logger.error(f"Error fetching scenario comparison data: {str(e)}")
            return {}
    
    async def _get_cross_statement_context(self, query: str) -> Dict[str, Any]:
        """
        Fetch cross-statement simulation data and run analysis using Cross-Statement Agent
        """
        try:
            import aiohttp
            
            # Try to execute an actual cross-statement simulation with sensible defaults
            # For questions like "How do changes flow through statements?", run a sample simulation
            logger.info("🔗 Executing cross-statement simulation for analysis")
            
            # Default parameters for demonstration
            sim_params = {
                "automotive_revenue_growth": 0.20,  # 20% growth example
                "gross_margin_automotive": 0.21,    # 21% margin
                "include_ai_insights": True
            }
            
            url = "http://127.0.0.1:8002/api/tesla/model/base/simulate-all-statements"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=sim_params, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.info("✅ Cross-statement simulation executed successfully")
                        
                        # Extract AI insights from the simulation
                        ai_insights = data.get('ai_insights', {})
                        
                        return {
                            "available": True,
                            "executed": True,
                            "simulation_data": data,
                            "ai_insights": ai_insights,
                            "executive_summary": ai_insights.get('executive_summary', ''),
                            "key_insights": ai_insights.get('key_insights', []),
                            "strategic_recommendations": ai_insights.get('strategic_recommendations', [])
                        }
                    else:
                        logger.warning(f"Simulation failed: {response.status}")
                        # Return fallback info
                        return {
                            "available": True,
                            "executed": False,
                            "endpoint": "/api/tesla/model/{scenario}/simulate-all-statements",
                            "note": "Cross-statement simulation endpoint is available for integrated analysis."
                        }
            
        except Exception as e:
            logger.error(f"Error executing cross-statement simulation: {str(e)}", exc_info=True)
            return {
                "available": True,
                "executed": False,
                "error": str(e)
            }
    
    def _format_workflow_result(self, workflow_result: Dict, query: str, classification) -> Dict[str, Any]:
        """Format workflow execution result to match standard response structure"""
        final_result = workflow_result.get("final_result", {})
        
        return {
            "executive_summary": final_result.get("executive_summary", f"Completed workflow analysis for: {query}"),
            "key_insights": final_result.get("key_insights", []),
            "recommendations": final_result.get("recommendations", []),
            "analysis_sections": [{
                "title": "Workflow Execution Summary",
                "content": f"Executed {workflow_result['workflow_plan']['tasks_completed']} out of {workflow_result['workflow_plan']['total_tasks']} tasks",
                "data_type": "workflow"
            }],
            "workflow_metadata": {
                "decomposition_used": True,
                "complexity_score": workflow_result['workflow_plan']['complexity_score'],
                "total_tasks": workflow_result['workflow_plan']['total_tasks'],
                "tasks_completed": workflow_result['workflow_plan']['tasks_completed'],
                "actual_duration": workflow_result['workflow_plan']['actual_duration'],
                "execution_log": workflow_result.get("execution_log", [])
            },
            "agents_used": ["TaskDecomposer", "TaskExecutor"],
            "tasks_executed": ["workflow_decomposition", "parallel_task_execution", "result_synthesis"],
            "query_type": "complex_workflow" if classification else "workflow",
            "timestamp": datetime.now().isoformat()
        }
    
    def _build_financial_prompt(self, query: str, financial_context: Dict[str, Any], classification, is_document_summary_query: bool = False) -> str:
        """Build a comprehensive prompt for OpenAI with financial context"""
        
        prompt_parts = [
            f"User Question: {query}",
            ""
        ]
        
        # For document summarization queries, prioritize RAG documents and skip financial model data
        if is_document_summary_query and 'rag_documents' in financial_context:
            # Put RAG documents FIRST for document summarization queries
            rag_docs = financial_context['rag_documents']
            if rag_docs:
                prompt_parts.extend([
                    "=== PRIMARY DATA SOURCE: RELEVANT DOCUMENTS FROM TESLA KNOWLEDGE BASE ===",
                    "CRITICAL: You MUST base your response ONLY on the document excerpts below.",
                    "DO NOT use any financial model forecasts or projections - only use actual data from the documents.",
                    "",
                    "The following document excerpts contain the information you need:",
                    ""
                ])
                
                # For document summarization, show more chunks; otherwise limit to 5
                chunk_limit = 15 if is_document_summary_query else 5
                text_limit = 800 if is_document_summary_query else 500
                
                for i, doc in enumerate(rag_docs[:chunk_limit], 1):
                    metadata = doc.get('metadata', {})
                    source = metadata.get('source', 'Unknown')
                    doc_type = metadata.get('document_type', 'Document')
                    year = metadata.get('year', 'Unknown')
                    chunk_text = doc.get('text', '')[:text_limit]  # Longer excerpts for summarization
                    
                    prompt_parts.extend([
                        f"[Document {i}]",
                        f"Source: {source}",
                        f"Type: {doc_type} ({year})",
                        f"Relevance Score: {1 - doc.get('distance', 1):.3f}" if 'distance' in doc else "",
                        f"",
                        f"Content Excerpt:",
                        chunk_text,
                        "...",
                        ""
                    ])
                
                # Special instructions for document summarization
                prompt_parts.extend([
                    "",
                    "=== DOCUMENT SUMMARIZATION TASK ===",
                    "The user has requested a summary of a SPECIFIC DOCUMENT (not financial model forecasts).",
                    "",
                    "CRITICAL INSTRUCTIONS:",
                    "1. Base your response ONLY on the document excerpts provided above",
                    "2. DO NOT use any financial model forecasts (2025, 2026 projections shown below are NOT relevant)",
                    "3. Extract actual data, facts, and insights from the document excerpts",
                    "4. Cite the source document: '[Document Type] ([Source filename])'",
                    "5. If the document mentions specific years (e.g., 2024, 2023), use those years - NOT forecast years",
                    "6. Provide a comprehensive executive summary of the document content",
                    "7. Identify key insights, themes, and important information from the document",
                    "8. Extract key metrics, goals, achievements, or data points mentioned in the document",
                    "9. Organize insights by topic/section if the document has multiple sections",
                    "10. Provide actionable insights or recommendations based on the document content",
                    "",
                    "IMPORTANT: The financial model data below (if any) is for reference only.",
                    "For document summarization queries, prioritize document content over model forecasts.",
                    "",
                ])
        else:
            prompt_parts.append("Available Financial Data:")
        
        # Add vehicle forecast data if available (skip for document summaries)
        if not is_document_summary_query and 'vehicle_forecasts_cached' in financial_context:
            forecasts_data = financial_context['vehicle_forecasts_cached']
            forecast_list = forecasts_data.get('forecasts', [])
            agent_status = financial_context.get('forecast_agent_status', {})
            
            # Ensure forecast_list is actually a list
            if forecast_list and isinstance(forecast_list, list):
                prompt_parts.extend([
                    "",
                    "=== Vehicle Forecasts (12-Month Autonomous Agent Predictions) ===",
                    f"Agent Status: {agent_status.get('status', 'N/A')}",
                    f"Last Updated: {agent_status.get('last_update', 'N/A')}",
                    f"Total Models Forecasted: {len(forecast_list)}",
                    ""
                ])
                
                # Add summary of each model's forecast - safely limit to first 5
                safe_limit = min(5, len(forecast_list))
                for forecast in forecast_list[:safe_limit]:
                    model_name = forecast.get('model_name', 'Unknown')
                    available_targets = forecast.get('metadata', {}).get('available_targets', [])
                    
                    prompt_parts.append(f"Model: {model_name}")
                    
                    # Univariate forecasts
                    uni_forecasts = forecast.get('univariate', {}).get('forecasts_by_target', {})
                    multi_forecasts = forecast.get('multivariate', {}).get('forecasts_by_target', {})
                    
                    for target in available_targets:
                        if target in uni_forecasts:
                            uni_data = uni_forecasts[target]
                            multi_data = multi_forecasts.get(target, {})
                            
                            uni_total = sum(f['forecast'] for f in uni_data.get('forecasts', []))
                            multi_total = sum(f['forecast'] for f in multi_data.get('forecasts', []))
                            
                            uni_mae = uni_data.get('model_metrics', {}).get('mae', 0)
                            uni_mape = uni_data.get('model_metrics', {}).get('mape', 0)
                            multi_mae = multi_data.get('model_metrics', {}).get('mae', 0)
                            multi_mape = multi_data.get('model_metrics', {}).get('mape', 0)
                            
                            prompt_parts.extend([
                                f"  - {target.capitalize()} (12-month total):",
                                f"    • Univariate Forecast: {int(uni_total):,} units (MAE: {uni_mae:.0f}, MAPE: {uni_mape*100:.1f}%)",
                                f"    • Multivariate Forecast: {int(multi_total):,} units (MAE: {multi_mae:.0f}, MAPE: {multi_mape*100:.1f}%)"
                            ])
                    
                    prompt_parts.append("")
                
                if len(forecast_list) > 5:
                    prompt_parts.append(f"... and {len(forecast_list) - 5} more models")
                    prompt_parts.append("")
        
        # Add vehicle summary data if available
        if 'vehicle_summary_2025' in financial_context:
            summary = financial_context['vehicle_summary_2025']
            totals = summary.get('totals', {})
            prompt_parts.extend([
                "",
                "=== 2025 Vehicle Performance ===",
                f"- Total Deliveries: {totals.get('total_deliveries', 'N/A'):,} units",
                f"- Total Production: {totals.get('total_production', 'N/A'):,} units",
                f"- Total Revenue: ${totals.get('total_revenue', 0)/1e9:.2f}B",
                f"- Total Costs: ${totals.get('total_costs', 0)/1e9:.2f}B",
                f"- Total Profit: ${totals.get('total_profit', 0)/1e9:.2f}B",
                f"- Profit Margin: {totals.get('profit_margin', 0):.1f}%"
            ])
        
        # Add financial statements - ONLY base case to reduce token usage
        # SKIP for document summarization queries (they should use document content only)
        if not is_document_summary_query:
            scenarios = ['base']  # Only base case for concise responses
            for scenario in scenarios:
                key = f'financial_statements_{scenario}'
                if key in financial_context:
                    statements = financial_context[key]
                
                # Income Statement
                income_statements = statements.get('income_statements', [])
                if income_statements:
                    income_2025 = next((is_data for is_data in income_statements if is_data.get('year') == 2025), None)
                    income_2026 = next((is_data for is_data in income_statements if is_data.get('year') == 2026), None)
                    
                    if income_2025:
                        # Calculate COGS and Gross Profit from margins if not provided
                        revenue_2025 = income_2025.get('total_revenue', 0)
                        gross_margin_2025 = income_2025.get('gross_margin', 0)
                        gross_profit_2025 = revenue_2025 * gross_margin_2025
                        cogs_2025 = revenue_2025 - gross_profit_2025
                        
                        # Calculate R&D and SG&A from operating income
                        operating_income_2025 = income_2025.get('operating_income', 0)
                        rd_sga_2025 = gross_profit_2025 - operating_income_2025
                        
                        prompt_parts.extend([
                            "",
                            f"=== Income Statement - {scenario.upper()} Case (2025) ===",
                            f"- Total Revenue: ${revenue_2025/1e9:.2f}B (100.0%)",
                            f"- COGS: ${cogs_2025/1e9:.2f}B ({(cogs_2025/revenue_2025*100):.1f}%)",
                            f"- Gross Profit: ${gross_profit_2025/1e9:.2f}B ({gross_margin_2025*100:.1f}%)",
                            f"- R&D + SG&A Expenses: ${rd_sga_2025/1e9:.2f}B ({(rd_sga_2025/revenue_2025*100):.1f}%)",
                            f"- Operating Income: ${operating_income_2025/1e9:.2f}B ({income_2025.get('operating_margin', 0)*100:.1f}%)",
                            f"- Net Income: ${income_2025.get('net_income', 0)/1e9:.2f}B ({income_2025.get('net_margin', 0)*100:.1f}%)",
                            f"- EPS: ${income_2025.get('earnings_per_share', 0):.2f}"
                        ])
                        
                        if income_2026:
                            # Calculate COGS and Gross Profit for 2026
                            revenue_2026 = income_2026.get('total_revenue', 0)
                            gross_margin_2026 = income_2026.get('gross_margin', 0)
                            gross_profit_2026 = revenue_2026 * gross_margin_2026
                            cogs_2026 = revenue_2026 - gross_profit_2026
                            operating_income_2026 = income_2026.get('operating_income', 0)
                            rd_sga_2026 = gross_profit_2026 - operating_income_2026
                            
                            prompt_parts.extend([
                                "",
                                f"=== Income Statement - {scenario.upper()} Case (2026 Forecast) ===",
                                f"- Total Revenue: ${revenue_2026/1e9:.2f}B (100.0%)",
                                f"- COGS: ${cogs_2026/1e9:.2f}B ({(cogs_2026/revenue_2026*100):.1f}%)",
                                f"- Gross Profit: ${gross_profit_2026/1e9:.2f}B ({gross_margin_2026*100:.1f}%)",
                                f"- R&D + SG&A Expenses: ${rd_sga_2026/1e9:.2f}B ({(rd_sga_2026/revenue_2026*100):.1f}%)",
                                f"- Operating Income: ${operating_income_2026/1e9:.2f}B ({income_2026.get('operating_margin', 0)*100:.1f}%)",
                                f"- Net Income: ${income_2026.get('net_income', 0)/1e9:.2f}B ({income_2026.get('net_margin', 0)*100:.1f}%)"
                            ])
                
                # Balance Sheet
                balance_sheets = statements.get('balance_sheets', [])
                if balance_sheets:
                    balance_2025 = next((bs for bs in balance_sheets if bs.get('year') == 2025), None)
                    balance_2026 = next((bs for bs in balance_sheets if bs.get('year') == 2026), None)
                    
                    if balance_2025:
                        prompt_parts.extend([
                            "",
                            f"=== Balance Sheet - {scenario.upper()} Case (2025) ===",
                            f"- Total Assets: ${balance_2025.get('total_assets', 0)/1e9:.2f}B",
                            f"- Cash & Equivalents: ${balance_2025.get('cash_and_equivalents', 0)/1e9:.2f}B",
                            f"- Accounts Receivable: ${balance_2025.get('accounts_receivable', 0)/1e9:.2f}B",
                            f"- Inventory: ${balance_2025.get('inventory', 0)/1e9:.2f}B",
                            f"- Total Current Assets: ${balance_2025.get('total_current_assets', 0)/1e9:.2f}B",
                            f"- Net PP&E: ${balance_2025.get('net_ppe', 0)/1e9:.2f}B",
                            f"- Total Non-Current Assets: ${balance_2025.get('total_non_current_assets', 0)/1e9:.2f}B",
                            f"- Accounts Payable: ${balance_2025.get('accounts_payable', 0)/1e9:.2f}B",
                            f"- Total Current Liabilities: ${balance_2025.get('total_current_liabilities', 0)/1e9:.2f}B",
                            f"- Long-Term Debt: ${balance_2025.get('long_term_debt', 0)/1e9:.2f}B",
                            f"- Total Non-Current Liabilities: ${balance_2025.get('total_non_current_liabilities', 0)/1e9:.2f}B",
                            f"- Total Liabilities: ${balance_2025.get('total_liabilities', 0)/1e9:.2f}B",
                            f"- Retained Earnings: ${balance_2025.get('retained_earnings', 0)/1e9:.2f}B",
                            f"- Total Equity: ${balance_2025.get('total_shareholders_equity', 0)/1e9:.2f}B",
                            f"- Debt to Equity: {balance_2025.get('debt_to_equity', 0):.2f}"
                        ])
                    
                    if balance_2026:
                        prompt_parts.extend([
                            "",
                            f"=== Balance Sheet - {scenario.upper()} Case (2026 Forecast) ===",
                            f"- Total Assets: ${balance_2026.get('total_assets', 0)/1e9:.2f}B",
                            f"- Cash & Equivalents: ${balance_2026.get('cash_and_equivalents', 0)/1e9:.2f}B",
                            f"- Accounts Receivable: ${balance_2026.get('accounts_receivable', 0)/1e9:.2f}B",
                            f"- Inventory: ${balance_2026.get('inventory', 0)/1e9:.2f}B",
                            f"- Total Current Assets: ${balance_2026.get('total_current_assets', 0)/1e9:.2f}B",
                            f"- Net PP&E: ${balance_2026.get('net_ppe', 0)/1e9:.2f}B",
                            f"- Total Non-Current Assets: ${balance_2026.get('total_non_current_assets', 0)/1e9:.2f}B",
                            f"- Accounts Payable: ${balance_2026.get('accounts_payable', 0)/1e9:.2f}B",
                            f"- Total Current Liabilities: ${balance_2026.get('total_current_liabilities', 0)/1e9:.2f}B",
                            f"- Long-Term Debt: ${balance_2026.get('long_term_debt', 0)/1e9:.2f}B",
                            f"- Total Non-Current Liabilities: ${balance_2026.get('total_non_current_liabilities', 0)/1e9:.2f}B",
                            f"- Total Liabilities: ${balance_2026.get('total_liabilities', 0)/1e9:.2f}B",
                            f"- Retained Earnings: ${balance_2026.get('retained_earnings', 0)/1e9:.2f}B",
                            f"- Total Equity: ${balance_2026.get('total_shareholders_equity', 0)/1e9:.2f}B",
                            f"- Debt to Equity: {balance_2026.get('debt_to_equity', 0):.2f}"
                        ])
                
                # Cash Flow Statement
                cash_flows = statements.get('cash_flow_statements', [])
                if cash_flows:
                    cf_2025 = next((cf for cf in cash_flows if cf.get('year') == 2025), None)
                    cf_2026 = next((cf for cf in cash_flows if cf.get('year') == 2026), None)
                    
                    if cf_2025:
                        prompt_parts.extend([
                            "",
                            f"=== Cash Flow Statement - {scenario.upper()} Case (2025) ===",
                            f"- Operating Cash Flow: ${cf_2025.get('operating_cash_flow', 0)/1e9:.2f}B",
                            f"- Investing Cash Flow: ${cf_2025.get('investing_cash_flow', 0)/1e9:.2f}B",
                            f"- Financing Cash Flow: ${cf_2025.get('financing_cash_flow', 0)/1e9:.2f}B",
                            f"- Free Cash Flow: ${cf_2025.get('free_cash_flow', 0)/1e9:.2f}B",
                            f"- Net Change in Cash: ${cf_2025.get('net_change_cash', 0)/1e9:.2f}B"
                        ])
                    
                    if cf_2026:
                        prompt_parts.extend([
                            "",
                            f"=== Cash Flow Statement - {scenario.upper()} Case (2026 Forecast) ===",
                            f"- Operating Cash Flow: ${cf_2026.get('operating_cash_flow', 0)/1e9:.2f}B",
                            f"- Investing Cash Flow: ${cf_2026.get('investing_cash_flow', 0)/1e9:.2f}B",
                            f"- Financing Cash Flow: ${cf_2026.get('financing_cash_flow', 0)/1e9:.2f}B",
                            f"- Free Cash Flow: ${cf_2026.get('free_cash_flow', 0)/1e9:.2f}B",
                            f"- Net Change in Cash: ${cf_2026.get('net_change_cash', 0)/1e9:.2f}B"
                        ])
        
        # Add scenario comparison data if available
        if 'scenario_comparison' in financial_context:
            comparison_data = financial_context['scenario_comparison']
            if comparison_data.get('success'):
                comparison_summary = comparison_data.get('comparison_summary', {})
                prompt_parts.extend([
                    "",
                    "=== SCENARIO COMPARISON (Best vs Base vs Worst) ===",
                    ""
                ])
                
                # Revenue comparison
                if 'revenue_comparison' in comparison_summary:
                    prompt_parts.append("Revenue Projections (2029):")
                    for scenario in ['best', 'base', 'worst']:
                        if scenario in comparison_summary['revenue_comparison']:
                            rev_data = comparison_summary['revenue_comparison'][scenario]
                            revenue_2029 = rev_data.get('2029_revenue', 0) / 1e9
                            cagr = rev_data.get('5yr_cagr', 0) * 100
                            prompt_parts.append(f"  - {scenario.capitalize()}: ${revenue_2029:.1f}B (5-yr CAGR: {cagr:.1f}%)")
                    prompt_parts.append("")
                
                # Valuation comparison
                if 'valuation_comparison' in comparison_summary:
                    prompt_parts.append("DCF Valuation:")
                    for scenario in ['best', 'base', 'worst']:
                        if scenario in comparison_summary['valuation_comparison']:
                            val_data = comparison_summary['valuation_comparison'][scenario]
                            price = val_data.get('price_per_share', 0)
                            ev = val_data.get('enterprise_value', 0) / 1e9
                            prompt_parts.append(f"  - {scenario.capitalize()}: ${price:.2f}/share (EV: ${ev:.1f}B)")
                    prompt_parts.append("")
                
                # Margin comparison
                if 'margin_comparison' in comparison_summary:
                    prompt_parts.append("Profitability Margins (2029):")
                    for scenario in ['best', 'base', 'worst']:
                        if scenario in comparison_summary['margin_comparison']:
                            margin_data = comparison_summary['margin_comparison'][scenario]
                            gross = margin_data.get('2029_gross_margin', 0) * 100
                            operating = margin_data.get('2029_operating_margin', 0) * 100
                            net = margin_data.get('2029_net_margin', 0) * 100
                            prompt_parts.append(f"  - {scenario.capitalize()}: Gross {gross:.1f}% | Operating {operating:.1f}% | Net {net:.1f}%")
                    prompt_parts.append("")
        
        # Add cross-statement simulation info if available
        if 'cross_statement_data' in financial_context:
            cross_data = financial_context['cross_statement_data']
            if cross_data.get('available'):
                prompt_parts.extend([
                    "",
                    "=== CROSS-STATEMENT SIMULATION CAPABILITY ===",
                    "Tesla FA can perform integrated financial statement simulations that show the",
                    "cascading impact across Income Statement, Balance Sheet, and Cash Flow Statement.",
                    "",
                    "Available simulation parameters:",
                    "  • Revenue growth (automotive & services)",
                    "  • Gross margins (automotive & services)",
                    "  • Operating expenses (R&D, SG&A as % of revenue)",
                    "  • Working capital (DSO, DIO, DPO)",
                    "  • Capital expenditures (CapEx as % of revenue)",
                    "  • Tax rate",
                    "",
                    f"Endpoint: {cross_data.get('endpoint', 'N/A')}",
                    "Note: This provides AI-powered insights on the integrated impact across all three statements.",
                    ""
                ])
        
        # Add RAG documents if available (for non-summarization queries only - summarization queries already handled above)
        if not is_document_summary_query and 'rag_documents' in financial_context:
            rag_docs = financial_context['rag_documents']
            if rag_docs:
                prompt_parts.extend([
                    "",
                    "=== RELEVANT DOCUMENTS FROM TESLA KNOWLEDGE BASE ===",
                    "The following documents from Tesla's SEC filings, annual reports, and impact reports",
                    "contain relevant information related to the user's question:",
                    ""
                ])
                
                # Limit to 5 chunks for regular queries
                for i, doc in enumerate(rag_docs[:5], 1):
                    metadata = doc.get('metadata', {})
                    source = metadata.get('source', 'Unknown')
                    doc_type = metadata.get('document_type', 'Document')
                    year = metadata.get('year', 'Unknown')
                    chunk_text = doc.get('text', '')[:500]  # Shorter excerpts for regular queries
                    
                    prompt_parts.extend([
                        f"[Document {i}]",
                        f"Source: {source}",
                        f"Type: {doc_type} ({year})",
                        f"Relevance Score: {1 - doc.get('distance', 1):.3f}" if 'distance' in doc else "",
                        f"",
                        f"Content Excerpt:",
                        chunk_text,
                        "...",
                        ""
                    ])
        
        # Special instructions for document summarization queries (if not already added above)
        if is_document_summary_query and 'rag_documents' in financial_context and not any('DOCUMENT SUMMARIZATION TASK' in p for p in prompt_parts):
            prompt_parts.extend([
                "",
                "=== DOCUMENT SUMMARIZATION TASK ===",
                "The user has requested a summary and insights for a specific document from Tesla's Knowledge Base.",
                "Your task is to:",
                "1. Provide a comprehensive executive summary of the document",
                "2. Identify and highlight key insights, themes, and important information",
                "3. Extract key metrics, goals, achievements, or data points mentioned",
                "4. Organize insights by topic/section if the document has multiple sections",
                "5. Cite specific information from the document using the source filename",
                "6. Provide actionable insights or recommendations based on the document content",
                "",
            ])
        
        # Phase 3: Add temporal context if available
        if 'temporal_info' in financial_context:
            temporal_info = financial_context['temporal_info']
            temporal_context = self.rag_service.temporal_reasoning.build_temporal_prompt_context(
                temporal_info,
                financial_context.get('rag_documents', [])
            )
            if temporal_context:
                prompt_parts.append(temporal_context)
        
        prompt_parts.extend([
            "",
            "=== Your Task ===",
            "Based on the financial data provided above" + (" and relevant documents" if 'rag_documents' in financial_context else "") + ":",
            "1. Answer the user's specific question with EXACT NUMBERS from the data above",
            "2. For VEHICLE FORECAST questions:",
            "   - Specify which model(s) the forecast is for",
            "   - Distinguish between univariate and multivariate forecasts",
            "   - Mention deliveries, production, and sold quantities as available",
            "   - Include forecast accuracy metrics (MAE, MAPE)",
            "   - Explain what the forecast means in business context",
            "3. For SCENARIO COMPARISON questions:",
            "   - Show side-by-side comparison of Best, Base, and Worst cases",
            "   - Highlight key differences in revenue, valuation, and margins",
            "   - Explain the assumptions driving each scenario",
            "   - Provide risk-adjusted recommendations",
            "4. For CROSS-STATEMENT SIMULATION questions:",
            "   - Explain how changes cascade through all three financial statements",
            "   - Show the integrated impact on profitability, cash flow, and balance sheet",
            "   - Highlight key relationships (e.g., how working capital affects cash)",
            "   - Mention that users can access detailed simulations via the Financial Statements page",
            "5. When comparing years (e.g., 2025 vs 2026), show BOTH years' data with specific values",
            "6. Calculate year-over-year changes: show absolute difference ($ amount/units) AND percentage change",
            "7. Explain key financial relationships and metrics",
            "8. Provide insights on trends and performance drivers",
            "9. Include actionable recommendations based on the analysis",
            "10. Cite specific data sources (e.g., 'According to Model 3 Multivariate Forecast' or 'Best Case Balance Sheet 2026')",
            "11. " + ("MANDATORY: If 'RELEVANT DOCUMENTS FROM TESLA KNOWLEDGE BASE' section appears above, you MUST cite the source document(s) when using their information. Start your response with: 'According to the [YEAR] [DOCUMENT TYPE] Report ([Source filename])' or include inline citations like '([YEAR] 10-K Report).' Example: 'According to the 2024 10-K Report (10K Report 2024.pdf), Tesla's total revenue was...' If you use ANY information from those documents, citations are REQUIRED." if 'rag_documents' in financial_context else "When using information from Tesla Knowledge Base documents (SEC filings, reports), cite the source document (e.g., 'According to the 2024 10-K Report...')"),
            "",
            "Format your response clearly with:",
            "- Direct answer to the question with specific numbers",
            "- For scenario comparisons: side-by-side table or bullet points",
            "- For cross-statement questions: explain the flow through all three statements",
            "- For forecast questions: show 12-month totals and explain methodology",
            "- Year-over-year comparison table if comparing multiple years",
            "- Key financial metrics with exact values from the data",
            "- Comparative analysis showing differences and % changes",
            "- Insights and implications",
            "- Recommendations",
            "",
            "IMPORTANT:",
            "- ALWAYS use the exact numbers provided in the data above",
            "- NEVER say 'data not provided' if the data is clearly shown above",
            "- For vehicle forecasts, explain the difference between univariate (time-series only) and multivariate (includes economic variables)",
            "- For balance sheet comparisons, show line-by-line changes between years",
            "- For scenario comparisons, use the exact valuation and margin data provided",
            "- For cross-statement simulations, explain the integrated nature of the three statements",
            "- Calculate meaningful ratios and metrics from the raw data",
            "",
            "Provide your comprehensive response:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _add_simulation_calculations(self, prompt: str, query: str, financial_context: Dict) -> str:
        """Pre-calculate simulation results and add to prompt"""
        try:
            logger.info(f"🔍 Attempting to add simulation calculations for query: {query}")
            # Extract growth percentage from query
            import re
            growth_match = re.search(r'(\d+)%?\s*(revenue\s*)?growth', query.lower())
            if not growth_match:
                logger.warning("Could not extract growth percentage from query")
                return prompt  # Can't parse growth rate
            
            growth_pct = float(growth_match.group(1)) / 100
            logger.info(f"📊 Extracted growth rate: {growth_pct*100:.0f}%")
            
            # Get 2025 base case data
            base_statements = financial_context.get('financial_statements_base', {})
            income_2025 = next((is_data for is_data in base_statements.get('income_statements', []) 
                              if is_data.get('year') == 2025), None)
            
            if not income_2025:
                return prompt
            
            # Calculate base values
            revenue_base = income_2025.get('total_revenue', 0) / 1e9
            gross_margin = income_2025.get('gross_margin', 0.195)
            cogs_base = revenue_base * (1 - gross_margin)
            gp_base = revenue_base * gross_margin
            op_income_base = income_2025.get('operating_income', 0) / 1e9
            net_income_base = income_2025.get('net_income', 0) / 1e9
            rd_sga_base = gp_base - op_income_base
            
            # Calculate simulated values
            revenue_sim = revenue_base * (1 + growth_pct)
            cogs_sim = revenue_sim * (1 - gross_margin)
            gp_sim = revenue_sim * gross_margin
            rd_sga_sim = rd_sga_base * 1.1  # Assume 10% increase in fixed costs
            op_income_sim = gp_sim - rd_sga_sim
            net_income_sim = op_income_sim * (net_income_base / op_income_base if op_income_base > 0 else 0.75)
            
            # Add pre-calculated table to prompt
            simulation_table = f"""

========================================
PRE-CALCULATED SIMULATION RESULTS
========================================

The user asked to simulate {growth_pct*100:.0f}% revenue growth. Here are the EXACT calculated results:

| Metric | Base 2025 | Simulated | Change ($B) | Change % |
|--------|-----------|-----------|-------------|----------|
| Revenue | ${revenue_base:.2f}B | ${revenue_sim:.2f}B | ${revenue_sim-revenue_base:+.2f}B | {growth_pct*100:+.1f}% |
| COGS | ${cogs_base:.2f}B | ${cogs_sim:.2f}B | ${cogs_sim-cogs_base:+.2f}B | {((cogs_sim/cogs_base-1)*100):+.1f}% |
| Gross Profit | ${gp_base:.2f}B | ${gp_sim:.2f}B | ${gp_sim-gp_base:+.2f}B | {((gp_sim/gp_base-1)*100):+.1f}% |
| R&D + SG&A | ${rd_sga_base:.2f}B | ${rd_sga_sim:.2f}B | ${rd_sga_sim-rd_sga_base:+.2f}B | {((rd_sga_sim/rd_sga_base-1)*100):+.1f}% |
| Operating Income | ${op_income_base:.2f}B | ${op_income_sim:.2f}B | ${op_income_sim-op_income_base:+.2f}B | {((op_income_sim/op_income_base-1)*100):+.1f}% |
| Net Income | ${net_income_base:.2f}B | ${net_income_sim:.2f}B | ${net_income_sim-net_income_base:+.2f}B | {((net_income_sim/net_income_base-1)*100):+.1f}% |

⚠️ CRITICAL: Use THESE EXACT VALUES in your response table. Do not recalculate.

"""
            logger.info(f"✅ Successfully added pre-calculated simulation table with COGS=${cogs_base:.2f}B")
            return prompt + simulation_table
            
        except Exception as e:
            logger.warning(f"Could not add simulation calculations: {e}")
            return prompt
    
    def _extract_insights_from_response(self, response: str) -> List[str]:
        """Extract key insights from OpenAI response"""
        insights = []
        
        # Look for bullet points or numbered lists
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            # Check if line starts with bullet point, number, or dash
            if line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
                insight = line.lstrip('•-*123456789. ')
                if len(insight) > 10 and len(insight) < 200:  # Reasonable insight length
                    insights.append(insight)
        
        # If no insights found, extract first few sentences
        if not insights:
            sentences = response.split('.')
            insights = [s.strip() + '.' for s in sentences[:3] if len(s.strip()) > 10]
        
        return insights[:5]  # Return top 5 insights
    
    def _extract_recommendations_from_response(self, response: str) -> List[str]:
        """Extract recommendations from OpenAI response"""
        recommendations = []
        
        # Look for recommendations section
        lower_response = response.lower()
        if 'recommend' in lower_response:
            # Find the recommendations section
            rec_start = lower_response.find('recommend')
            rec_section = response[rec_start:]
            
            lines = rec_section.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')):
                    rec = line.lstrip('•-*123456789. ')
                    if len(rec) > 10 and len(rec) < 200:
                        recommendations.append(rec)
        
        # If no recommendations found, provide general ones
        if not recommendations:
            recommendations = [
                "Monitor financial performance trends closely",
                "Review key financial metrics regularly",
                "Stay informed about market conditions"
            ]
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _build_user_document_prompt(self, query: str) -> str:
        """
        Build a prompt for user-uploaded document queries
        This prompt ONLY uses the document content provided in the query itself
        
        Args:
            query: User query (already contains document excerpts from FRA endpoint)
            
        Returns:
            Prompt string that instructs AI to ONLY use the provided document content
        """
        prompt_parts = [
            "=== USER-UPLOADED DOCUMENT ANALYSIS ===",
            "",
            "CRITICAL INSTRUCTIONS:",
            "1. You are analyzing a document that the user has uploaded (PDF, Excel, CSV, or Image)",
            "2. You MUST base your response ONLY on the document excerpts provided below",
            "3. DO NOT use any Tesla Knowledge Base documents, financial model forecasts, or external data",
            "4. DO NOT reference Tesla's 10-K reports, Impact Reports, or any pre-existing Tesla documents",
            "5. DO NOT use any financial model data (2025, 2026 forecasts, vehicle forecasts, etc.)",
            "6. If the document doesn't contain the information needed to answer the question, say so explicitly",
            "7. Answer based ONLY on what is in the uploaded document excerpts",
            "",
            "=== USER QUESTION ===",
            query,
            "",
            "=== YOUR TASK ===",
            "Provide a clear, accurate answer based ONLY on the document content provided above.",
            "If information is missing, state that it's not available in the uploaded document.",
            "Do not make assumptions or use knowledge from other sources.",
            "",
        ]
        
        return "\n".join(prompt_parts)
