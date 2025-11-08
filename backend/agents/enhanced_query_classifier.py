"""
Enhanced Query Classifier for Market Intelligence Integration
Classifies user queries to route to appropriate agents (financial modeling vs market intelligence)
"""

import re
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueryType(Enum):
    FINANCIAL_MODELING = "financial_modeling"
    MARKET_INTELLIGENCE = "market_intelligence"
    HYBRID = "hybrid"
    GENERAL = "general"

@dataclass
class QueryClassification:
    query_type: QueryType
    confidence: float
    detected_categories: List[str]
    market_score: int
    financial_score: int
    keywords: List[str]

class EnhancedQueryClassifier:
    """
    Enhanced query classifier that can distinguish between financial modeling
    and market intelligence queries, enabling smart routing to appropriate agents
    """
    
    def __init__(self):
        self.market_intelligence_patterns = {
            "stock_performance": [
                "stock price", "share price", "stock performance", "tsla stock",
                "market cap", "trading volume", "stock analysis", "price movement",
                "technical analysis", "chart", "rsi", "macd", "moving average",
                "bollinger bands", "support", "resistance", "breakout", "downtrend",
                "uptrend", "volatility", "beta", "alpha", "sharpe ratio"
            ],
            "market_sentiment": [
                "market sentiment", "analyst sentiment", "social sentiment",
                "news sentiment", "investor sentiment", "bullish", "bearish",
                "analyst rating", "price target", "upgrade", "downgrade",
                "hold", "buy", "sell", "outperform", "underperform",
                "recommendation", "coverage", "initiate coverage",
                "news", "recent news", "latest news", "tesla news", "headlines",
                "articles", "media coverage", "press", "announcement"
            ],
            "competitor_analysis": [
                "competitors", "rivian", "lucid", "nio", "byd", "ford", "gm",
                "volkswagen", "bmw", "mercedes", "toyota", "honda", "nissan",
                "market share", "competitive advantage", "vs", "comparison",
                "industry leader", "market position", "competitive landscape",
                "market dominance", "pricing strategy", "product portfolio"
            ],
            "risk_assessment": [
                "risk", "risks", "risk factors", "market risk", "operational risk",
                "regulatory risk", "volatility", "uncertainty", "threats",
                "challenges", "headwinds", "downside risk", "upside risk",
                "risk management", "risk mitigation", "exposure", "vulnerability",
                "risk alert", "risk monitoring", "risk dashboard", "danger",
                "threat level", "monitoring", "alert", "alerts", "warning signs"
            ],
            "investment_decision": [
                "should i invest", "investment decision", "invest in tesla",
                "buy tesla", "investing in", "investment opportunity",
                "portfolio", "allocation", "position", "market conditions",
                "investment thesis", "investment analysis", "investment recommendation"
            ],
            "industry_trends": [
                "ev market", "electric vehicle", "automotive industry",
                "battery technology", "charging infrastructure", "autonomous driving",
                "industry trends", "market trends", "adoption rate", "penetration",
                "market growth", "industry outlook", "technology trends",
                "innovation", "disruption", "transformation"
            ],
            "options_flow": [
                "options", "options flow", "unusual options activity", "put",
                "call", "options volume", "options sentiment", "implied volatility",
                "options chain", "strike price", "expiration", "moneyness"
            ]
        }
        
        self.financial_modeling_patterns = {
            "vehicle_forecast": [
                "vehicle forecast", "deliveries forecast", "production forecast",
                "sold forecast", "model forecast", "delivery prediction", "production prediction",
                "forecast deliveries", "forecast production", "forecast sold",
                "how many deliveries", "how many units", "delivery outlook",
                "production outlook", "sales forecast", "unit forecast",
                "model 3 forecast", "model y forecast", "model s forecast", "model x forecast",
                "cybertruck forecast", "semi forecast", "vehicle prediction",
                "delivery estimates", "production estimates", "sales projection",
                "univariate forecast", "multivariate forecast", "forecast accuracy"
            ],
            "income_statement": [
                "revenue", "income statement", "earnings", "profit", "margin",
                "cost of goods sold", "cogs", "operating expenses", "net income",
                "gross profit", "operating profit", "ebitda", "ebit", "tax",
                "interest expense", "depreciation", "amortization", "sg&a", "r&d"
            ],
            "balance_sheet": [
                "balance sheet", "assets", "liabilities", "equity", "cash",
                "inventory", "debt", "working capital", "current assets",
                "current liabilities", "long term debt", "accounts receivable",
                "accounts payable", "property plant equipment", "ppe",
                "intangible assets", "goodwill", "retained earnings"
            ],
            "cash_flow": [
                "cash flow", "operating cash flow", "free cash flow", "fcf",
                "cash position", "liquidity", "cash generation", "cash burn",
                "capital expenditure", "capex", "dividends", "share buyback",
                "cash conversion", "cash cycle", "cash management"
            ],
            "forecasting": [
                "forecast", "projection", "prediction", "future", "growth",
                "scenario", "simulation", "modeling", "assumptions", "drivers",
                "sensitivity analysis", "stress test", "base case", "best case",
                "worst case", "variance analysis", "budget", "planning"
            ],
            "valuation": [
                "valuation", "dcf", "discounted cash flow", "pe ratio", "pb ratio",
                "ev/ebitda", "enterprise value", "equity value", "wacc", "terminal value",
                "present value", "npv", "irr", "payback period", "target price",
                "financial performance", "financial fundamentals", "fundamentals"
            ],
            "financial_metrics": [
                "financial metrics", "kpis", "performance metrics", "roi", "roe",
                "roa", "roic", "debt to equity", "current ratio", "quick ratio",
                "inventory turnover", "receivables turnover", "payables turnover"
            ],
            "document_analysis": [
                "summarize", "summary", "summarise", "report summary", "document summary",
                "10-k", "10k", "10-k report", "10k report", "sec filing", "sec filings",
                "annual report", "impact report", "tesla impact report", "esg report",
                "financial report", "quarterly report", "10-q", "10-q report",
                "document analysis", "report analysis", "analyze report", "summarize report",
                "key insights", "insights from", "insights in", "report insights",
                "document insights", "extract insights", "generate insights",
                "report highlights", "document highlights", "main findings",
                "report findings", "document findings", "tesla report", "tesla document"
            ]
        }
        
        # Compile patterns for faster matching
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for faster matching"""
        self.market_regex = {}
        self.financial_regex = {}
        
        # Compile market intelligence patterns
        for category, patterns in self.market_intelligence_patterns.items():
            pattern_str = '|'.join([re.escape(p) for p in patterns])
            self.market_regex[category] = re.compile(pattern_str, re.IGNORECASE)
        
        # Compile financial modeling patterns
        for category, patterns in self.financial_modeling_patterns.items():
            pattern_str = '|'.join([re.escape(p) for p in patterns])
            self.financial_regex[category] = re.compile(pattern_str, re.IGNORECASE)
    
    async def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a user query to determine if it's about financial modeling,
        market intelligence, or both (hybrid)
        """
        try:
            query_lower = query.lower()
            
            # Extract keywords and calculate scores
            market_score, market_categories, market_keywords = self._calculate_market_score(query_lower)
            financial_score, financial_categories, financial_keywords = self._calculate_financial_score(query_lower)
            
            # Combine all detected categories
            detected_categories = market_categories + financial_categories
            all_keywords = list(set(market_keywords + financial_keywords))
            
            # Determine query type based on scores
            total_score = market_score + financial_score
            confidence = 0.0
            
            if total_score == 0:
                query_type = QueryType.GENERAL
                confidence = 0.0
            elif market_score > 0 and financial_score > 0:
                # Hybrid query detected!
                query_type = QueryType.HYBRID
                confidence = min(market_score, financial_score) / max(market_score, financial_score)
                logger.info(f"🔗 HYBRID query detected: market_score={market_score}, financial_score={financial_score}")
            elif market_score > financial_score:
                query_type = QueryType.MARKET_INTELLIGENCE
                confidence = market_score / (market_score + financial_score) if total_score > 0 else 0
            else:
                query_type = QueryType.FINANCIAL_MODELING
                confidence = financial_score / (market_score + financial_score) if total_score > 0 else 0
            
            classification = QueryClassification(
                query_type=query_type,
                confidence=confidence,
                detected_categories=detected_categories,
                market_score=market_score,
                financial_score=financial_score,
                keywords=all_keywords
            )
            
            logger.info(f"Query classified as {query_type.value} with confidence {confidence:.2f}")
            logger.debug(f"Market score: {market_score}, Financial score: {financial_score}")
            logger.debug(f"Detected categories: {detected_categories}")
            
            return classification
            
        except Exception as e:
            logger.error(f"Error classifying query: {str(e)}")
            # Return general classification as fallback
            return QueryClassification(
                query_type=QueryType.GENERAL,
                confidence=0.0,
                detected_categories=[],
                market_score=0,
                financial_score=0,
                keywords=[]
            )
    
    def _calculate_market_score(self, query: str) -> Tuple[int, List[str], List[str]]:
        """Calculate market intelligence score and extract categories/keywords"""
        score = 0
        categories = []
        keywords = []
        
        for category, regex in self.market_regex.items():
            matches = regex.findall(query)
            if matches:
                score += len(matches)
                categories.append(f"market_{category}")
                keywords.extend(matches)
        
        return score, categories, list(set(keywords))
    
    def _calculate_financial_score(self, query: str) -> Tuple[int, List[str], List[str]]:
        """Calculate financial modeling score and extract categories/keywords"""
        score = 0
        categories = []
        keywords = []
        
        for category, regex in self.financial_regex.items():
            matches = regex.findall(query)
            if matches:
                score += len(matches)
                categories.append(f"financial_{category}")
                keywords.extend(matches)
        
        return score, categories, list(set(keywords))
    
    def get_suggestions_for_category(self, category: str) -> List[str]:
        """Get suggestion questions for a specific category"""
        suggestions = {
            "market_stock_performance": [
                "How is Tesla's stock performing today?",
                "What's Tesla's current stock price?",
                "Show me Tesla's technical analysis",
                "What are Tesla's key support and resistance levels?"
            ],
            "market_sentiment": [
                "What's the current market sentiment for Tesla?",
                "How do analysts rate Tesla stock?",
                "What are Tesla's latest analyst upgrades or downgrades?",
                "Show me social media sentiment for Tesla"
            ],
            "market_competitor_analysis": [
                "How does Tesla compare to Rivian?",
                "What's Tesla's market share vs competitors?",
                "Compare Tesla's performance to other EV companies",
                "Who are Tesla's main competitors in the EV space?"
            ],
            "financial_income_statement": [
                "What's Tesla's revenue for Q4 2024?",
                "Analyze Tesla's profit margins",
                "Show me Tesla's income statement",
                "What are Tesla's key revenue drivers?"
            ],
            "financial_balance_sheet": [
                "What's Tesla's cash position?",
                "Show me Tesla's balance sheet",
                "What's Tesla's debt level?",
                "Analyze Tesla's working capital"
            ],
            "hybrid": [
                "How might Tesla's stock performance affect its revenue projections?",
                "What's the relationship between Tesla's market sentiment and financial performance?",
                "How do Tesla's financial metrics compare to its stock valuation?"
            ]
        }
        
        return suggestions.get(category, [])
    
    def get_all_suggestions(self) -> Dict[str, List[str]]:
        """Get all suggestion questions organized by category"""
        return {
            "Market Intelligence": [
                "How is Tesla's stock performing today?",
                "What's the current market sentiment for Tesla?",
                "How does Tesla compare to its competitors?",
                "What are the main risks facing Tesla?",
                "What are the latest EV industry trends?"
            ],
            "Financial Modeling": [
                "What's Tesla's revenue projection for 2025?",
                "Analyze Tesla's profit margins",
                "Show me Tesla's cash flow analysis",
                "What are Tesla's key financial drivers?",
                "Compare Tesla's financial performance across scenarios"
            ],
            "Hybrid Analysis": [
                "How might market sentiment affect Tesla's financial projections?",
                "What's the relationship between Tesla's stock performance and financial metrics?",
                "How do Tesla's financial results impact its market valuation?"
            ]
        }
