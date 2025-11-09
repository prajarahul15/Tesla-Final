from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import uuid
from datetime import datetime
import sys
import asyncio

# Load environment early (before importing any modules that use env)
ROOT_DIR = Path(__file__).parent
try:
    # Prefer UTF-8; if the file was saved with a different encoding, try a fallback
    load_dotenv(ROOT_DIR / '.env', override=True, encoding='utf-8')
except Exception:
    try:
        load_dotenv(ROOT_DIR / '.env', override=True, encoding='utf-16')
    except Exception:
        pass

# Add the backend directory to Python path for imports
sys.path.append(str(ROOT_DIR))

from models.financial_models import *
from services.financial_calculator import TeslaFinancialCalculator
from services.enhanced_financial_calculator import EnhancedTeslaCalculator
from services.segment_analyzer import TeslaSegmentAnalyzer
from services.analytics_engine import AnalyticsEngine
from services.ai_agents import proactive_insights_agent, prophet_forecasting_agent, tesla_ai_agent, _chat_completion, income_statement_insights_agent, cross_statement_insights_agent
from services.tesla_fa_agent import tesla_fa_agent
from services.agent_orchestrator import agent_orchestrator, AgentType, TaskType
from agents.enhanced_agent_orchestrator import EnhancedAgentOrchestrator
from services.metric_forecasting import metric_forecasting_service
from services.energy_services_loader import get_energy_services_loader
from data.tesla_data import generate_all_tesla_assumptions, TESLA_BASE_YEAR_DATA, MACRO_ASSUMPTIONS

# MongoDB connection
try:
    mongo_url = os.environ['MONGO_URL']
    client = AsyncIOMotorClient(mongo_url)
    db = client[os.environ['DB_NAME']]
    print(f"Connected to MongoDB: {os.environ['DB_NAME']}")
except KeyError as e:
    print(f"Missing environment variable: {e}")
    # Fallback for development
    client = None
    db = None

# Create the main app without a prefix
app = FastAPI(title="Tesla Financial Model & Analytics API", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Initialize calculators (with FCF support)
tesla_calculator = TeslaFinancialCalculator()
enhanced_calculator = EnhancedTeslaCalculator()
segment_analyzer = TeslaSegmentAnalyzer()
analytics_engine = AnalyticsEngine()

# Initialize enhanced agent orchestrator
enhanced_agent_orchestrator = EnhancedAgentOrchestrator()

# Initialize analytics engine on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Tesla Financial Model & Analytics API started successfully")
    logger.info("Loading analytics data...")
    
    # Load analytics data
    if analytics_engine.load_data():
        logger.info("Analytics data loaded successfully")
        # Provide monthly economic variables to vehicle forecast engine (optional)
        try:
            if getattr(analytics_engine, "mv_parameters", None) is not None:
                econ_vars = [
                    'Consumer Price Index', 'Dow_Jones_Bank', 'S&P Index', 
                    'FED_FUND_RATE', 'NASDAQ_TECH', 'KBW_FINTECH', 'FIS Price', 'FIS_Volue'
                ]
                mv = analytics_engine.mv_parameters
                cols = [c for c in econ_vars if c in mv.columns]
                if 'DATE' in mv.columns and cols:
                    econ_df = mv[['DATE'] + cols].copy()
                    econ_df['DATE'] = econ_df['DATE'].dt.to_period('M').dt.to_timestamp()
                    econ_df = econ_df.groupby('DATE', as_index=False).mean(numeric_only=True)
                    vehicle_engine.set_economic_variables(econ_df)
                    logger.info(f"Vehicle engine econ columns set: {cols}")
        except Exception as e:
            logger.warning(f"Econ wiring failed: {e}")
    else:
        logger.error("Failed to load analytics data")
    
    # Initialize and run autonomous vehicle forecast agent
    logger.info("🤖 Initializing Vehicle Forecast Agent...")
    try:
        from agents.vehicle_forecast_agent import initialize_vehicle_forecast_agent
        agent = initialize_vehicle_forecast_agent(vehicle_engine)
        # Run initial forecast generation in background
        asyncio.create_task(agent.generate_all_forecasts())
        logger.info("✅ Vehicle Forecast Agent started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Vehicle Forecast Agent: {e}")

# AI AGENTS ENDPOINTS

class ProactiveInsightsRequest(BaseModel):
    scenario: str
    model_data: Optional[Dict] = None

@api_router.post("/ai/proactive-insights")
async def get_proactive_insights(request: ProactiveInsightsRequest):
    """Get proactive AI insights for financial model"""
    try:
        # Check if OpenAI API key is available
        if not os.getenv('OPENAI_API_KEY'):
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # If no model data provided, generate it
        if not request.model_data:
            scenario_enum = ScenarioType(request.scenario.lower())
            # Use module-level tesla_calculator
            model = tesla_calculator.build_complete_financial_model(scenario_enum)
            model_data = model.dict()
        else:
            model_data = request.model_data
        
        insights = proactive_insights_agent.analyze_financial_model(model_data, request.scenario)
        
        return {
            "success": True,
            "scenario": request.scenario,
            "insights": insights,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

class ProphetForecastRequest(BaseModel):
    historical_data: List[Dict]
    periods: int = 12
    metric_name: str = "revenue"

@api_router.post("/ai/prophet-forecast")
async def generate_prophet_forecast(request: ProphetForecastRequest):
    """Generate Prophet-based forecast with AI insights"""
    try:
        if not os.getenv('OPENAI_API_KEY'):
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        forecast_result = prophet_forecasting_agent.generate_prophet_forecast(
            request.historical_data, 
            request.periods
        )
        
        return {
            "success": True,
            "metric": request.metric_name,
            "forecast_result": forecast_result,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

class TeslaAIRequest(BaseModel):
    scenario: str = "base"
    changes: Dict[str, float] = {}

@api_router.post("/ai/tesla-agent/initialize")
async def initialize_tesla_ai_agent(scenario: str = "base"):
    """Initialize Tesla AI Agent with base scenario"""
    try:
        result = tesla_ai_agent.initialize_base_model(scenario)
        
        return {
            "success": True,
            "initialization": result,
            "available_sliders": {
                "asp_change": {"min": -30, "max": 30, "step": 1, "unit": "%"},
                "cost_change": {"min": -20, "max": 40, "step": 1, "unit": "%"},
                "delivery_change": {"min": -50, "max": 100, "step": 5, "unit": "%"}
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing Tesla AI agent: {str(e)}")

@api_router.post("/ai/tesla-agent/simulate")
async def simulate_tesla_changes(request: TeslaAIRequest):
    """Simulate Tesla model changes with AI insights"""
    try:
        if not os.getenv('OPENAI_API_KEY'):
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        # Initialize if not already done
        if not tesla_ai_agent.base_assumptions:
            tesla_ai_agent.initialize_base_model(request.scenario)
        
        simulation_result = tesla_ai_agent.simulate_slider_changes(request.changes)
        
        return {
            "success": True,
            "simulation": simulation_result,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error simulating changes: {str(e)}")

@api_router.get("/ai/tesla-agent/vehicle-models")
async def get_vehicle_models_info():
    """Get information about Tesla vehicle models for the AI agent"""
    try:
        from data.tesla_enhanced_data import VEHICLE_MODEL_DATA
        
        models_info = {}
        for model_key, model_data in VEHICLE_MODEL_DATA["models"].items():
            models_info[model_key] = {
                "name": model_data["name"],
                "segment": model_data["segment"],
                "base_asp": model_data["base_asp"],
                "max_capacity": model_data["max_capacity"],
                "growth_trajectory": model_data["growth_trajectory"]
            }
        
        return {
            "success": True,
            "vehicle_models": models_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting vehicle models: {str(e)}")

# Tesla Financial Assistant Chat Endpoints
class TeslaFAChatRequest(BaseModel):
    message: str = Field(..., description="User's question or message")
    context: str = Field(default="financial_modeling", description="Context type for the conversation")

@api_router.post("/tesla-fa/chat")
async def tesla_fa_chat(request: TeslaFAChatRequest):
    """Chat with Tesla Financial Assistant"""
    try:
        if not os.getenv('OPENAI_API_KEY'):
            raise HTTPException(status_code=500, detail="OpenAI API key not configured")
        
        response = tesla_fa_agent.generate_response(request.message, request.context)
        
        return {
            "success": True,
            "response": response,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")

@api_router.get("/tesla-fa/capabilities")
async def get_tesla_fa_capabilities():
    """Get Tesla FA capabilities and information"""
    try:
        capabilities = tesla_fa_agent.get_capabilities_summary()
        return {
            "success": True,
            "capabilities": capabilities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting capabilities: {str(e)}")

# Agent Orchestrator Endpoints
class OrchestratedQueryRequest(BaseModel):
    query: str = Field(..., description="User's query to be processed by orchestrator")
    session_id: Optional[str] = Field(None, description="Session ID for context tracking")
    context: Optional[Dict] = Field(default_factory=dict, description="Additional context (scenario, year, etc.)")

@api_router.get("/test/news-sentiment")
async def test_news_sentiment():
    """Test endpoint for news sentiment - bypasses orchestrator"""
    from agents.market_sentiment_agent import MarketSentimentAgent
    agent = MarketSentimentAgent()
    
    # Get news sentiment
    news_data = await agent.get_news_sentiment("TSLA")
    
    # Get comprehensive sentiment
    comprehensive = await agent.get_comprehensive_sentiment("TSLA")
    
    return {
        "success": True,
        "news_data": news_data,
        "comprehensive_sentiment": comprehensive,
        "has_articles": len(news_data.get("recent_news", [])) > 0
    }

@api_router.get("/market-intelligence/news-sentiment/{symbol}")
async def get_news_and_sentiment(symbol: str = "TSLA"):
    """Direct endpoint for news and sentiment - for frontend use"""
    from agents.market_sentiment_agent import MarketSentimentAgent
    agent = MarketSentimentAgent()
    
    # Get comprehensive sentiment which includes news
    result = await agent.get_comprehensive_sentiment(symbol)
    
    return {
        "success": True,
        "data": result
    }

@api_router.get("/market-intelligence/industry-trends/{symbol}")
async def get_industry_trends(symbol: str = "TSLA"):
    """Get comprehensive EV industry trends with real-time data"""
    from agents.industry_trends_agent import IndustryTrendsAgent
    
    logger.info(f"API endpoint called for industry trends: {symbol}")
    
    try:
        agent = IndustryTrendsAgent()
        trends_data = await agent.get_industry_trends(symbol)
        
        logger.info(f"Industry trends fetched - API count: {trends_data.get('api_count', 0)}/{trends_data.get('total_metrics', 6)}")
        
        return {
            "success": True,
            "data": trends_data
        }
    except Exception as e:
        logger.error(f"Error fetching industry trends: {str(e)}")
        # Return fallback data on error
        from agents.industry_trends_agent import IndustryTrendsAgent
        agent = IndustryTrendsAgent()
        return {
            "success": True,
            "data": agent._get_fallback_data()
        }

@api_router.post("/orchestrator/ask")
async def orchestrated_query(request: OrchestratedQueryRequest):
    """
    Enhanced orchestrated AI endpoint - routes query to appropriate agents
    and coordinates multi-agent workflows including market intelligence
    """
    try:
        if not os.getenv('OPENAI_API_KEY'):
            raise HTTPException(status_code=500, detail="OpenAI API key not configured for orchestration")
        
        # Use enhanced orchestrator for market intelligence integration
        result = await enhanced_agent_orchestrator.execute_workflow(
            query=request.query,
            session_id=request.session_id,
            context=request.context
        )
        
        return {
            "success": True,
            "result": result,
            "query_type": result.get("query_type", "general"),
            "agents_used": result.get("agents_used", []),
            "tasks_executed": result.get("tasks_executed", []),
            "timestamp": result.get("timestamp", datetime.now().isoformat()),
            "coordination": result.get("coordination", {"enabled": False})  # Multi-Agent Coordination metadata
        }
        
    except Exception as e:
        logger.error(f"Enhanced orchestration error: {str(e)}")
        # Fallback to original orchestrator
        try:
            fallback_result = await agent_orchestrator.execute_workflow(
                query=request.query,
                session_id=request.session_id,
                context=request.context
            )
            return {
                "success": True,
                "result": fallback_result,
                "query_type": "fallback",
                "agents_used": ["OriginalAgentOrchestrator"],
                "tasks_executed": ["fallback_orchestration"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as fallback_error:
            raise HTTPException(status_code=500, detail=f"Orchestration error: {str(e)}")

@api_router.get("/orchestrator/agents")
async def list_orchestrator_agents():
    """List all available agents in the orchestration system"""
    try:
        agents = agent_orchestrator.list_available_agents()
        return {
            "success": True,
            "total_agents": len(agents),
            "agents": agents
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing agents: {str(e)}")

@api_router.get("/orchestrator/session/{session_id}")
async def get_session_summary(session_id: str):
    """Get summary of an orchestration session"""
    try:
        summary = agent_orchestrator.get_session_summary(session_id)
        return {
            "success": True,
            "session": summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting session: {str(e)}")

@api_router.delete("/orchestrator/session/{session_id}")
async def clear_session(session_id: str):
    """Clear an orchestration session"""
    try:
        agent_orchestrator.context_store.clear_session(session_id)
        return {
            "success": True,
            "message": f"Session {session_id} cleared"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

# ==========================================
# MARKET INTELLIGENCE ENDPOINTS
# ==========================================

@api_router.get("/market-intelligence/stock-history/{symbol}")
async def get_stock_historical_data(symbol: str = "TSLA", years: int = 5):
    """
    Get historical stock data for a symbol (default: Tesla)
    Returns up to 5 years of daily stock data with technical indicators
    """
    try:
        from agents.stock_data_agent import StockDataAgent
        
        stock_agent = StockDataAgent()
        
        logger.info(f"API endpoint called for {symbol} with {years} years")
        
        # Fetch historical data
        historical_data = await stock_agent.get_historical_data(symbol, years)
        
        logger.info(f"Received {len(historical_data) if historical_data else 0} days of data")
        
        if not historical_data or len(historical_data) == 0:
            logger.error(f"No historical data available for {symbol}")
            raise HTTPException(
                status_code=404, 
                detail=f"No historical data found for symbol {symbol}. This could be due to API rate limits or invalid symbol."
            )
        
        # Calculate technical indicators
        technical_indicators = stock_agent.calculate_technical_indicators_from_history(historical_data)
        
        # Calculate 52-week high/low from historical data
        recent_52_weeks = historical_data[-252:] if len(historical_data) >= 252 else historical_data
        highs = [day['high'] for day in recent_52_weeks]
        lows = [day['low'] for day in recent_52_weeks]
        volumes = [day['volume'] for day in recent_52_weeks]
        
        week_52_high = max(highs) if highs else 0
        week_52_low = min(lows) if lows else 0
        avg_volume = sum(volumes) // len(volumes) if volumes else 0
        
        return {
            "success": True,
            "symbol": symbol,
            "data_period": f"{years} years",
            "total_days": len(historical_data),
            "historical_data": historical_data,
            "technical_indicators": technical_indicators,
            "summary": {
                "current_price": historical_data[-1]['close'] if historical_data else 0,
                "fifty_two_week_high": week_52_high,
                "fifty_two_week_low": week_52_low,
                "average_volume": avg_volume,
                "latest_date": historical_data[-1]['date'] if historical_data else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stock history for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching stock history: {str(e)}"
        )

@api_router.get("/market-intelligence/stock-current/{symbol}")
async def get_current_stock_data(symbol: str = "TSLA"):
    """
    Get current stock data for a symbol (default: Tesla)
    Returns current price, daily change, volume, market cap, 52-week range
    """
    try:
        from agents.stock_data_agent import StockDataAgent
        
        stock_agent = StockDataAgent()
        analysis = await stock_agent.get_comprehensive_analysis(symbol)
        
        if 'error' in analysis:
            raise HTTPException(status_code=500, detail=analysis['error'])
        
        return {
            "success": True,
            "symbol": symbol,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching current stock data for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching current stock data: {str(e)}"
        )

# Year-specific AI simulation endpoint (isolated for New Vehicle Models)
from data.tesla_enhanced_data import get_enhanced_tesla_drivers

class TeslaAISimulateYearReq(BaseModel):
    scenario: str = "base"
    year: int
    changes: Dict[str, float]

# Internal helper to fetch Excel-backed year summary as a plain dict
def _get_year_summary_dict(year: int) -> Optional[Dict]:
    try:
        # Ensure data is loaded
        if vehicle_engine.monthly_data is None:
            if not vehicle_engine.load_data():
                return None
        df = vehicle_engine.monthly_data.copy()
        df_year = df[df["DATE"].dt.year == int(year)]
        if df_year.empty:
            return {"totals": {"total_deliveries": 0.0, "total_production": 0.0}, "models": []}
        grp = df_year.groupby("model_key", as_index=False).agg({"deliveries": "sum", "production": "sum"})
        models = [
            {
                "model_key": str(row["model_key"]),
                "deliveries": float(row.get("deliveries", 0) or 0),
                "production": float(row.get("production", 0) or 0),
            }
            for _, row in grp.iterrows()
        ]
        totals = {
            "total_deliveries": float(df_year["deliveries"].sum()),
            "total_production": float(df_year["production"].sum()),
        }
        return {"totals": totals, "models": models}
    except Exception:
        return None

@api_router.post("/ai/tesla-agent/initialize-year")
async def initialize_tesla_ai_agent_for_year(scenario: str = "base", year: int = 2025):
    """Initialize Tesla AI Agent for a specific year and scenario.

    Mirrors the /ai/tesla-agent/initialize response shape but uses year-aware
    base assumptions so the UI numbers align with the selected year.
    """
    try:
        from services.ai_agents import TeslaAIAgent as _TeslaAIAgent
        local_agent = _TeslaAIAgent()
        local_agent.current_scenario = scenario

        # Start from scenario drivers to inherit pricing multipliers, etc.
        base_drivers = get_enhanced_tesla_drivers(ScenarioType(scenario.lower()), int(year))

        # If Excel-backed yearly summary is available, override deliveries with it
        summary = _get_year_summary_dict(int(year))

        if summary and isinstance(summary, dict) and "models" in summary:
            excel_deliveries = {}
            for m in summary["models"]:
                try:
                    key = m.get("model_key")
                    val = int(m.get("deliveries") or 0)
                    if key:
                        excel_deliveries[key] = val
                except Exception:
                    continue
            if excel_deliveries:
                base_drivers["projected_deliveries"] = excel_deliveries

        local_agent.base_assumptions = base_drivers

        initialization = {
            "initialized": True,
            "scenario": scenario,
            "year": int(year),
            "base_deliveries": local_agent.base_assumptions["projected_deliveries"],
            "base_asp_multiplier": local_agent.base_assumptions.get("asp_multiplier", 1.0),
        }

        return {
            "success": True,
            "initialization": initialization,
            "available_sliders": {
                "asp_change": {"min": -30, "max": 30, "step": 1, "unit": "%"},
                "cost_change": {"min": -20, "max": 40, "step": 1, "unit": "%"},
                "delivery_change": {"min": -50, "max": 100, "step": 5, "unit": "%"},
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing year model: {str(e)}")

@api_router.post("/ai/tesla-agent/simulate-year")
async def simulate_tesla_changes_for_year(req: TeslaAISimulateYearReq):
    try:
        # Create a fresh agent instance to avoid global state mutation
        from services.ai_agents import TeslaAIAgent as _TeslaAIAgent
        local_agent = _TeslaAIAgent()
        local_agent.current_scenario = req.scenario
        # Base on scenario drivers, then override deliveries with Excel summary if present
        base_drivers = get_enhanced_tesla_drivers(ScenarioType(req.scenario.lower()), int(req.year))
        summary = _get_year_summary_dict(int(req.year))
        if summary and isinstance(summary, dict) and "models" in summary:
            excel_deliveries = {}
            for m in summary["models"]:
                try:
                    key = m.get("model_key")
                    val = int(m.get("deliveries") or 0)
                    if key:
                        excel_deliveries[key] = val
                except Exception:
                    continue
            if excel_deliveries:
                base_drivers["projected_deliveries"] = excel_deliveries
        local_agent.base_assumptions = base_drivers
        result = local_agent.simulate_slider_changes(req.changes)
        return {"success": True, "simulation": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error simulating year changes: {str(e)}")

# Professional Dashboard API Endpoints

@api_router.get("/analytics/overview")
async def get_data_overview():
    """Get comprehensive data overview metrics"""
    try:
        metrics = analytics_engine.get_data_overview_metrics()
        if metrics is None:
            raise HTTPException(status_code=500, detail="Failed to calculate overview metrics")
        
        return {
            "success": True,
            "data": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating overview: {str(e)}")

@api_router.get("/analytics/economic-variables")
async def get_economic_variables():
    """Get economic variables data for analysis"""
    try:
        econ_data = analytics_engine.get_economic_variables_data()
        if econ_data is None:
            raise HTTPException(status_code=500, detail="Economic variables data not available")
        
        return {
            "success": True,
            "data": econ_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting economic data: {str(e)}")

@api_router.get("/analytics/lineups")
async def get_available_lineups():
    """Get list of available lineups for forecasting"""
    try:
        if analytics_engine.sample_data is None:
            if not analytics_engine.load_data():
                raise HTTPException(status_code=500, detail="Failed to load data")
        
        lineups = analytics_engine.sample_data['Lineup'].unique().tolist()
        
        # Get metadata for each lineup
        lineup_metadata = []
        for lineup in lineups:
            lineup_data = analytics_engine.sample_data[analytics_engine.sample_data['Lineup'] == lineup]
            lineup_metadata.append({
                'lineup': lineup,
                'profile': lineup_data['Profile'].iloc[0],
                'line_item': lineup_data['Line_Item'].iloc[0],
                'records': int(len(lineup_data)),
                'date_range': {
                    'start': lineup_data['DATE'].min().strftime('%Y-%m-%d'),
                    'end': lineup_data['DATE'].max().strftime('%Y-%m-%d')
                },
                'total_actual': int(lineup_data['Actual'].sum()),
                'total_plan': int(lineup_data['Plan'].sum())
            })
        
        return {
            "success": True,
            "lineups": lineup_metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting lineups: {str(e)}")

class ForecastRequest(BaseModel):
    lineup: str
    forecast_type: str  # 'univariate' or 'multivariate'
    months_ahead: int = 12

@api_router.post("/analytics/forecast")
async def generate_forecast(request: ForecastRequest):
    """Generate forecast for specified lineup and type"""
    try:
        if request.forecast_type not in ['univariate', 'multivariate']:
            raise HTTPException(status_code=400, detail="Forecast type must be 'univariate' or 'multivariate'")
        
        if request.months_ahead < 1 or request.months_ahead > 24:
            raise HTTPException(status_code=400, detail="Months ahead must be between 1 and 24")
        
        # Ensure data is loaded
        if analytics_engine.sample_data is None:
            if not analytics_engine.load_data():
                raise HTTPException(status_code=500, detail="Failed to load data")
        
        # Check if lineup exists
        available_lineups = analytics_engine.sample_data['Lineup'].unique()
        if request.lineup not in available_lineups:
            raise HTTPException(status_code=400, detail=f"Lineup {request.lineup} not found. Available: {list(available_lineups)}")
        
        # Generate forecast
        forecast_result = analytics_engine.generate_forecast(
            request.lineup, 
            request.forecast_type, 
            request.months_ahead
        )
        
        if forecast_result is None:
            raise HTTPException(status_code=500, detail="Failed to generate forecast")
        
        return {
            "success": True,
            "forecast": forecast_result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating forecast: {str(e)}")

class CompareRequest(BaseModel):
    lineup: str
    months_ahead: int = 12

class UnivariateForecastRequest(BaseModel):
    metric: str = Field(..., description="Metric to forecast (e.g., 'revenue_millions')")
    horizon: int = Field(12, description="Number of months to forecast into the future")
    test_months: int = Field(6, description="Number of months to use for testing")

class MultivariateForecastRequest(BaseModel):
    metrics: List[str] = Field(..., description="List of metrics to forecast")
    horizon: int = Field(12, description="Number of months to forecast into the future")
    test_months: int = Field(6, description="Number of months to use for testing")

@api_router.post("/analytics/compare-forecasts")
async def compare_forecasts(request: CompareRequest):
    """Compare univariate vs multivariate forecasting methods"""
    try:
        # Ensure data is loaded
        if analytics_engine.sample_data is None:
            if not analytics_engine.load_data():
                raise HTTPException(status_code=500, detail="Failed to load data")
        
        # Check if lineup exists
        available_lineups = analytics_engine.sample_data['Lineup'].unique()
        if request.lineup not in available_lineups:
            raise HTTPException(status_code=400, detail=f"Lineup {request.lineup} not found")
        
        # Generate comparison
        comparison_result = analytics_engine.compare_forecast_methods(
            request.lineup, 
            request.months_ahead
        )
        
        if comparison_result is None:
            raise HTTPException(status_code=500, detail="Failed to generate forecast comparison")
        
        return {
            "success": True,
            "comparison": comparison_result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing forecasts: {str(e)}")

@api_router.get("/tesla/test-enhanced")
async def test_enhanced_features():
    """Test enhanced features"""
    try:
        # Test basic enhanced calculator functionality
        from data.tesla_enhanced_data import get_enhanced_tesla_drivers
        
        drivers = get_enhanced_tesla_drivers(ScenarioType.BASE, 2024)
        
        return {
            "success": True,
            "message": "Enhanced features working",
            "sample_data": {
                "scenario": drivers["scenario"].value,
                "year": drivers["year"],
                "projected_deliveries": {k: int(v) for k, v in drivers["projected_deliveries"].items()},
                "energy_growth_rate": float(drivers["energy_growth_rate"])
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Enhanced Tesla Financial Model Endpoints (PHASE 1-3)

@api_router.post("/tesla/enhanced-model/{scenario}")
async def generate_enhanced_financial_model(scenario: str):
    """Generate enhanced financial model with driver-based calculations"""
    try:
        scenario_enum = ScenarioType(scenario.lower())
        model = enhanced_calculator.build_enhanced_financial_model(scenario_enum)
        
        # Store in database if available (create a copy to avoid ObjectId issues)
        if db is not None:
            try:
                import copy
                model_copy = copy.deepcopy(model)
                await db.enhanced_financial_models.insert_one(model_copy)
            except Exception as e:
                print(f"Database insert error: {e}")
        
        return {
            "success": True,
            "message": f"Enhanced financial model generated for {scenario} scenario",
            "model": model
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario. Use 'best', 'base', or 'worst'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating enhanced model: {str(e)}")

@api_router.get("/tesla/enhanced-comparison")
async def get_enhanced_scenario_comparison():
    """Enhanced scenario comparison with vehicle models and 10-year forecasts"""
    try:
        enhanced_models = {}
        for scenario in ["best", "base", "worst"]:
            scenario_enum = ScenarioType(scenario)
            model = enhanced_calculator.build_enhanced_financial_model(scenario_enum)
            enhanced_models[scenario] = model
        
        # Create enhanced comparison
        comparison = {
            "revenue_comparison": {},
            "valuation_comparison": {},
            "vehicle_model_comparison": {},
            "segment_comparison": {}
        }
        
        for scenario, model in enhanced_models.items():
            # Final year projections (2033)
            final_income = model["income_statements"][-1]
            dcf = model["dcf_valuation"]
            
            comparison["revenue_comparison"][scenario] = {
                "final_year_revenue": final_income["total_revenue"],
                "automotive_revenue": final_income["automotive_revenue"],
                "energy_revenue": final_income["energy_revenue"],
                "services_revenue": final_income["services_revenue"],
                "10yr_cagr": ((final_income["total_revenue"] / TESLA_BASE_YEAR_DATA["total_revenue"]) ** (1/10)) - 1
            }
            
            comparison["valuation_comparison"][scenario] = {
                "price_per_share": dcf["price_per_share"],
                "enterprise_value": dcf["enterprise_value"],
                "wacc": dcf["wacc"]
            }
            
            # Vehicle model breakdown for final year
            vehicle_breakdown = final_income["revenue_breakdown"]["automotive_revenue_by_model"]
            comparison["vehicle_model_comparison"][scenario] = vehicle_breakdown
            
            # Segment analysis
            comparison["segment_comparison"][scenario] = {
                "automotive_margin": final_income["margins"]["automotive_margin"],
                "energy_margin": final_income["margins"]["energy_margin"],
                "services_margin": final_income["margins"]["services_margin"]
            }
        
        return {
            "success": True,
            "enhanced_models": enhanced_models,
            "comparison_summary": comparison
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/tesla/vehicle-analysis/{scenario}")
async def get_vehicle_model_analysis(scenario: str):
    """PHASE 1: Detailed vehicle model analysis"""
    try:
        scenario_enum = ScenarioType(scenario.lower())
        model = enhanced_calculator.build_enhanced_financial_model(scenario_enum)
        
        vehicle_analysis = {
            "scenario": scenario,
            "vehicle_trends": {},
            "model_performance": {},
            "delivery_projections": {}
        }
        
        # Extract vehicle data for all years
        for income_stmt in model["income_statements"]:
            year = income_stmt["year"]
            vehicle_data = income_stmt["revenue_breakdown"]["automotive_revenue_by_model"]
            
            vehicle_analysis["vehicle_trends"][year] = vehicle_data
        
        # Calculate model performance metrics
        for model_key in vehicle_analysis["vehicle_trends"][2024]:
            initial_year_data = vehicle_analysis["vehicle_trends"][2024][model_key]
            final_year_data = vehicle_analysis["vehicle_trends"][2033][model_key]
            
            delivery_cagr = ((final_year_data["deliveries"] / initial_year_data["deliveries"]) ** (1/9)) - 1 if initial_year_data["deliveries"] > 0 else 0
            revenue_cagr = ((final_year_data["revenue"] / initial_year_data["revenue"]) ** (1/9)) - 1 if initial_year_data["revenue"] > 0 else 0
            
            vehicle_analysis["model_performance"][model_key] = {
                "delivery_cagr": delivery_cagr,
                "revenue_cagr": revenue_cagr,
                "initial_deliveries": initial_year_data["deliveries"],
                "final_deliveries": final_year_data["deliveries"],
                "initial_asp": initial_year_data["asp"],
                "final_asp": final_year_data["asp"],
                "asp_trend": (final_year_data["asp"] / initial_year_data["asp"]) - 1 if initial_year_data["asp"] > 0 else 0
            }
        
        return {
            "success": True,
            "vehicle_analysis": vehicle_analysis
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/tesla/segment-analysis")
async def get_business_segment_analysis():
    """PHASE 2: Business segment analysis across all scenarios"""
    try:
        enhanced_models = {}
        for scenario in ["best", "base", "worst"]:
            scenario_enum = ScenarioType(scenario)
            model = enhanced_calculator.build_enhanced_financial_model(scenario_enum)
            enhanced_models[scenario] = model
        
        segment_analysis = segment_analyzer.analyze_business_segments(enhanced_models)
        
        return {
            "success": True,
            "segment_analysis": segment_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/tesla/bridge-analysis/{scenario}")
async def get_bridge_analysis(scenario: str):
    """PHASE 3: Bridge analysis (waterfall charts)"""
    try:
        scenario_enum = ScenarioType(scenario.lower())
        model = enhanced_calculator.build_enhanced_financial_model(scenario_enum)
        
        # Calculate revenue bridge from first to last year
        income_statements = model["income_statements"]
        
        if len(income_statements) >= 2:
            base_year_data = income_statements[0]
            final_year_data = income_statements[-1]
            
            revenue_bridge = segment_analyzer.calculate_revenue_bridge(base_year_data, final_year_data)
            
            # Cash flow bridge
            dcf_data = model["dcf_valuation"]
            if dcf_data["projected_free_cash_flows"]:
                base_fcf = dcf_data["projected_free_cash_flows"][0]
                final_fcf = dcf_data["projected_free_cash_flows"][-1]
                cash_flow_bridge = segment_analyzer.calculate_cash_flow_bridge(base_fcf, final_fcf, income_statements)
            else:
                cash_flow_bridge = {}
            
            return {
                "success": True,
                "scenario": scenario,
                "revenue_bridge": revenue_bridge,
                "cash_flow_bridge": cash_flow_bridge
            }
        else:
            raise HTTPException(status_code=400, detail="Insufficient data for bridge analysis")
            
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/tesla/price-volume-mix")
async def get_price_volume_mix_analysis():
    """PHASE 3: Price-Volume-Mix analysis"""
    try:
        enhanced_models = {}
        for scenario in ["best", "base", "worst"]:
            scenario_enum = ScenarioType(scenario)
            model = enhanced_calculator.build_enhanced_financial_model(scenario_enum)
            enhanced_models[scenario] = model
        
        pvm_analysis = segment_analyzer.analyze_price_volume_mix(enhanced_models)
        
        return {
            "success": True,
            "price_volume_mix_analysis": pvm_analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/tesla/comprehensive-analysis")
async def get_comprehensive_analysis():
    """Complete analysis combining all Phase 1-3 features"""
    try:
        enhanced_models = {}
        for scenario in ["best", "base", "worst"]:
            scenario_enum = ScenarioType(scenario)
            model = enhanced_calculator.build_enhanced_financial_model(scenario_enum)
            enhanced_models[scenario] = model
        
        comprehensive_analysis = segment_analyzer.generate_comprehensive_analysis(enhanced_models)
        
        return {
            "success": True,
            "comprehensive_analysis": comprehensive_analysis,
            "model_features": [
                "10-year forecasts (2024-2033)",
                "Vehicle model granularity (6 models)",
                "Driver-based revenue calculations",
                "Business segment analysis (Automotive/Energy/Services)",
                "Enhanced DCF with sensitivity analysis",
                "Working capital modeling",
                "Bridge analysis (Revenue/Cash Flow)",
                "Price-Volume-Mix analytics"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/tesla/overview")
async def get_tesla_overview():
    """Get Tesla overview data and market assumptions"""
    return {
        "tesla_base_data": TESLA_BASE_YEAR_DATA,
        "macro_assumptions": MACRO_ASSUMPTIONS,
        "model_description": "Tesla 5-Year Financial Model (2025-2029) with DCF Valuation",
        "scenarios": ["best", "base", "worst"],
        "forecast_years": [2025, 2026, 2027, 2028, 2029]
    }

@api_router.get("/tesla/assumptions/{scenario}")
async def get_scenario_assumptions(scenario: str):
    """Get all assumptions for a specific scenario"""
    try:
        scenario_enum = ScenarioType(scenario.lower())
        assumptions = []
        
        for year in [2025, 2026, 2027, 2028, 2029]:
            from data.tesla_data import get_tesla_assumptions
            assumption_dict = get_tesla_assumptions(scenario_enum, year)
            assumptions.append(assumption_dict)
        
        return {
            "scenario": scenario,
            "assumptions": assumptions
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario. Use 'best', 'base', or 'worst'")

# Debug endpoint removed - using fresh calculator instances now

@api_router.get("/tesla/model/{scenario}")
async def get_financial_model(scenario: str):
    """Get complete financial model for scenario"""
    try:
        scenario_enum = ScenarioType(scenario.lower())
        # Use module-level tesla_calculator
        model = tesla_calculator.build_complete_financial_model(scenario_enum)
        
        return model.dict()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario. Use 'best', 'base', or 'worst'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating model: {str(e)}")

@api_router.post("/tesla/model/{scenario}")
async def generate_financial_model(scenario: str):
    """Generate and store complete financial model for scenario"""
    try:
        scenario_enum = ScenarioType(scenario.lower())
        # Use module-level tesla_calculator
        model = tesla_calculator.build_complete_financial_model(scenario_enum)
        
        # Store in database if available
        if db is not None:
            try:
                await db.financial_models.insert_one(model.dict())
            except Exception as e:
                print(f"Database insert error: {e}")
        
        return {
            "success": True,
            "message": f"Financial model generated for {scenario} scenario",
            "model": model.dict()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario. Use 'best', 'base', or 'worst'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating model: {str(e)}")

@api_router.get("/tesla/model/{scenario}/income-statement")
async def get_income_statements(scenario: str):
    """Get income statements for all years in scenario"""
    try:
        scenario_enum = ScenarioType(scenario.lower())
        # Use module-level tesla_calculator
        model = tesla_calculator.build_complete_financial_model(scenario_enum)
        
        income_statements = [stmt.dict() for stmt in model.income_statements]
        
        return {
            "scenario": scenario,
            "income_statements": income_statements
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/tesla/model/{scenario}/balance-sheet")
async def get_balance_sheets(scenario: str):
    """Get balance sheets for all years in scenario"""
    try:
        scenario_enum = ScenarioType(scenario.lower())
        # Use module-level tesla_calculator
        model = tesla_calculator.build_complete_financial_model(scenario_enum)
        
        balance_sheets = [bs.dict() for bs in model.balance_sheets]
        
        return {
            "scenario": scenario,
            "balance_sheets": balance_sheets
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/tesla/model/{scenario}/cash-flow")
async def get_cash_flows(scenario: str):
    """Get cash flow statements for all years in scenario"""
    try:
        scenario_enum = ScenarioType(scenario.lower())
        # Use module-level tesla_calculator
        model = tesla_calculator.build_complete_financial_model(scenario_enum)
        
        cash_flows = [cf.dict() for cf in model.cash_flow_statements]
        
        return {
            "scenario": scenario,
            "cash_flow_statements": cash_flows
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/tesla/model/{scenario}/dcf-valuation")
async def get_dcf_valuation(scenario: str):
    """Get DCF valuation for scenario"""
    try:
        scenario_enum = ScenarioType(scenario.lower())
        # Use module-level tesla_calculator
        model = tesla_calculator.build_complete_financial_model(scenario_enum)
        
        return {
            "scenario": scenario,
            "dcf_valuation": model.dcf_valuation.dict()
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------- Income Statement Simulation (Standard model) --------
class IncomeStatementSimReq(BaseModel):
    """User overrides to simulate a single-year income statement.

    Only specified fields are overridden; others remain per scenario defaults.
    """
    year: Optional[int] = Field(None, description="Target forecast year. If omitted, apply to all years 2025-2029.")
    automotive_revenue_growth: Optional[float] = Field(None, description="Override Y/Y automotive revenue growth (e.g., 0.22)")
    services_revenue_growth: Optional[float] = Field(None, description="Override Y/Y services revenue growth (e.g., 0.30)")
    gross_margin_automotive: Optional[float] = Field(None, description="Override automotive gross margin (0-1)")
    gross_margin_services: Optional[float] = Field(None, description="Override services gross margin (0-1)")
    rd_as_percent_revenue: Optional[float] = Field(None, description="Override R&D as % of revenue (0-1)")
    sga_as_percent_revenue: Optional[float] = Field(None, description="Override SG&A as % of revenue (0-1)")
    include_ai_insights: bool = Field(default=True)

class CrossStatementSimRequest(BaseModel):
    """Request model for simulating all three financial statements with comprehensive parameters"""
    year: Optional[int] = Field(None, description="Target forecast year. If omitted, apply to all years 2025-2029.")
    
    # Revenue drivers
    automotive_revenue_growth: Optional[float] = Field(None, description="Override Y/Y automotive revenue growth (e.g., 0.22)")
    services_revenue_growth: Optional[float] = Field(None, description="Override Y/Y services revenue growth (e.g., 0.30)")
    
    # Margin drivers  
    gross_margin_automotive: Optional[float] = Field(None, description="Override automotive gross margin (0-1)")
    gross_margin_services: Optional[float] = Field(None, description="Override services gross margin (0-1)")
    
    # Opex drivers
    rd_as_percent_revenue: Optional[float] = Field(None, description="Override R&D as % of revenue (0-1)")
    sga_as_percent_revenue: Optional[float] = Field(None, description="Override SG&A as % of revenue (0-1)")
    
    # Working capital drivers
    days_sales_outstanding: Optional[int] = Field(None, description="Override days sales outstanding")
    days_inventory_outstanding: Optional[int] = Field(None, description="Override days inventory outstanding")  
    days_payable_outstanding: Optional[int] = Field(None, description="Override days payable outstanding")
    
    # Investment drivers
    capex_as_percent_revenue: Optional[float] = Field(None, description="Override CapEx as % of revenue (0-1)")
    
    # Tax driver
    tax_rate: Optional[float] = Field(None, description="Override effective tax rate (0-1)")
    
    include_ai_insights: bool = Field(default=False, description="Generate AI insights comparing original vs updated statements")

def _build_updated_income_statement(
    scenario_enum: ScenarioType,
    target_year: int,
    overrides: IncomeStatementSimReq,
):
    """Recompute up to target_year with optional overrides applied only to the target year."""
    from data.tesla_data import get_tesla_assumptions

    # Use module-level tesla_calculator
    prev_data: Dict = {}
    updated_income: Optional[IncomeStatement] = None

    for yr in [2025, 2026, 2027, 2028, 2029]:
        if yr > target_year:
            break
        assumption_dict = get_tesla_assumptions(scenario_enum, yr)

        # Apply overrides on the target year only
        if yr == target_year:
            if overrides.automotive_revenue_growth is not None:
                assumption_dict["automotive_revenue_growth"] = float(overrides.automotive_revenue_growth)
            if overrides.services_revenue_growth is not None:
                assumption_dict["services_revenue_growth"] = float(overrides.services_revenue_growth)
            if overrides.gross_margin_automotive is not None:
                # Clamp to sensible bounds [0, 1]
                gm = max(0.0, min(1.0, float(overrides.gross_margin_automotive)))
                assumption_dict["gross_margin_automotive"] = gm
            if overrides.gross_margin_services is not None:
                gm_s = max(0.0, min(1.0, float(overrides.gross_margin_services)))
                assumption_dict["gross_margin_services"] = gm_s
            if overrides.rd_as_percent_revenue is not None:
                assumption_dict["rd_as_percent_revenue"] = max(0.0, float(overrides.rd_as_percent_revenue))
            if overrides.sga_as_percent_revenue is not None:
                assumption_dict["sga_as_percent_revenue"] = max(0.0, float(overrides.sga_as_percent_revenue))

        assumptions = TeslaAssumptions(**assumption_dict)
        income_stmt = tesla_calculator.calculate_income_statement(assumptions, prev_data)

        if yr == target_year:
            updated_income = income_stmt

        # Prepare prev_data for next iteration
        prev_data = {
            "automotive_revenue": income_stmt.automotive_revenue,
            "services_revenue": income_stmt.services_revenue,
        }

    return updated_income

def _build_updated_income_statements_all(
    scenario_enum: ScenarioType,
    overrides: IncomeStatementSimReq,
):
    """Recompute all years (2025-2029), applying overrides to every year."""
    from data.tesla_data import get_tesla_assumptions

    # Use module-level tesla_calculator
    prev_data: Dict = {}
    updated_statements: List[IncomeStatement] = []

    for yr in [2025, 2026, 2027, 2028, 2029]:
        assumption_dict = get_tesla_assumptions(scenario_enum, yr)

        # Apply same overrides to each year if provided
        if overrides.automotive_revenue_growth is not None:
            assumption_dict["automotive_revenue_growth"] = float(overrides.automotive_revenue_growth)
        if overrides.services_revenue_growth is not None:
            assumption_dict["services_revenue_growth"] = float(overrides.services_revenue_growth)
        if overrides.gross_margin_automotive is not None:
            gm = max(0.0, min(1.0, float(overrides.gross_margin_automotive)))
            assumption_dict["gross_margin_automotive"] = gm
        if overrides.gross_margin_services is not None:
            gm_s = max(0.0, min(1.0, float(overrides.gross_margin_services)))
            assumption_dict["gross_margin_services"] = gm_s
        if overrides.rd_as_percent_revenue is not None:
            assumption_dict["rd_as_percent_revenue"] = max(0.0, float(overrides.rd_as_percent_revenue))
        if overrides.sga_as_percent_revenue is not None:
            assumption_dict["sga_as_percent_revenue"] = max(0.0, float(overrides.sga_as_percent_revenue))

        assumptions = TeslaAssumptions(**assumption_dict)
        income_stmt = tesla_calculator.calculate_income_statement(assumptions, prev_data)
        updated_statements.append(income_stmt)

        # prev for next year
        prev_data = {
            "automotive_revenue": income_stmt.automotive_revenue,
            "services_revenue": income_stmt.services_revenue,
        }

    return updated_statements

def _diff_income_statements(a: Dict, b: Dict) -> Dict:
    """Compute absolute and percentage deltas for overlapping numeric fields."""
    diff: Dict[str, Dict[str, float]] = {}
    keys = set(a.keys()) & set(b.keys())
    for k in keys:
        try:
            av = float(a[k])
            bv = float(b[k])
        except Exception:
            continue
        delta = bv - av
        pct = (delta / av) * 100 if av not in [0, 0.0] else 0.0
        diff[k] = {"absolute": delta, "percent": pct}
    return diff

def _apply_cross_statement_overrides(assumption_dict: Dict, overrides: CrossStatementSimRequest) -> Dict:
    """Apply cross-statement simulation overrides to a single year's assumptions"""
    updated_dict = assumption_dict.copy()
    
    # Revenue drivers
    if overrides.automotive_revenue_growth is not None:
        updated_dict["automotive_revenue_growth"] = float(overrides.automotive_revenue_growth)
    if overrides.services_revenue_growth is not None:
        updated_dict["services_revenue_growth"] = float(overrides.services_revenue_growth)
    
    # Margin drivers
    if overrides.gross_margin_automotive is not None:
        gm = max(0.0, min(1.0, float(overrides.gross_margin_automotive)))
        updated_dict["gross_margin_automotive"] = gm
    if overrides.gross_margin_services is not None:
        gm_s = max(0.0, min(1.0, float(overrides.gross_margin_services)))
        updated_dict["gross_margin_services"] = gm_s
    
    # Opex drivers
    if overrides.rd_as_percent_revenue is not None:
        updated_dict["rd_as_percent_revenue"] = max(0.0, float(overrides.rd_as_percent_revenue))
    if overrides.sga_as_percent_revenue is not None:
        updated_dict["sga_as_percent_revenue"] = max(0.0, float(overrides.sga_as_percent_revenue))
    
    # Working capital drivers
    if overrides.days_sales_outstanding is not None:
        updated_dict["days_sales_outstanding"] = max(1, int(overrides.days_sales_outstanding))
    if overrides.days_inventory_outstanding is not None:
        updated_dict["days_inventory_outstanding"] = max(1, int(overrides.days_inventory_outstanding))
    if overrides.days_payable_outstanding is not None:
        updated_dict["days_payable_outstanding"] = max(1, int(overrides.days_payable_outstanding))
    
    # Investment drivers
    if overrides.capex_as_percent_revenue is not None:
        updated_dict["capex_as_percent_revenue"] = max(0.0, float(overrides.capex_as_percent_revenue))
    
    # Tax driver
    if overrides.tax_rate is not None:
        updated_dict["tax_rate"] = max(0.0, min(1.0, float(overrides.tax_rate)))
    
    return updated_dict

def _build_updated_all_statements(scenario_enum: ScenarioType, overrides: CrossStatementSimRequest) -> Dict:
    """Build complete financial model with cross-statement overrides applied to all years"""
    from data.tesla_data import get_tesla_assumptions
    
    # Use module-level tesla_calculator
    years_to_simulate = [overrides.year] if overrides.year else [2025, 2026, 2027, 2028, 2029]
    
    updated_income_statements = []
    updated_balance_sheets = []
    updated_cash_flow_statements = []
    
    prev_income_data = {}
    prev_balance_sheet = None
    
    for year in years_to_simulate:
        # Get original assumptions and apply overrides
        original_assumptions_dict = get_tesla_assumptions(scenario_enum, year)
        updated_assumptions_dict = _apply_cross_statement_overrides(original_assumptions_dict, overrides)
        
        assumptions = TeslaAssumptions(**updated_assumptions_dict)
        
        # Calculate Income Statement
        income_stmt = tesla_calculator.calculate_income_statement(assumptions, prev_income_data)
        updated_income_statements.append(income_stmt)
        
        # Calculate Balance Sheet (needs income statement)
        balance_sheet = tesla_calculator.calculate_balance_sheet(
            assumptions, income_stmt, prev_balance_sheet)
        updated_balance_sheets.append(balance_sheet)
        
        # Calculate Cash Flow Statement (needs both current and previous balance sheet)
        cash_flow = tesla_calculator.calculate_cash_flow_statement(
            assumptions, income_stmt, balance_sheet, prev_balance_sheet)
        
        # Update balance sheet cash with cash flow ending cash
        balance_sheet.cash_and_equivalents = cash_flow.ending_cash
        balance_sheet.total_current_assets = (
            cash_flow.ending_cash + balance_sheet.accounts_receivable + 
            balance_sheet.inventory + balance_sheet.prepaid_expenses + 
            balance_sheet.other_current_assets
        )
        balance_sheet.total_assets = (
            balance_sheet.total_current_assets + balance_sheet.total_non_current_assets
        )
        
        # Recalculate Shareholders' Equity components after cash update
        balance_sheet.retained_earnings = balance_sheet.total_assets - balance_sheet.total_liabilities - balance_sheet.common_stock - balance_sheet.other_equity
        balance_sheet.total_shareholders_equity = (
            balance_sheet.common_stock + balance_sheet.other_equity + balance_sheet.retained_earnings
        )
        
        balance_sheet.total_liab_and_equity = (
            balance_sheet.total_liabilities + balance_sheet.total_shareholders_equity
        )
        
        updated_cash_flow_statements.append(cash_flow)
        
        # Set for next iteration
        prev_income_data = {
            "automotive_revenue": income_stmt.automotive_revenue,
            "services_revenue": income_stmt.services_revenue
        }
        prev_balance_sheet = balance_sheet
    
    return {
        "income_statements": updated_income_statements,
        "balance_sheets": updated_balance_sheets,
        "cash_flow_statements": updated_cash_flow_statements
    }

def _calculate_cross_statement_deltas(original_model: FinancialModel, updated_statements: Dict) -> Dict:
    """Calculate deltas across all three statement types"""
    deltas = {}
    
    # Income Statement deltas
    income_deltas = {}
    for orig, upd in zip(original_model.income_statements, updated_statements["income_statements"]):
        income_deltas[upd.year] = _diff_income_statements(orig.dict(), upd.dict())
    deltas["income_statements"] = income_deltas
    
    # Balance Sheet deltas
    balance_deltas = {}
    for orig, upd in zip(original_model.balance_sheets, updated_statements["balance_sheets"]):
        balance_deltas[upd.year] = _diff_income_statements(orig.dict(), upd.dict())  # Reuse the same diff function
    deltas["balance_sheets"] = balance_deltas
    
    # Cash Flow deltas
    cash_flow_deltas = {}
    for orig, upd in zip(original_model.cash_flow_statements, updated_statements["cash_flow_statements"]):
        cash_flow_deltas[upd.year] = _diff_income_statements(orig.dict(), upd.dict())  # Reuse the same diff function
    deltas["cash_flow_statements"] = cash_flow_deltas
    
    return deltas

@api_router.post("/tesla/model/{scenario}/simulate-income-statement")
async def simulate_income_statement(scenario: str, req: IncomeStatementSimReq):
    """Simulate a single-year income statement with user-provided overrides.

    Returns original vs updated statements, deltas, and optional AI insights.
    """
    try:
        scenario_enum = ScenarioType(scenario.lower())

        # Use module-level tesla_calculator
        base_model = tesla_calculator.build_complete_financial_model(scenario_enum)

        if req.year is None:
            # Apply overrides to all years
            updated_all = _build_updated_income_statements_all(scenario_enum, req)
            original_list = [stmt.dict() for stmt in base_model.income_statements]
            updated_list = [stmt.dict() for stmt in updated_all]
            # Per-year deltas by index/year
            deltas_by_year: Dict[int, Dict] = {}
            for orig, upd in zip(original_list, updated_list):
                deltas_by_year[upd.get("year")] = _diff_income_statements(orig, upd)
            ai_result: Optional[Dict] = None
            if req.include_ai_insights:
                try:
                    insight_obj = income_statement_insights_agent.generate_comparison_summary(original_list, updated_list, scenario)
                    ai_result = insight_obj
                except Exception as e:
                    ai_result = {"error": f"AI analysis unavailable: {e}"}

            return {
                "success": True,
                "scenario": scenario,
                "original_income_statements": original_list,
                "updated_income_statements": updated_list,
                "deltas_by_year": deltas_by_year,
                "ai_insights": ai_result,
            }
        else:
            # Original model → one year
            original: Optional[IncomeStatement] = None
            for stmt in base_model.income_statements:
                if stmt.year == req.year:
                    original = stmt
                    break
            if original is None:
                raise HTTPException(status_code=400, detail=f"Year {req.year} not available in standard model")

            # Updated statement with overrides for a single year
            updated_stmt = _build_updated_income_statement(scenario_enum, req.year, req)
            if updated_stmt is None:
                raise HTTPException(status_code=500, detail="Failed to build updated income statement")

            original_dict = original.dict()
            updated_dict = updated_stmt.dict()
            deltas = _diff_income_statements(original_dict, updated_dict)

        ai_result: Optional[Dict] = None
        if req.include_ai_insights:
            try:
                # Reuse proactive insights agent with a minimal context (two statements)
                model_like = {
                    "income_statements": [original_dict, updated_dict],
                }
                ai_result = proactive_insights_agent.analyze_financial_model(model_like, scenario)
            except Exception as e:
                ai_result = {"error": f"AI analysis unavailable: {e}"}

        return {
            "success": True,
            "scenario": scenario,
            "year": req.year,
            "original_income_statement": original_dict,
            "updated_income_statement": updated_dict,
            "deltas": deltas,
            "ai_insights": ai_result,
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario. Use 'best', 'base', or 'worst'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error simulating income statement: {str(e)}")

@api_router.post("/tesla/model/{scenario}/simulate-all-statements")
async def simulate_all_statements(scenario: str, req: CrossStatementSimRequest):
    """Simulate all three financial statements (Income, Balance Sheet, Cash Flow) with user-provided overrides.
    
    This endpoint allows comprehensive financial modeling by adjusting key drivers that affect
    all three financial statements, including revenue growth, margins, working capital, CapEx, and tax rates.
    
    Returns original vs updated statements, deltas across all statements, and optional AI insights.
    """
    try:
        scenario_enum = ScenarioType(scenario.lower())
        
        # Use module-level tesla_calculator
        # Build original complete financial model
        original_model = tesla_calculator.build_complete_financial_model(scenario_enum)
        
        # Build updated model with overrides
        updated_statements = _build_updated_all_statements(scenario_enum, req)
        
        # Calculate deltas across all statement types
        deltas = _calculate_cross_statement_deltas(original_model, updated_statements)
        
        # Generate AI insights if requested
        ai_insights = None
        if req.include_ai_insights:
            try:
                # Use the cross-statement insights agent for comprehensive analysis
                original_statements_dict = {
                    "income_statements": [stmt.dict() for stmt in original_model.income_statements],
                    "balance_sheets": [stmt.dict() for stmt in original_model.balance_sheets],
                    "cash_flow_statements": [stmt.dict() for stmt in original_model.cash_flow_statements]
                }
                updated_statements_dict = {
                    "income_statements": [stmt.dict() for stmt in updated_statements["income_statements"]],
                    "balance_sheets": [stmt.dict() for stmt in updated_statements["balance_sheets"]],
                    "cash_flow_statements": [stmt.dict() for stmt in updated_statements["cash_flow_statements"]]
                }
                ai_insights = cross_statement_insights_agent.analyze_cross_statement_simulation(
                    original_statements_dict, updated_statements_dict, scenario
                )
            except Exception as e:
                ai_insights = {"error": f"AI analysis unavailable: {str(e)}"}
        
        return {
            "success": True,
            "scenario": scenario,
            "original_statements": {
                "income_statements": [stmt.dict() for stmt in original_model.income_statements],
                "balance_sheets": [stmt.dict() for stmt in original_model.balance_sheets],
                "cash_flow_statements": [stmt.dict() for stmt in original_model.cash_flow_statements]
            },
            "updated_statements": {
                "income_statements": [stmt.dict() for stmt in updated_statements["income_statements"]],
                "balance_sheets": [stmt.dict() for stmt in updated_statements["balance_sheets"]],
                "cash_flow_statements": [stmt.dict() for stmt in updated_statements["cash_flow_statements"]]
            },
            "deltas": deltas,
            "ai_insights": ai_insights,
            "key_metrics": {
                "revenue_cagr": _calculate_revenue_cagr(updated_statements["income_statements"]),
                "free_cash_flow_margin": _calculate_fcf_margin(updated_statements["income_statements"], updated_statements["cash_flow_statements"]),
                "working_capital_days": _calculate_working_capital_days(updated_statements["balance_sheets"][-1]),
                "roic": _calculate_roic(updated_statements["income_statements"][-1], updated_statements["balance_sheets"][-1])
            }
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario. Use 'best', 'base', or 'worst'")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error simulating all statements: {str(e)}")

def _calculate_revenue_cagr(income_statements: List) -> float:
    """Calculate revenue CAGR from income statements"""
    if len(income_statements) < 2:
        return 0.0
    
    first_year_revenue = income_statements[0].total_revenue
    last_year_revenue = income_statements[-1].total_revenue
    years = len(income_statements) - 1
    
    if first_year_revenue <= 0:
        return 0.0
    
    cagr = ((last_year_revenue / first_year_revenue) ** (1/years)) - 1
    return round(cagr * 100, 2)  # Return as percentage

def _calculate_fcf_margin(income_statements: List, cash_flow_statements: List) -> float:
    """Calculate average free cash flow margin"""
    if not income_statements or not cash_flow_statements:
        return 0.0
    
    # Calculate FCF = Operating Cash Flow + Capital Expenditures (CapEx is negative)
    total_fcf = sum(cf.operating_cash_flow + cf.capital_expenditures for cf in cash_flow_statements)
    total_revenue = sum(inc.total_revenue for inc in income_statements)
    
    if total_revenue <= 0:
        return 0.0
    
    return round((total_fcf / total_revenue) * 100, 2)  # Return as percentage

def _calculate_working_capital_days(balance_sheet) -> float:
    """Calculate cash conversion cycle (DIO + DSO - DPO)"""
    try:
        # This is simplified - in reality you'd calculate from actual balance sheet items
        # For now, return a representative number
        return 45.0  # days
    except:
        return 0.0

def _calculate_roic(income_statement, balance_sheet) -> float:
    """Calculate Return on Invested Capital"""
    try:
        nopat = income_statement.operating_income * (1 - income_statement.effective_tax_rate)
        invested_capital = balance_sheet.total_shareholders_equity + balance_sheet.long_term_debt
        
        if invested_capital <= 0:
            return 0.0
        
        roic = (nopat / invested_capital) * 100
        return round(roic, 2)  # Return as percentage
    except:
        return 0.0

@api_router.get("/tesla/comparison")
async def get_scenario_comparison():
    """Compare all three scenarios side by side"""
    try:
        # Use module-level tesla_calculator
        models = {}
        for scenario in ["best", "base", "worst"]:
            scenario_enum = ScenarioType(scenario)
            model = tesla_calculator.build_complete_financial_model(scenario_enum)
            models[scenario] = model.dict()
        
        # Create comparison summary
        comparison = {
            "revenue_comparison": {},
            "valuation_comparison": {},
            "margin_comparison": {}
        }
        
        for scenario, model in models.items():
            # 2029 projections
            final_income = model["income_statements"][-1]
            dcf = model["dcf_valuation"]
            
            comparison["revenue_comparison"][scenario] = {
                "2029_revenue": final_income["total_revenue"],
                "5yr_cagr": ((final_income["total_revenue"] / TESLA_BASE_YEAR_DATA["total_revenue"]) ** (1/5)) - 1
            }
            
            comparison["valuation_comparison"][scenario] = {
                "price_per_share": dcf["price_per_share"],
                "enterprise_value": dcf["enterprise_value"],
                "wacc": dcf["wacc"]
            }
            
            comparison["margin_comparison"][scenario] = {
                "2029_gross_margin": final_income["gross_margin"],
                "2029_operating_margin": final_income["operating_margin"],
                "2029_net_margin": final_income["net_margin"]
            }
        
        return {
            "success": True,
            "models": models,
            "comparison_summary": comparison
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/tesla/update-assumption")
async def update_assumption(update: ModelInput):
    """Update a specific assumption and recalculate model"""
    try:
        # This would be more complex in a real system with persistent storage
        # For now, we'll return the instruction on how this would work
        return {
            "success": True,
            "message": f"Would update {update.field_name} to {update.field_value} for {update.scenario} {update.year}",
            "note": "Real-time updates would require storing model state and recalculating affected statements"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/tesla/sensitivity/{scenario}")
async def get_sensitivity_analysis(scenario: str):
    """Get detailed sensitivity analysis for DCF valuation"""
    try:
        scenario_enum = ScenarioType(scenario.lower())
        # Use module-level tesla_calculator
        model = tesla_calculator.build_complete_financial_model(scenario_enum)
        
        dcf = model.dcf_valuation
        
        return {
            "scenario": scenario,
            "base_valuation": dcf.price_per_share,
            "sensitivity_analysis": {
                "growth_rates": dcf.sensitivity_growth_rates,
                "wacc_rates": dcf.sensitivity_wacc_rates,
                "price_matrix": dcf.sensitivity_matrix
            },
            "key_assumptions": {
                "terminal_growth_rate": dcf.terminal_growth_rate,
                "wacc": dcf.wacc,
                "final_year_fcf": dcf.projected_free_cash_flows[-1]
            }
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid scenario")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Original status check endpoints (keeping for compatibility)
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

@api_router.get("/")
async def root():
    return {"message": "Tesla Financial Model & Analytics API - Ready for Analysis"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    if db is not None:
        try:
            await db.status_checks.insert_one(status_obj.dict())
        except Exception as e:
            print(f"Database insert error: {e}")
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    if db is not None:
        try:
            status_checks = await db.status_checks.find().to_list(1000)
            return [StatusCheck(**status_check) for status_check in status_checks]
        except Exception as e:
            print(f"Database query error: {e}")
    return []

# Metric Forecasting Endpoints
@api_router.post("/forecast/univariate")
async def univariate_forecast(request: UnivariateForecastRequest):
    """Generate univariate forecast for a single metric"""
    try:
        # Ensure data is loaded
        if not metric_forecasting_service.load_data():
            raise HTTPException(status_code=500, detail="Failed to load forecasting data")
        
        result = metric_forecasting_service.univariate_forecast(
            metric=request.metric,
            horizon=request.horizon,
            test_months=request.test_months
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Univariate forecast error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating univariate forecast: {str(e)}")

@api_router.post("/forecast/multivariate")
async def multivariate_forecast(request: MultivariateForecastRequest):
    """Generate multivariate forecast for multiple metrics"""
    try:
        # Ensure data is loaded
        if not metric_forecasting_service.load_data():
            raise HTTPException(status_code=500, detail="Failed to load forecasting data")
        
        result = metric_forecasting_service.multivariate_forecast(
            metrics=request.metrics,
            horizon=request.horizon,
            test_months=request.test_months
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Multivariate forecast error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating multivariate forecast: {str(e)}")

@api_router.get("/forecast/metrics")
async def get_available_metrics():
    """Get list of available metrics for forecasting"""
    return {
        "success": True,
        "metrics": [
            {"id": "revenue_millions", "name": "Revenue", "unit": "millions USD", "description": "Total revenue from all sources"},
            {"id": "cogs_millions", "name": "Cost of Goods Sold", "unit": "millions USD", "description": "Direct costs of producing goods"},
            {"id": "rd_expense_millions", "name": "R&D Expense", "unit": "millions USD", "description": "Research and development expenses"},
            {"id": "sga_expense_millions", "name": "SG&A Expense", "unit": "millions USD", "description": "Selling, general & administrative expenses"}
        ]
    }

@api_router.get("/forecast/macro-indicators")
async def get_macro_indicators():
    """Get available macroeconomic indicators"""
    return {
        "success": True,
        "indicators": [
            {"id": "gdp_growth", "name": "GDP Growth", "unit": "percentage", "description": "Gross domestic product growth rate"},
            {"id": "inflation_rate", "name": "Inflation Rate", "unit": "percentage", "description": "Consumer price inflation rate"},
            {"id": "interest_rate", "name": "Interest Rate", "unit": "percentage", "description": "Federal funds rate"},
            {"id": "unemployment_rate", "name": "Unemployment Rate", "unit": "percentage", "description": "Unemployment rate"},
            {"id": "consumer_confidence", "name": "Consumer Confidence", "unit": "index", "description": "Consumer confidence index"},
            {"id": "oil_price", "name": "Oil Price", "unit": "USD per barrel", "description": "Crude oil price"},
            {"id": "ev_market_size", "name": "EV Market Size", "unit": "billions USD", "description": "Electric vehicle market size"}
        ]
    }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    if client:
        client.close()

# ---------------- Vehicle Forecast API (Isolated) ----------------
from services.vehicle_forecast_engine import VehicleForecastEngine

vehicle_engine = VehicleForecastEngine()

# Extend vehicle forecast schemas for optional revenue and elasticity
class VehicleForecastReq(BaseModel):
    model_key: str
    forecast_type: str = "univariate"  # 'univariate' | 'multivariate'
    months_ahead: int = 12
    include_revenue: bool = False
    asp_per_unit: Optional[float] = None  # if provided, constant ASP used for revenue
    price_change: Optional[float] = None  # e.g., -0.05 for -5% price
    elasticity: Optional[float] = None    # e.g., -1.0; deliveries *= (1 + elasticity*price_change)
    test_window: Optional[int] = None     # 3 or 6 to control test months

# Dedicated router to ensure registration order for vehicle endpoints
from fastapi import APIRouter as _APIRouter
vehicles_router = _APIRouter(prefix="/api/vehicles")

@vehicles_router.get("/models")
async def list_vehicle_models_v2():
    try:
        models = vehicle_engine.get_available_models()
        if not models:
            try:
                from data.tesla_enhanced_data import VEHICLE_MODEL_DATA
                fallback = []
                for mk, md in VEHICLE_MODEL_DATA.get("models", {}).items():
                    fallback.append({
                        "model_key": mk,
                        "display_name": md.get("name", mk.replace("_"," ").title()),
                        "first_date": None,
                        "last_date": None,
                        "records": 0,
                    })
                return {"success": True, "models": fallback, "note": "Excel loader returned no models; using static fallback."}
            except Exception:
                pass
        return {"success": True, "models": [m.__dict__ for m in models]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== AGENT-BASED AUTONOMOUS FORECASTING ==========
@vehicles_router.get("/forecast/cached/{model_key}")
async def get_cached_forecast(model_key: str):
    """
    Get pre-computed forecast for a model from the autonomous agent
    Returns both univariate and multivariate forecasts
    """
    try:
        from agents.vehicle_forecast_agent import get_vehicle_forecast_agent
        agent = get_vehicle_forecast_agent()
        
        if agent is None:
            raise HTTPException(status_code=503, detail="Vehicle forecast agent not initialized")
        
        forecast = agent.get_forecast(model_key)
        
        if forecast is None:
            # Agent hasn't generated forecast yet or it's expired
            raise HTTPException(
                status_code=404, 
                detail=f"No cached forecast available for {model_key}. Agent may still be processing."
            )
        
        return {
            "success": True,
            "model_key": forecast.model_key,
            "model_name": forecast.model_name,
            "forecast_date": forecast.forecast_date.isoformat(),
            "age_hours": (datetime.now() - forecast.forecast_date).total_seconds() / 3600,
            "univariate": forecast.univariate,
            "multivariate": forecast.multivariate,
            "metadata": forecast.metadata
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@vehicles_router.get("/forecast/cached")
async def get_all_cached_forecasts():
    """
    Get all pre-computed forecasts from the autonomous agent
    """
    try:
        from agents.vehicle_forecast_agent import get_vehicle_forecast_agent
        agent = get_vehicle_forecast_agent()
        
        if agent is None:
            raise HTTPException(status_code=503, detail="Vehicle forecast agent not initialized")
        
        all_forecasts = agent.get_all_forecasts()
        
        return {
            "success": True,
            "count": len(all_forecasts),
            "forecasts": {
                key: {
                    "model_key": forecast.model_key,
                    "model_name": forecast.model_name,
                    "forecast_date": forecast.forecast_date.isoformat(),
                    "age_hours": (datetime.now() - forecast.forecast_date).total_seconds() / 3600,
                    "has_univariate": forecast.univariate is not None,
                    "has_multivariate": forecast.multivariate is not None
                }
                for key, forecast in all_forecasts.items()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@vehicles_router.post("/forecast/refresh")
async def refresh_forecasts(model_key: Optional[str] = None):
    """
    Trigger manual refresh of forecasts
    If model_key provided, refresh only that model
    Otherwise, refresh all models
    """
    try:
        from agents.vehicle_forecast_agent import get_vehicle_forecast_agent
        agent = get_vehicle_forecast_agent()
        
        if agent is None:
            raise HTTPException(status_code=503, detail="Vehicle forecast agent not initialized")
        
        if agent.is_running:
            return {
                "success": False,
                "message": "Forecast generation already in progress",
                "status": "running"
            }
        
        # Trigger refresh in background
        asyncio.create_task(agent.generate_all_forecasts())
        
        return {
            "success": True,
            "message": f"Forecast refresh triggered {'for ' + model_key if model_key else 'for all models'}",
            "status": "started"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@vehicles_router.get("/forecast/agent-status")
async def get_agent_status():
    """
    Get status of the autonomous forecast agent
    """
    try:
        from agents.vehicle_forecast_agent import get_vehicle_forecast_agent
        agent = get_vehicle_forecast_agent()
        
        if agent is None:
            return {
                "success": False,
                "error": "Vehicle forecast agent not initialized"
            }
        
        status = agent.get_status()
        return {
            "success": True,
            "agent_status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== SIMULATION ENDPOINT (for ASP/Elasticity) ==========
class VehicleForecastSimulationReq(BaseModel):
    """Request for simulating forecast with ASP and elasticity changes"""
    model_key: str
    price_change: Optional[float] = Field(None, description="Price change as decimal (e.g., -0.05 for -5%)")
    elasticity: Optional[float] = Field(None, description="Price elasticity (e.g., -1.0)")
    asp_per_unit: Optional[float] = Field(None, description="Override ASP per unit for revenue calculation")

@vehicles_router.post("/forecast/simulate")
async def simulate_forecast(req: VehicleForecastSimulationReq):
    """
    Apply ASP and elasticity adjustments to pre-computed forecast
    This allows "what-if" analysis without regenerating the base forecast
    """
    try:
        from agents.vehicle_forecast_agent import get_vehicle_forecast_agent
        import copy
        
        agent = get_vehicle_forecast_agent()
        
        if agent is None:
            raise HTTPException(status_code=503, detail="Vehicle forecast agent not initialized")
        
        # Get cached forecast
        forecast = agent.get_forecast(req.model_key)
        
        if forecast is None:
            raise HTTPException(
                status_code=404,
                detail=f"No cached forecast available for {req.model_key}"
            )
        
        # Deep copy to avoid modifying cache
        univariate = copy.deepcopy(forecast.univariate)
        multivariate = copy.deepcopy(forecast.multivariate)
        
        # Apply elasticity adjustment if provided
        if req.price_change is not None and req.elasticity is not None:
            adj_multiplier = 1.0 + (req.elasticity * req.price_change)
            
            # Apply to all targets in forecasts_by_target
            for target, target_data in univariate.get("forecasts_by_target", {}).items():
                for f in target_data.get("forecasts", []):
                    base = f.get("forecast", 0)
                    f["forecast"] = max(0.0, float(base) * adj_multiplier)
                    f["adjusted"] = True
            
            for target, target_data in multivariate.get("forecasts_by_target", {}).items():
                for f in target_data.get("forecasts", []):
                    base = f.get("forecast", 0)
                    f["forecast"] = max(0.0, float(base) * adj_multiplier)
                    f["adjusted"] = True
        
        # Apply ASP for revenue calculation if requested
        if req.asp_per_unit is not None:
            asp = float(req.asp_per_unit)
            monthly_decay = 0.02 / 12.0
            
            # Apply to all targets in forecasts_by_target
            for target, target_data in univariate.get("forecasts_by_target", {}).items():
                asp_cur = asp
                for idx, f in enumerate(target_data.get("forecasts", []), start=1):
                    if idx > 1:
                        asp_cur *= (1.0 - monthly_decay)
                    revenue = float(f["forecast"]) * asp_cur
                    f["asp"] = asp_cur
                    f["revenue"] = revenue
            
            for target, target_data in multivariate.get("forecasts_by_target", {}).items():
                asp_cur = asp
                for idx, f in enumerate(target_data.get("forecasts", []), start=1):
                    if idx > 1:
                        asp_cur *= (1.0 - monthly_decay)
                    revenue = float(f["forecast"]) * asp_cur
                    f["asp"] = asp_cur
                    f["revenue"] = revenue
        
        return {
            "success": True,
            "model_key": req.model_key,
            "model_name": forecast.model_name,
            "simulation_applied": {
                "price_change": req.price_change,
                "elasticity": req.elasticity,
                "asp_per_unit": req.asp_per_unit
            },
            "univariate": univariate,
            "multivariate": multivariate
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== LEGACY ON-DEMAND FORECASTING (kept for compatibility) ==========
@vehicles_router.post("/forecast")
async def vehicle_forecast_v2(req: VehicleForecastReq):
    """
    LEGACY: On-demand forecast generation
    For new implementations, use /forecast/cached/{model_key} instead
    """
    try:
        if req.months_ahead < 1 or req.months_ahead > 24:
            raise HTTPException(status_code=400, detail="months_ahead must be between 1 and 24")
        if req.forecast_type not in ["univariate", "multivariate"]:
            raise HTTPException(status_code=400, detail="forecast_type must be 'univariate' or 'multivariate'")
        result = vehicle_engine.generate_forecast(req.model_key, req.forecast_type, req.months_ahead, req.test_window)
        if result is None:
            raise HTTPException(status_code=400, detail="Insufficient data or unknown model_key")

        # Optional elasticity adjustment
        if req.price_change is not None and req.elasticity is not None:
            for f in result.get("forecasts", []):
                base = f.get("forecast", 0)
                adj_multiplier = 1.0 + (req.elasticity * req.price_change)
                f["forecast"] = max(0.0, float(base) * adj_multiplier)
            result["elasticity_applied"] = True
            result["price_change"] = req.price_change
            result["elasticity"] = req.elasticity

        # Optional revenue computation
        if req.include_revenue:
            asp = req.asp_per_unit
            if asp is None:
                try:
                    from data.tesla_enhanced_data import PRICE_VOLUME_MIX_DATA
                    model_trend = PRICE_VOLUME_MIX_DATA.get("historical_asp_trends", {}).get(req.model_key)
                    if model_trend:
                        last_year = max(model_trend.keys())
                        asp = float(model_trend[last_year])
                    else:
                        asp = 50000.0
                except Exception:
                    asp = 50000.0
            monthly_decay = 0.02 / 12.0
            asp_cur = float(asp)
            revenue_series = []
            for idx, f in enumerate(result.get("forecasts", []), start=1):
                if idx > 1:
                    asp_cur *= (1.0 - monthly_decay)
                revenue = float(f["forecast"]) * asp_cur
                f["asp"] = asp_cur
                f["revenue"] = revenue
                revenue_series.append(revenue)
            result["revenue_total"] = float(sum(revenue_series))
            result["asp_start"] = float(asp)
            result["asp_monthly_decay"] = monthly_decay

        return {"success": True, "forecast": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@vehicles_router.post("/compare")
async def vehicle_compare_v2(req: VehicleForecastReq):
    try:
        if req.months_ahead < 1 or req.months_ahead > 24:
            raise HTTPException(status_code=400, detail="months_ahead must be between 1 and 24")
        result = vehicle_engine.compare_methods(req.model_key, req.months_ahead)
        if result is None:
            raise HTTPException(status_code=400, detail="Insufficient data or unknown model_key")
        return {"success": True, "comparison": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Econ inspect endpoint for vehicle forecasts
@api_router.get("/vehicles/econ-columns")
async def vehicle_econ_columns():
    try:
        cols = getattr(vehicle_engine, "_econ_columns", [])
        return {"success": True, "econ_columns": cols}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Additional Excel-backed vehicle data endpoints
@vehicles_router.get("/years")
async def vehicle_years():
    try:
        if vehicle_engine.monthly_data is None:
            if not vehicle_engine.load_data():
                raise HTTPException(status_code=500, detail="Failed to load vehicle data. Check VEHICLE_DATA_XLSX path.")
        df = vehicle_engine.monthly_data.copy()
        years = sorted(df["DATE"].dt.year.unique().tolist())
        return {"success": True, "years": years}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@vehicles_router.get("/summary/{year}")
async def vehicle_year_summary(year: int):
    try:
        if vehicle_engine.monthly_data is None:
            if not vehicle_engine.load_data():
                raise HTTPException(status_code=500, detail="Failed to load vehicle data. Check VEHICLE_DATA_XLSX path.")
        df = vehicle_engine.monthly_data.copy()
        df_year = df[df["DATE"].dt.year == year]
        if df_year.empty:
            return {"success": True, "summary": {"totals": {"total_deliveries": 0, "total_production": 0}, "models": []}}
        # Per model aggregates
        agg_map = {"deliveries": "sum", "production": "sum"}
        if "sold" in df_year.columns:
            agg_map["sold"] = "sum"
        if "revenue" in df_year.columns:
            agg_map["revenue"] = "sum"
        grp = df_year.groupby("model_key", as_index=False).agg(agg_map)
        # Compute average ASP per model for the year if asp column exists
        asp_map = {}
        if "asp" in df_year.columns:
            asp_grp = df_year.groupby("model_key")["asp"].mean(numeric_only=True)
            asp_map = {str(k): float(v) for k, v in asp_grp.to_dict().items()}
        models = []
        for _, row in grp.iterrows():
            model_item = {
                "model_key": str(row["model_key"]),
                "deliveries": float(row.get("deliveries", 0) or 0),
                "production": float(row.get("production", 0) or 0),
                **({"sold": float(row.get("sold", 0) or 0)} if "sold" in df_year.columns else {}),
                **({"revenue": float(row.get("revenue", 0) or 0)} if "revenue" in df_year.columns else {})
            }
            if asp_map:
                model_item["asp"] = asp_map.get(str(row["model_key"]), None)
            models.append(model_item)
        totals = {
            "total_deliveries": float(df_year["deliveries"].sum()),
            "total_production": float(df_year["production"].sum()),
            **({"total_sold": float(df_year["sold"].sum())} if "sold" in df_year.columns else {}),
            **({"total_revenue": float(df_year["revenue"].sum())} if "revenue" in df_year.columns else {}),
        }
        return {"success": True, "summary": {"totals": totals, "models": models}}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include the vehicles router after definitions to ensure route registration
app.include_router(vehicles_router)

# ============================================================================
# Energy & Services API Routes
# ============================================================================

energy_services_router = APIRouter(prefix="/api/energy-services", tags=["Energy & Services"])

# Initialize Energy Services Loader
energy_services_loader = get_energy_services_loader()

@energy_services_router.get("/summary/{year}")
async def energy_services_year_summary(year: int):
    """
    Get aggregated Energy & Services summary for a specific year
    
    Returns:
        - Energy: Revenue, COGS, Margin, YoY Growth, CAGR
        - Services: Revenue, COGS, Margin, YoY Growth, CAGR
    """
    try:
        summary = energy_services_loader.get_summary_for_year(year)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@energy_services_router.get("/monthly/{year}")
async def energy_services_monthly_data(year: int):
    """
    Get month-by-month breakdown for Energy & Services for a specific year
    """
    try:
        monthly = energy_services_loader.get_monthly_data(year)
        return monthly
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@energy_services_router.get("/available-years")
async def energy_services_available_years():
    """Get list of years with available Energy & Services data"""
    try:
        years = energy_services_loader.get_available_years()
        return {"success": True, "years": years}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Include the energy services router
app.include_router(energy_services_router)
