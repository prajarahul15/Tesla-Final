"""
AI Agents for Tesla Financial Model with OpenAI Integration
Implements proactive insights, forecasting, and interactive analysis
"""

import pandas as pd
import numpy as np
from openai import OpenAI
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os
from models.financial_models import ScenarioType, TeslaAssumptions
from data.tesla_enhanced_data import get_enhanced_tesla_drivers, TESLA_HISTORICAL_DATA, VEHICLE_MODEL_DATA
# Do not load .env here to avoid encoding issues and duplicate loads.
# server.py is responsible for loading environment variables early.

def _get_client() -> Optional[OpenAI]:
    """Return an OpenAI v1 client if API key is configured, else None."""
    if not os.getenv('OPENAI_API_KEY'):
        return None
    # Add a network timeout so UI doesn't hang indefinitely
    try:
        return OpenAI(timeout=60.0)  # Increased timeout for complex requests
    except Exception:
        return OpenAI()

def _chat_completion(messages: List[Dict[str, str]], *, temperature: float = 0.7, max_tokens: int = 1000):
    """OpenAI v1 chat completion with graceful model fallback.

    Tries, in order: OPENAI_MODEL env → gpt-4o-mini → gpt-4o → gpt-4-turbo → gpt-3.5-turbo
    """
    client = _get_client()
    if client is None:
        print("❌ OpenAI client is None - API key not configured")
        raise RuntimeError("OPENAI_API_KEY not configured")

    model_candidates: List[str] = []
    env_model = os.getenv('OPENAI_MODEL')
    if env_model:
        model_candidates.append(env_model)
    model_candidates += ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]

    print(f"🔍 Trying models: {model_candidates}")
    last_error: Optional[Exception] = None
    for model_name in model_candidates:
        try:
            print(f"📞 Calling OpenAI with model: {model_name}")
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            print(f"✅ Success with model: {model_name}")
            return response
        except Exception as e:
            print(f"❌ Model {model_name} failed: {e}")
            last_error = e
            continue
    print(f"❌ All models failed. Last error: {last_error}")
    raise last_error or RuntimeError("All chat completion model candidates failed")
class ProactiveInsightsAgent:
    """AI Agent that provides proactive insights using OpenAI GPT"""
    
    def __init__(self):
        self.insights_cache = {}
        
    def analyze_financial_model(self, model_data: Dict, scenario: str) -> Dict:
        """Generate proactive insights from financial model data using OpenAI"""
        
        try:
            # Check if OpenAI API key is available
            if not os.getenv('OPENAI_API_KEY'):
                return self._get_fallback_insights(model_data, scenario)
            
            # Prepare context for OpenAI
            context = self._prepare_financial_context(model_data, scenario)
            
            # Create structured prompt for insights
            prompt = self._create_insights_prompt(context, scenario)
            
            # Call OpenAI API (with fallback models)
            response = _chat_completion(
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a senior financial analyst specializing in Tesla and the EV industry. "
                            "Provide structured, actionable insights grounded strictly in the given data."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=2000,
            )
            
            # Parse OpenAI response
            insights_text = response.choices[0].message.content
            insights = self._parse_insights_response(insights_text)
            
            return insights
            
        except Exception as e:
            import traceback
            print(f"❌ Error generating insights: {e}")
            print(f"📍 Traceback:\n{traceback.format_exc()}")
            return self._get_fallback_insights(model_data, scenario)
    
    def _prepare_financial_context(self, model_data: Dict, scenario: str) -> Dict:
        """Prepare financial context for OpenAI analysis"""
        
        if not model_data or "income_statements" not in model_data:
            return {}
            
        income_statements = model_data["income_statements"]
        
        # Calculate key metrics
        context = {
            "scenario": scenario,
            "years": [2025, 2026, 2027, 2028, 2029],
            "revenue_data": [],
            "margin_data": [],
            "growth_metrics": {},
            "vehicle_data": {},
            "cash_flow_data": []
        }
        
        # Revenue analysis
        for i, stmt in enumerate(income_statements):
            context["revenue_data"].append({
                "year": 2025 + i,
                "total_revenue": stmt["total_revenue"],
                "automotive_revenue": stmt["automotive_revenue"],
                "services_revenue": stmt["services_revenue"],
                "gross_margin": stmt["gross_margin"],
                "operating_margin": stmt["operating_margin"],
                "net_margin": stmt["net_margin"]
            })
        
        # Growth calculations
        if len(income_statements) >= 2:
            first_year = income_statements[0]
            last_year = income_statements[-1]
            
            context["growth_metrics"] = {
                "revenue_cagr": ((last_year["total_revenue"] / first_year["total_revenue"]) ** (1/4)) - 1,
                "margin_improvement": last_year["gross_margin"] - first_year["gross_margin"],
                "final_revenue": last_year["total_revenue"],
                "final_margin": last_year["gross_margin"]
            }
        
        # Vehicle model data (if available)
        if "revenue_breakdown" in income_statements[-1]:
            vehicle_breakdown = income_statements[-1]["revenue_breakdown"]
            if "automotive_revenue_by_model" in vehicle_breakdown:
                context["vehicle_data"] = vehicle_breakdown["automotive_revenue_by_model"]
        
        # Cash flow data
        if "cash_flow_statements" in model_data:
            for cf in model_data["cash_flow_statements"]:
                context["cash_flow_data"].append({
                    "free_cash_flow": cf.get("free_cash_flow", 0),
                    "operating_cash_flow": cf.get("operating_cash_flow", 0),
                    "capital_expenditures": cf.get("capital_expenditures", 0)
                })
        
        return context
    
    def _create_insights_prompt(self, context: Dict, scenario: str) -> str:
        """Create structured prompt for OpenAI insights generation"""
        
        prompt = f"""
        Analyze the following Tesla financial model data for the {scenario.upper()} scenario and provide structured insights.
        
        FINANCIAL DATA:
        {json.dumps(context, indent=2)}
        
        Please provide insights in the following JSON format:
        {{
            "key_insights": [
                {{
                    "type": "growth|risk|opportunity|financial_strength|operational_efficiency|product_mix|competitive_pressure|growth_concern",
                    "title": "Brief title",
                    "description": "Detailed description with specific numbers and context",
                    "impact": "positive|negative|neutral",
                    "confidence": 0.0-1.0
                }}
            ],
            "risk_alerts": [
                {{
                    "type": "risk_type",
                    "title": "Risk title",
                    "description": "Risk description with mitigation suggestions",
                    "impact": "negative",
                    "confidence": 0.0-1.0
                }}
            ],
            "opportunities": [
                {{
                    "type": "opportunity_type",
                    "title": "Opportunity title", 
                    "description": "Opportunity description with implementation suggestions",
                    "impact": "positive",
                    "confidence": 0.0-1.0
                }}
            ],
            "recommendations": [
                {{
                    "category": "operational|strategic|financial|market",
                    "title": "Recommendation title",
                    "description": "Detailed recommendation",
                    "priority": "high|medium|low",
                    "timeline": "immediate|short-term|medium-term|long-term"
                }}
            ],
            "market_context": [
                {{
                    "factor": "Factor name",
                    "description": "Market factor description",
                    "relevance": "high|medium|low"
                }}
            ]
        }}
        
        Focus on:
        1. Revenue growth trajectory and sustainability
        2. Margin expansion opportunities and risks
        3. Vehicle model mix optimization
        4. Cash flow generation and capital allocation
        5. Competitive positioning and market dynamics
        6. Operational efficiency improvements
        7. Strategic recommendations for the {scenario} scenario
        
        Provide specific, actionable insights with concrete numbers and clear reasoning.
        """
        
        return prompt
    
    def _parse_insights_response(self, response_text: str) -> Dict:
        """Parse OpenAI response into structured insights"""
        
        try:
            print(f"📝 Raw response from OpenAI (first 500 chars): {response_text[:500]}")
            
            # Remove markdown code fences if present
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]  # Remove ```json
            elif cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]  # Remove ```
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]  # Remove trailing ```
            cleaned_text = cleaned_text.strip()
            
            # Try to extract JSON from response
            start_idx = cleaned_text.find('{')
            end_idx = cleaned_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = cleaned_text[start_idx:end_idx]
                print(f"🔍 Extracted JSON (first 300 chars): {json_str[:300]}")
                parsed = json.loads(json_str)
                print(f"✅ Successfully parsed JSON with {len(parsed.get('key_insights', []))} insights")
                return parsed
            else:
                print(f"❌ No JSON found in response")
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            import traceback
            print(f"❌ Error parsing insights response: {e}")
            print(f"📍 Traceback:\n{traceback.format_exc()}")
            return self._get_fallback_insights({}, "base")
    
    def _get_fallback_insights(self, model_data: Dict, scenario: str) -> Dict:
        """Fallback insights when OpenAI fails"""
        
        return {
            "key_insights": [
                {
                    "type": "system_status",
                    "title": "AI Analysis Unavailable",
                    "description": "Unable to generate AI insights at this time. Please check OpenAI API key configuration.",
                    "impact": "neutral",
                    "confidence": 0.5
                }
            ],
            "risk_alerts": [],
            "opportunities": [],
            "recommendations": [],
            "market_context": []
        }

class ProphetForecastingAgent:
    """AI Agent for Prophet-based forecasting with OpenAI insights"""
    
    def __init__(self):
        self.models = {}
        
    def generate_prophet_forecast(self, historical_data: List[Dict], periods: int = 12) -> Dict:
        """Generate Prophet forecast with AI insights"""
        
        try:
            # Simulate Prophet forecasting (in real implementation, use actual Prophet)
            forecast_data = self._simulate_prophet_forecast(historical_data, periods)
            
            # Generate AI insights for the forecast
            insights = self._generate_forecast_insights_with_ai(forecast_data, historical_data)
            
            return {
                "forecasts": forecast_data,
                "model_info": {
                    "type": "prophet_simulation",
                    "periods": periods
                },
                "insights": insights
            }
            
        except Exception as e:
            return {"error": f"Forecast generation failed: {str(e)}"}
    
    def _simulate_prophet_forecast(self, historical_data: List[Dict], periods: int) -> List[Dict]:
        """Simulate Prophet forecasting"""
        
        forecast_data = []
        
        if not historical_data:
            return forecast_data
        
        # Get last known value and trend
        last_value = historical_data[-1]["value"]
        
        # Simple trend calculation
        if len(historical_data) >= 2:
            trend = (historical_data[-1]["value"] - historical_data[0]["value"]) / len(historical_data)
        else:
            trend = 0
        
        # Generate forecasts with seasonality and trend
        base_date = datetime.strptime(historical_data[-1]["date"], "%Y-%m-%d")
        
        for i in range(1, periods + 1):
            forecast_date = base_date + timedelta(days=30 * i)
            
            # Add trend and seasonal component
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * i / 12)  # Annual seasonality
            forecast_value = (last_value + trend * i) * seasonal_factor
            
            # Add confidence intervals
            uncertainty = forecast_value * 0.1 * (i / periods)  # Increasing uncertainty
            
            forecast_data.append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "forecast": max(0, forecast_value),
                "lower_bound": max(0, forecast_value - uncertainty),
                "upper_bound": forecast_value + uncertainty,
                "confidence": max(0.5, 0.9 - (i * 0.05))  # Decreasing confidence
            })
        
        return forecast_data
    
    def _generate_forecast_insights_with_ai(self, forecasts: List[Dict], historical: List[Dict]) -> List[Dict]:
        """Generate AI insights for forecast results"""
        
        try:
            if not os.getenv('OPENAI_API_KEY'):
                return [{"type": "info", "description": "OpenAI API key not configured", "confidence": 0.0}]
            
            # Prepare forecast context
            context = {
                "historical_data": historical,
                "forecast_data": forecasts,
                "growth_rate": self._calculate_growth_rate(forecasts, historical)
            }
            
            prompt = f"""
            Analyze this Tesla revenue forecast and provide insights:
            
            HISTORICAL DATA: {json.dumps(historical, indent=2)}
            FORECAST DATA: {json.dumps(forecasts, indent=2)}
            
            Provide insights in JSON format:
            {{
                "insights": [
                    {{
                        "type": "growth_acceleration|decline_warning|seasonality|trend_analysis",
                        "description": "Detailed insight about the forecast",
                        "confidence": 0.0-1.0,
                        "recommendation": "Actionable recommendation"
                    }}
                ]
            }}
            """
            
            response = _chat_completion(
                messages=[
                    {"role": "system", "content": "You are a financial forecasting expert specializing in Tesla and EV industry trends. Return concise JSON insights only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=1000,
            )
            
            insights_text = response.choices[0].message.content
            return self._parse_forecast_insights(insights_text)
            
        except Exception as e:
            print(f"Error generating forecast insights: {e}")
            return [{"type": "error", "description": "Unable to generate forecast insights", "confidence": 0.0}]
    
    def _calculate_growth_rate(self, forecasts: List[Dict], historical: List[Dict]) -> float:
        """Calculate growth rate from forecast"""
        
        if not forecasts or not historical:
            return 0.0
        
        last_historical = historical[-1]["value"]
        last_forecast = forecasts[-1]["forecast"]
        
        return ((last_forecast / last_historical) ** (1/len(forecasts))) - 1
    
    def _parse_forecast_insights(self, response_text: str) -> List[Dict]:
        """Parse forecast insights response"""
        
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
                return data.get("insights", [])
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Error parsing forecast insights: {e}")
            return []

class TeslaAIAgent:
    """Interactive AI Agent for Tesla model analysis with OpenAI insights"""
    
    def __init__(self):
        self.base_assumptions = None
        self.current_scenario = "base"
        
    def initialize_base_model(self, scenario: str = "base"):
        """Initialize base model assumptions"""
        self.current_scenario = scenario
        self.base_assumptions = get_enhanced_tesla_drivers(ScenarioType(scenario), 2024)
        
        return {
            "initialized": True,
            "scenario": scenario,
            "base_deliveries": self.base_assumptions["projected_deliveries"],
            "base_asp_multiplier": self.base_assumptions["asp_multiplier"]
        }
    
    def simulate_slider_changes(self, changes: Dict[str, float]) -> Dict:
        """Simulate impact of slider changes with AI insights"""
        
        if not self.base_assumptions:
            return {"error": "Model not initialized"}
        
        # Calculate new values
        new_values = self._calculate_new_values(changes)
        
        # Calculate impacts
        impacts = self._calculate_impacts(new_values, changes)
        
        # Generate AI insights
        ai_insights = self._generate_ai_insights(changes, impacts)
        
        return {
            "scenario": self.current_scenario,
            "changes_applied": changes,
            "new_values": new_values,
            "impact_analysis": impacts,
            "ai_insights": ai_insights
        }
    
    def _calculate_new_values(self, changes: Dict[str, float]) -> Dict:
        """Calculate new values based on changes"""
        
        asp_change = changes.get("asp_change", 0)
        cost_change = changes.get("cost_change", 0)
        delivery_change = changes.get("delivery_change", 0)
        
        new_asp_multiplier = self.base_assumptions["asp_multiplier"] * (1 + asp_change / 100)
        new_cost_multiplier = 1 + cost_change / 100
        
        new_deliveries = {}
        for model, base_delivery in self.base_assumptions["projected_deliveries"].items():
            new_delivery = int(base_delivery * (1 + delivery_change / 100))
            new_deliveries[model] = new_delivery
        
        return {
            "deliveries": new_deliveries,
            "asp_multiplier": new_asp_multiplier,
            "cost_multiplier": new_cost_multiplier
        }
    
    def _calculate_impacts(self, new_values: Dict, changes: Dict[str, float]) -> Dict:
        """Calculate financial impacts"""
        
        # Revenue impact
        revenue_impact = self._calculate_revenue_impact(new_values)
        
        # Margin impact
        margin_impact = self._calculate_margin_impact(revenue_impact, new_values["cost_multiplier"])
        
        # Delivery impact
        total_delivery_change = sum(new_values["deliveries"].values()) - sum(self.base_assumptions["projected_deliveries"].values())
        
        return {
            "revenue_impact": revenue_impact,
            "margin_impact": margin_impact,
            "total_delivery_change": total_delivery_change
        }
    
    def _calculate_revenue_impact(self, new_values: Dict) -> Dict:
        """Calculate revenue impact"""
        
        base_revenue = 0
        new_revenue = 0
        
        for model, new_delivery in new_values["deliveries"].items():
            base_delivery = self.base_assumptions["projected_deliveries"].get(model, 0)
            # Some years may include aggregate/unknown keys like 'others' not present in VEHICLE_MODEL_DATA
            # Use a conservative default ASP if missing (e.g., average of known base ASPs)
            if model in VEHICLE_MODEL_DATA["models"]:
                base_asp = VEHICLE_MODEL_DATA["models"][model]["base_asp"]
            else:
                try:
                    asps = [m.get("base_asp", 0) for m in VEHICLE_MODEL_DATA["models"].values()]
                    base_asp = float(sum(asps) / max(1, len(asps)))
                except Exception:
                    base_asp = 0.0
            
            model_base_revenue = base_delivery * base_asp * self.base_assumptions["asp_multiplier"]
            base_revenue += model_base_revenue
            
            model_new_revenue = new_delivery * base_asp * new_values["asp_multiplier"]
            new_revenue += model_new_revenue
        
        return {
            "base_revenue": base_revenue,
            "new_revenue": new_revenue,
            "absolute_change": new_revenue - base_revenue,
            "percentage_change": ((new_revenue / base_revenue) - 1) * 100 if base_revenue > 0 else 0
        }
    
    def _calculate_margin_impact(self, revenue_impact: Dict, cost_multiplier: float) -> Dict:
        """Calculate margin impact"""
        
        base_margin = 0.19  # Assume 19% base automotive margin
        base_costs = revenue_impact["base_revenue"] * (1 - base_margin)
        
        new_costs = base_costs * cost_multiplier
        new_margin = (revenue_impact["new_revenue"] - new_costs) / revenue_impact["new_revenue"] if revenue_impact["new_revenue"] > 0 else 0
        
        return {
            "base_margin": base_margin,
            "new_margin": new_margin,
            "margin_change": (new_margin - base_margin) * 100,
            "cost_impact": new_costs - base_costs
        }
    
    def _generate_ai_insights(self, changes: Dict[str, float], impacts: Dict) -> List[Dict]:
        """Generate AI insights using OpenAI"""
        
        try:
            if not os.getenv('OPENAI_API_KEY'):
                return [{"type": "info", "title": "AI Analysis Unavailable", "description": "OpenAI API key not configured", "risk_level": "low"}]
            
            context = {
                "changes": changes,
                "impacts": impacts,
                "scenario": self.current_scenario
            }
            
            prompt = f"""
            Analyze these Tesla model changes and provide insights:
            
            CHANGES: {json.dumps(changes, indent=2)}
            IMPACTS: {json.dumps(impacts, indent=2)}
            SCENARIO: {self.current_scenario}
            
            Provide insights in JSON format:
            {{
                "insights": [
                    {{
                        "type": "pricing_strategy|competitive_pressure|production_scaling|demand_concern|cost_inflation|operational_efficiency|financial_impact",
                        "title": "Insight title",
                        "description": "Detailed description with specific numbers",
                        "recommendation": "Actionable recommendation",
                        "risk_level": "high|medium|low"
                    }}
                ]
            }}
            """
            
            response = _chat_completion(
                messages=[
                    {"role": "system", "content": "You are a Tesla financial analyst. Return JSON only; quantify impacts (%, $, units) and be specific."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=1500,
            )
            
            insights_text = response.choices[0].message.content
            return self._parse_ai_insights(insights_text)
            
        except Exception as e:
            print(f"Error generating AI insights: {e}")
            return [{"type": "error", "title": "AI Analysis Unavailable", "description": "Unable to generate insights", "risk_level": "low"}]
    
    def _parse_ai_insights(self, response_text: str) -> List[Dict]:
        """Parse AI insights response"""
        
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
                return data.get("insights", [])
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Error parsing AI insights: {e}")
            return []

# Initialize global agents
proactive_insights_agent = ProactiveInsightsAgent()
prophet_forecasting_agent = ProphetForecastingAgent()
tesla_ai_agent = TeslaAIAgent()

# ---------------- Income Statement Insights Agent ----------------
class IncomeStatementInsightsAgent:
    """Generates insights comparing BASE vs UPDATED income statements.

    Uses GPT-3.5 Turbo explicitly as requested for concise summaries.
    """

    def __init__(self):
        self.model_name = "gpt-3.5-turbo"

    def generate_comparison_summary(self, original_list: List[Dict], updated_list: List[Dict], scenario: str) -> Dict:
        try:
            client = _get_client()
            if client is None:
                return self._fallback_summary_json(original_list, updated_list, scenario)

            rows = []
            for orig, upd in zip(original_list, updated_list):
                rows.append({
                    "year": upd.get("year"),
                    "revenue_base": orig.get("total_revenue"),
                    "revenue_updated": upd.get("total_revenue"),
                    "gross_margin_base": orig.get("gross_margin"),
                    "gross_margin_updated": upd.get("gross_margin"),
                    "operating_income_base": orig.get("operating_income"),
                    "operating_income_updated": upd.get("operating_income"),
                    "net_income_base": orig.get("net_income"),
                    "net_income_updated": upd.get("net_income"),
                })

            prompt = (
                "You are an equity research analyst. Compare UPDATED vs BASE Income Statements across 2025–2029. "
                "Quantify absolute and % changes for Total Revenue, Gross Margin (in bps), Operating Income, Net Income, and EPS.\n\n"
                "Return strictly valid JSON with this schema and nothing else (no prose outside the JSON):\n"
                "{\n"
                "  \"executive_summary\": string,\n"
                "  \"commentary\": string,\n"
                "  \"highlights\": [string, ...],\n"
                "  \"watchouts\": [string, ...],\n"
                "  \"key_deltas\": [string, ...],\n"
                "  \"actionable_insights\": [string, ...],\n"
                "  \"recommendations\": [string, ...]\n"
                "}\n\n"
                "Rules:\n"
                "- Format money as $X.YB or $X.YM; margins as X.Y%; basis points as +/−XX bps.\n"
                "- Quantify every bullet (absolute $ or bps AND % where relevant).\n"
                "- Call out best/worst year for Revenue, GM bps, Operating Margin, Net Margin.\n"
                "- Attribute drivers (e.g., GM +60 bps from COGS −$0.9B).\n"
                "- Each bullet ≤ 22 words.\n\n"
                f"SCENARIO: {scenario}\nROWS: {json.dumps(rows)}"
            )

            resp = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Return only minified JSON matching the schema. Be precise and numeric."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.4,
                max_tokens=600,
            )
            text = resp.choices[0].message.content
            try:
                # Extract JSON object
                start = text.find('{')
                end = text.rfind('}') + 1
                json_str = text[start:end] if start != -1 and end != -1 else text
                data = json.loads(json_str)
                # Ensure keys present
                return {
                    "executive_summary": data.get("executive_summary", ""),
                    "commentary": data.get("commentary", ""),
                    "highlights": data.get("highlights", []),
                    "watchouts": data.get("watchouts", []),
                    "key_deltas": data.get("key_deltas", []),
                    "actionable_insights": data.get("actionable_insights", []),
                    "recommendations": data.get("recommendations", []),
                    "raw": text,
                }
            except Exception:
                return self._fallback_summary_json(original_list, updated_list, scenario)
        except Exception:
            return self._fallback_summary_json(original_list, updated_list, scenario)

    def _fallback_summary_json(self, original_list: List[Dict], updated_list: List[Dict], scenario: str) -> Dict:
        # Deterministic fallback structured summary
        lines = []
        for orig, upd in zip(original_list, updated_list):
            year = upd.get("year")
            def pct(a, b):
                try:
                    return ((b / a) - 1) * 100 if a else 0.0
                except Exception:
                    return 0.0
            rev_delta = upd.get("total_revenue", 0) - orig.get("total_revenue", 0)
            op_delta = upd.get("operating_income", 0) - orig.get("operating_income", 0)
            ni_delta = upd.get("net_income", 0) - orig.get("net_income", 0)
            lines.append(
                f"{year}: Revenue Δ {rev_delta:,.0f} ({pct(orig.get('total_revenue',0), upd.get('total_revenue',0)):.1f}%), "
                f"OpInc Δ {op_delta:,.0f}, NetInc Δ {ni_delta:,.0f}"
            )
        exec_sum = (
            f"Income Statement comparison for {scenario.title()} shows measurable changes across years; "
            f"see key deltas for magnitudes and directions."
        )
        commentary = (
            "Trajectory indicates operating leverage potential if opex growth is held below revenue growth. "
            "Monitor gross margin bps and SG&A efficiency to sustain NI improvements."
        )
        recs = [
            "Tighten cost control to convert revenue gains into operating leverage (target 30–50 bps opex).",
            "Prioritize high-margin offerings/regions to lift gross margin (50–100 bps goal).",
            "Phase SG&A efficiency programs (automation, channel mix) to add ~20–40 bps to NI.",
            "Set quarterly tracking: revenue mix, bps progress, and $ impact vs plan.",
        ]
        actionables = [
            "Launch cost-to-serve review to remove 20–30 bps from SG&A within 12 months.",
            "Negotiate supplier terms focusing on logistics and materials; target 30–50 bps COGS relief.",
            "Shift marketing to top-quartile margin regions/models contributing >15% of revenue delta.",
        ]
        return {
            "executive_summary": exec_sum,
            "commentary": commentary,
            "highlights": [],
            "watchouts": [],
            "key_deltas": lines,
            "actionable_insights": actionables,
            "recommendations": recs,
            "raw": "fallback"
        }

# Singleton
income_statement_insights_agent = IncomeStatementInsightsAgent()

# ---------------- Cross-Statement Insights Agent ----------------
class CrossStatementInsightsAgent:
    """AI agent for analyzing cross-statement financial simulations"""
    
    def __init__(self):
        self.model_name = "gpt-4o-mini"
        self.client = None  # Don't initialize at module load - do it per request
    
    def analyze_cross_statement_simulation(self, original_statements: Dict, updated_statements: Dict, scenario: str) -> Dict:
        """Generate comprehensive insights across Income Statement, Balance Sheet, and Cash Flow using OpenAI"""
        try:
            # Always re-initialize client per request to ensure we have latest environment
            api_key = os.getenv('OPENAI_API_KEY')
            print(f"[CrossStatementInsights] Checking API key...")
            print(f"[CrossStatementInsights] API key exists: {api_key is not None}")
            if api_key:
                print(f"[CrossStatementInsights] API key starts with: {api_key[:10]}...")
                print(f"[CrossStatementInsights] API key length: {len(api_key)}")
            print(f"[CrossStatementInsights] API key valid: {api_key != 'your_openai_api_key_here' if api_key else False}")
            
            # Clean the API key (remove quotes and whitespace)
            if api_key:
                api_key = api_key.strip().strip('"').strip("'")
                print(f"[CrossStatementInsights] Cleaned API key starts with: {api_key[:10]}...")
            
            if api_key and api_key != 'your_openai_api_key_here' and len(api_key) > 20:
                try:
                    # Re-initialize client each time
                    self.client = OpenAI(api_key=api_key, timeout=60.0)
                    print(f"[CrossStatementInsights] OpenAI client initialized successfully")
                except Exception as init_error:
                    print(f"[CrossStatementInsights] Failed to initialize: {init_error}")
                    import traceback
                    traceback.print_exc()
                    return self._fallback_cross_statement_summary_original(original_statements, updated_statements, scenario)
            else:
                print(f"[CrossStatementInsights] No valid API key - using fallback (key length: {len(api_key) if api_key else 0})")
                return self._fallback_cross_statement_summary_original(original_statements, updated_statements, scenario)
            
            # Prepare data for all three statements
            analysis_data = self._prepare_cross_statement_data(original_statements, updated_statements)
            
            prompt = self._build_cross_statement_prompt(analysis_data, scenario)
            
            print(f"Calling OpenAI API for cross-statement insights...")
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a senior financial analyst specializing in integrated financial statement analysis. Provide precise, quantitative insights in the specified JSON format."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=2000,
                timeout=45.0
            )
            
            content = resp.choices[0].message.content
            print(f"OpenAI response received successfully")
            
            # Parse JSON response - handle markdown code blocks
            try:
                # Remove markdown code blocks if present
                cleaned_content = content.strip()
                if cleaned_content.startswith('```json'):
                    cleaned_content = cleaned_content[7:]  # Remove ```json
                if cleaned_content.startswith('```'):
                    cleaned_content = cleaned_content[3:]  # Remove ```
                if cleaned_content.endswith('```'):
                    cleaned_content = cleaned_content[:-3]  # Remove trailing ```
                cleaned_content = cleaned_content.strip()
                
                print(f"Parsing cleaned JSON (first 100 chars): {cleaned_content[:100]}...")
                return json.loads(cleaned_content)
            except json.JSONDecodeError as je:
                print(f"Failed to parse OpenAI JSON response: {je}")
                print(f"Response content: {content[:200]}...")
                return self._fallback_cross_statement_summary_original(original_statements, updated_statements, scenario)
            
        except Exception as e:
            print(f"Error in cross-statement analysis: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_cross_statement_summary_original(original_statements, updated_statements, scenario)
    
    def _prepare_cross_statement_data(self, original: Dict, updated: Dict) -> Dict:
        """Prepare structured data for analysis"""
        data = {"years": []}
        
        # Combine data by year
        for i, year in enumerate([2025, 2026, 2027, 2028, 2029]):
            if i < len(original["income_statements"]) and i < len(updated["income_statements"]):
                orig_inc = original["income_statements"][i]
                upd_inc = updated["income_statements"][i]
                orig_bs = original["balance_sheets"][i] if i < len(original["balance_sheets"]) else {}
                upd_bs = updated["balance_sheets"][i] if i < len(updated["balance_sheets"]) else {}
                orig_cf = original["cash_flow_statements"][i] if i < len(original["cash_flow_statements"]) else {}
                upd_cf = updated["cash_flow_statements"][i] if i < len(updated["cash_flow_statements"]) else {}
                
                year_data = {
                    "year": year,
                    "revenue_base": orig_inc.get("total_revenue", 0),
                    "revenue_updated": upd_inc.get("total_revenue", 0),
                    "operating_income_base": orig_inc.get("operating_income", 0),
                    "operating_income_updated": upd_inc.get("operating_income", 0),
                    "net_income_base": orig_inc.get("net_income", 0),
                    "net_income_updated": upd_inc.get("net_income", 0),
                    "free_cash_flow_base": orig_cf.get("operating_cash_flow", 0) - abs(orig_cf.get("capital_expenditures", 0)),
                    "free_cash_flow_updated": upd_cf.get("operating_cash_flow", 0) - abs(upd_cf.get("capital_expenditures", 0)),
                    "total_assets_base": orig_bs.get("total_assets", 0),
                    "total_assets_updated": upd_bs.get("total_assets", 0),
                    "cash_base": orig_bs.get("cash_and_equivalents", 0),
                    "cash_updated": upd_bs.get("cash_and_equivalents", 0)
                }
                data["years"].append(year_data)
        
        return data
    
    def _build_cross_statement_prompt(self, data: Dict, scenario: str) -> str:
        return f"""
Analyze this cross-statement financial simulation for Tesla ({scenario} scenario). 
Provide insights for BOTH year 2025 (near-term focus) AND year 2029 (long-term outlook) in EXACTLY this JSON format:

{{
  "executive_summary": "2-3 sentence overview covering both 2025 and 2029 changes",
  "financial_performance": {{
    "revenue_impact": "Revenue analysis for 2025, then 2029 with specific numbers and growth rates",
    "profitability_impact": "Operating and net income analysis for 2025, then 2029 with margins",
    "cash_generation": "Free cash flow impact for 2025, then 2029 with cash position changes"
  }},
  "balance_sheet_insights": {{
    "asset_efficiency": "Asset utilization changes for 2025, then 2029",
    "working_capital": "Working capital impact for 2025, then 2029 on cash cycle",
    "financial_position": "Overall balance sheet strength for 2025, then 2029"
  }},
  "integrated_metrics": {{
    "cash_conversion": "Earnings-to-cash conversion for 2025, then 2029",
    "capital_efficiency": "ROIC and asset turnover for 2025, then 2029",
    "growth_sustainability": "Growth sustainability assessment for 2025, then 2029 given capital needs"
  }},
  "key_risks": ["3-4 specific risks from the simulation covering both near-term and long-term"],
  "strategic_recommendations": ["3-4 actionable recommendations with quantified targets for 2025 and 2029"]
}}

Data: {json.dumps(data)}

Rules:
- MUST include insights for BOTH 2025 (first/prime focus year) AND 2029 (final year)
- For each metric, provide 2025 analysis FIRST, then 2029 analysis
- Include specific dollar amounts (e.g., "$45.2B in 2025, growing to $64.6B by 2029")  
- Calculate and mention percentage changes for both years
- Focus on cause-and-effect relationships between statements
- Highlight cash flow vs earnings quality for both periods
- Keep each field concise but quantitative for both time periods
"""
    
    def _fallback_cross_statement_summary_original(self, original: Dict, updated: Dict, scenario: str) -> Dict:
        """Provide basic summary when AI is unavailable - ORIGINAL VERSION"""
        try:
            revenue_change = 0
            if original.get("income_statements") and updated.get("income_statements"):
                orig_rev = sum(stmt.get("total_revenue", 0) for stmt in original["income_statements"])
                upd_rev = sum(stmt.get("total_revenue", 0) for stmt in updated["income_statements"])
                revenue_change = ((upd_rev - orig_rev) / orig_rev * 100) if orig_rev > 0 else 0
            
            return {
                "executive_summary": f"Cross-statement simulation shows {revenue_change:.1f}% revenue impact with corresponding balance sheet and cash flow changes.",
                "financial_performance": {
                    "revenue_impact": f"Revenue change of {revenue_change:.1f}% across the simulation period",
                    "profitability_impact": "Profitability analysis unavailable in fallback mode",
                    "cash_generation": "Cash flow analysis unavailable in fallback mode"
                },
                "balance_sheet_insights": {
                    "asset_efficiency": "Asset analysis unavailable in fallback mode",
                    "working_capital": "Working capital analysis unavailable in fallback mode", 
                    "financial_position": "Balance sheet analysis unavailable in fallback mode"
                },
                "integrated_metrics": {
                    "cash_conversion": "Detailed metrics unavailable in fallback mode",
                    "capital_efficiency": "ROIC analysis unavailable in fallback mode",
                    "growth_sustainability": "Sustainability analysis unavailable in fallback mode"
                },
                "key_risks": ["AI analysis unavailable", "Manual review recommended"],
                "strategic_recommendations": ["Review simulation parameters", "Validate assumptions", "Consider scenario analysis"]
            }
        except:
            return {
                "executive_summary": "Cross-statement analysis completed with basic fallback metrics",
                "error": "Detailed analysis unavailable"
            }

# Singleton for cross-statement analysis
cross_statement_insights_agent = CrossStatementInsightsAgent()