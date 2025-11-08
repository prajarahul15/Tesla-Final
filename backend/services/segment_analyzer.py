"""
PHASE 2: Business Segment Analysis and Working Capital Modeling
PHASE 3: Bridge Analysis and Price-Volume-Mix Analytics
"""

from typing import Dict, List, Tuple
from data.tesla_enhanced_data import BUSINESS_SEGMENTS, PRICE_VOLUME_MIX_DATA, TESLA_HISTORICAL_DATA
import numpy as np

class TeslaSegmentAnalyzer:
    """Advanced segment analysis and bridge calculations"""
    
    def __init__(self):
        self.segments = BUSINESS_SEGMENTS
        self.historical_data = TESLA_HISTORICAL_DATA
        self.pvm_data = PRICE_VOLUME_MIX_DATA
    
    def analyze_business_segments(self, enhanced_models: Dict) -> Dict:
        """
        PHASE 2: Comprehensive business segment analysis
        """
        segment_analysis = {
            "segment_summary": {},
            "segment_trends": {},
            "segment_margins": {},
            "segment_growth_rates": {}
        }
        
        for scenario, model_data in enhanced_models.items():
            scenario_segments = {
                "automotive": {"revenue": [], "margins": [], "years": []},
                "energy": {"revenue": [], "margins": [], "years": []},
                "services": {"revenue": [], "margins": [], "years": []}
            }
            
            for income_stmt in model_data["income_statements"]:
                year = income_stmt["year"]
                
                # Automotive segment
                automotive_revenue = income_stmt["automotive_revenue"]
                automotive_margin = income_stmt["margins"]["automotive_margin"]
                
                scenario_segments["automotive"]["revenue"].append(automotive_revenue)
                scenario_segments["automotive"]["margins"].append(automotive_margin)
                scenario_segments["automotive"]["years"].append(year)
                
                # Energy segment
                energy_revenue = income_stmt["energy_revenue"]
                energy_margin = income_stmt["margins"]["energy_margin"]
                
                scenario_segments["energy"]["revenue"].append(energy_revenue)
                scenario_segments["energy"]["margins"].append(energy_margin)
                scenario_segments["energy"]["years"].append(year)
                
                # Services segment
                services_revenue = income_stmt["services_revenue"]
                services_margin = income_stmt["margins"]["services_margin"]
                
                scenario_segments["services"]["revenue"].append(services_revenue)
                scenario_segments["services"]["margins"].append(services_margin)
                scenario_segments["services"]["years"].append(year)
            
            # Calculate segment growth rates and trends
            for segment_name, segment_data in scenario_segments.items():
                if len(segment_data["revenue"]) > 1:
                    # Calculate CAGR
                    initial_revenue = segment_data["revenue"][0]
                    final_revenue = segment_data["revenue"][-1]
                    years = len(segment_data["revenue"]) - 1
                    
                    if initial_revenue > 0:
                        cagr = ((final_revenue / initial_revenue) ** (1/years)) - 1
                    else:
                        cagr = 0
                    
                    # Average margin
                    avg_margin = sum(segment_data["margins"]) / len(segment_data["margins"])
                    
                    if scenario not in segment_analysis["segment_summary"]:
                        segment_analysis["segment_summary"][scenario] = {}
                    
                    segment_analysis["segment_summary"][scenario][segment_name] = {
                        "revenue_cagr": cagr,
                        "average_margin": avg_margin,
                        "initial_revenue": initial_revenue,
                        "final_revenue": final_revenue,
                        "revenue_multiple": final_revenue / initial_revenue if initial_revenue > 0 else 0
                    }
        
        return segment_analysis
    
    def calculate_revenue_bridge(self, base_year_data: Dict, current_year_data: Dict) -> Dict:
        """
        PHASE 3: Revenue bridge analysis (waterfall chart data)
        """
        base_revenue = base_year_data["total_revenue"]
        current_revenue = current_year_data["total_revenue"]
        total_change = current_revenue - base_revenue
        
        # Break down revenue change by components
        volume_effect = 0
        price_effect = 0
        mix_effect = 0
        
        # Calculate volume effect (deliveries × base ASP)
        if "revenue_breakdown" in current_year_data and "revenue_breakdown" in base_year_data:
            for model_key in current_year_data["revenue_breakdown"]["automotive_revenue_by_model"]:
                if model_key in base_year_data["revenue_breakdown"]["automotive_revenue_by_model"]:
                    current_model = current_year_data["revenue_breakdown"]["automotive_revenue_by_model"][model_key]
                    base_model = base_year_data["revenue_breakdown"]["automotive_revenue_by_model"][model_key]
                    
                    # Volume effect = ΔDeliveries × Base ASP
                    delivery_change = current_model["deliveries"] - base_model["deliveries"]
                    volume_effect += delivery_change * base_model["asp"]
                    
                    # Price effect = Base Deliveries × ΔASP
                    asp_change = current_model["asp"] - base_model["asp"]
                    price_effect += base_model["deliveries"] * asp_change
        
        # Mix effect (residual)
        mix_effect = total_change - volume_effect - price_effect
        
        # Energy and services growth
        energy_growth = current_year_data.get("energy_revenue", 0) - base_year_data.get("energy_revenue", 0)
        services_growth = current_year_data.get("services_revenue", 0) - base_year_data.get("services_revenue", 0)
        
        return {
            "base_revenue": base_revenue,
            "current_revenue": current_revenue,
            "total_change": total_change,
            "bridge_components": {
                "volume_effect": volume_effect,
                "price_effect": price_effect,
                "mix_effect": mix_effect,
                "energy_growth": energy_growth,
                "services_growth": services_growth
            },
            "bridge_percentages": {
                "volume_contribution": (volume_effect / abs(total_change)) * 100 if total_change != 0 else 0,
                "price_contribution": (price_effect / abs(total_change)) * 100 if total_change != 0 else 0,
                "mix_contribution": (mix_effect / abs(total_change)) * 100 if total_change != 0 else 0,
                "energy_contribution": (energy_growth / abs(total_change)) * 100 if total_change != 0 else 0,
                "services_contribution": (services_growth / abs(total_change)) * 100 if total_change != 0 else 0
            }
        }
    
    def analyze_price_volume_mix(self, enhanced_models: Dict) -> Dict:
        """
        PHASE 3: Comprehensive Price-Volume-Mix analysis
        """
        pvm_analysis = {}
        
        for scenario, model_data in enhanced_models.items():
            scenario_pvm = {
                "historical_trends": {},
                "forecast_trends": {},
                "elasticity_analysis": {},
                "mix_shift_analysis": {}
            }
            
            # Analyze each model's price-volume relationship
            for income_stmt in model_data["income_statements"]:
                year = income_stmt["year"]
                
                if "revenue_breakdown" in income_stmt:
                    year_pvm = {}
                    total_deliveries = 0
                    total_automotive_revenue = 0
                    
                    for model_key, model_data_year in income_stmt["revenue_breakdown"]["automotive_revenue_by_model"].items():
                        deliveries = model_data_year["deliveries"]
                        asp = model_data_year["asp"]
                        revenue = model_data_year["revenue"]
                        
                        year_pvm[model_key] = {
                            "deliveries": deliveries,
                            "asp": asp,
                            "revenue": revenue,
                            "market_share": 0  # Will calculate after totals
                        }
                        
                        total_deliveries += deliveries
                        total_automotive_revenue += revenue
                    
                    # Calculate market share and blended ASP
                    for model_key in year_pvm:
                        if total_deliveries > 0:
                            year_pvm[model_key]["market_share"] = year_pvm[model_key]["deliveries"] / total_deliveries
                    
                    year_pvm["totals"] = {
                        "total_deliveries": total_deliveries,
                        "blended_asp": total_automotive_revenue / total_deliveries if total_deliveries > 0 else 0,
                        "total_revenue": total_automotive_revenue
                    }
                    
                    scenario_pvm["forecast_trends"][year] = year_pvm
            
            # Calculate price elasticity impacts
            elasticity_analysis = {}
            years = sorted(scenario_pvm["forecast_trends"].keys())
            
            for i in range(1, len(years)):
                current_year = years[i]
                previous_year = years[i-1]
                
                current_data = scenario_pvm["forecast_trends"][current_year]
                previous_data = scenario_pvm["forecast_trends"][previous_year]
                
                for model_key in current_data:
                    if model_key != "totals" and model_key in previous_data:
                        current_model = current_data[model_key]
                        previous_model = previous_data[model_key]
                        
                        # Calculate price and volume changes
                        if previous_model["asp"] > 0 and previous_model["deliveries"] > 0:
                            price_change = (current_model["asp"] - previous_model["asp"]) / previous_model["asp"]
                            volume_change = (current_model["deliveries"] - previous_model["deliveries"]) / previous_model["deliveries"]
                            
                            # Implied elasticity
                            if price_change != 0:
                                implied_elasticity = volume_change / price_change
                            else:
                                implied_elasticity = 0
                            
                            if model_key not in elasticity_analysis:
                                elasticity_analysis[model_key] = []
                            
                            elasticity_analysis[model_key].append({
                                "year": current_year,
                                "price_change": price_change,
                                "volume_change": volume_change,
                                "implied_elasticity": implied_elasticity
                            })
            
            scenario_pvm["elasticity_analysis"] = elasticity_analysis
            
            # Mix shift analysis (how product mix changes over time)
            mix_analysis = {}
            for year in years:
                year_data = scenario_pvm["forecast_trends"][year]
                mix_analysis[year] = {}
                
                for model_key in year_data:
                    if model_key != "totals":
                        mix_analysis[year][model_key] = {
                            "volume_share": year_data[model_key]["market_share"],
                            "revenue_share": year_data[model_key]["revenue"] / year_data["totals"]["total_revenue"] if year_data["totals"]["total_revenue"] > 0 else 0,
                            "asp_premium": year_data[model_key]["asp"] / year_data["totals"]["blended_asp"] if year_data["totals"]["blended_asp"] > 0 else 0
                        }
            
            scenario_pvm["mix_shift_analysis"] = mix_analysis
            pvm_analysis[scenario] = scenario_pvm
        
        return pvm_analysis
    
    def calculate_cash_flow_bridge(self, base_cash_flow: float, current_cash_flow: float, 
                                 income_statements: List[Dict]) -> Dict:
        """
        PHASE 3: Cash flow bridge analysis
        """
        total_change = current_cash_flow - base_cash_flow
        
        # Estimate components (simplified for demonstration)
        if len(income_statements) >= 2:
            current_stmt = income_statements[-1]
            previous_stmt = income_statements[-2]
            
            operating_change = current_stmt["operating_income"] - previous_stmt["operating_income"]
            working_capital_change = -abs(operating_change) * 0.1  # Assume 10% WC impact
            capex_change = -current_stmt["automotive_revenue"] * 0.02  # Assume incremental CapEx
            
        else:
            operating_change = total_change * 0.7
            working_capital_change = total_change * 0.1
            capex_change = total_change * 0.2
        
        return {
            "base_cash_flow": base_cash_flow,
            "current_cash_flow": current_cash_flow,
            "total_change": total_change,
            "bridge_components": {
                "operating_performance": operating_change,
                "working_capital_impact": working_capital_change,
                "capex_investment": capex_change,
                "other_items": total_change - operating_change - working_capital_change - capex_change
            }
        }
    
    def generate_comprehensive_analysis(self, enhanced_models: Dict) -> Dict:
        """
        Generate comprehensive analysis combining all Phase 2 & 3 features
        """
        analysis = {
            "segment_analysis": self.analyze_business_segments(enhanced_models),
            "price_volume_mix": self.analyze_price_volume_mix(enhanced_models),
            "bridge_analysis": {},
            "key_insights": {}
        }
        
        # Generate bridge analysis for each scenario
        for scenario, model_data in enhanced_models.items():
            income_statements = model_data["income_statements"]
            
            if len(income_statements) >= 2:
                base_year_data = income_statements[0]
                final_year_data = income_statements[-1]
                
                revenue_bridge = self.calculate_revenue_bridge(base_year_data, final_year_data)
                
                # Estimate cash flow bridge
                base_fcf = model_data["dcf_valuation"]["projected_free_cash_flows"][0] if model_data["dcf_valuation"]["projected_free_cash_flows"] else 0
                final_fcf = model_data["dcf_valuation"]["projected_free_cash_flows"][-1] if model_data["dcf_valuation"]["projected_free_cash_flows"] else 0
                
                cash_flow_bridge = self.calculate_cash_flow_bridge(base_fcf, final_fcf, income_statements)
                
                analysis["bridge_analysis"][scenario] = {
                    "revenue_bridge": revenue_bridge,
                    "cash_flow_bridge": cash_flow_bridge
                }
        
        # Generate key insights
        insights = []
        
        # Segment growth insights
        if "segment_summary" in analysis["segment_analysis"]:
            for scenario, segments in analysis["segment_analysis"]["segment_summary"].items():
                if "energy" in segments and segments["energy"]["revenue_cagr"] > 0.3:
                    insights.append(f"Energy segment shows exceptional growth in {scenario} scenario with {segments['energy']['revenue_cagr']:.1%} CAGR")
                
                if "services" in segments and segments["services"]["average_margin"] > 0.4:
                    insights.append(f"Services maintains high-margin profile in {scenario} scenario with {segments['services']['average_margin']:.1%} average margin")
        
        analysis["key_insights"] = insights
        
        return analysis