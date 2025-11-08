"""
Enhanced Tesla Financial Calculator with Driver-Based Modeling
Implements PHASE 1-3: Vehicle models, 10-year forecasts, segment analysis
"""

import math
from typing import List, Dict, Tuple, Optional
from models.financial_models import *
from data.tesla_enhanced_data import (
    TESLA_HISTORICAL_DATA, VEHICLE_MODEL_DATA, BUSINESS_SEGMENTS,
    get_enhanced_tesla_drivers, PRICE_VOLUME_MIX_DATA
)

class EnhancedTeslaCalculator:
    """Enhanced calculator with driver-based modeling and vehicle granularity"""
    
    def __init__(self):
        self.historical_data = TESLA_HISTORICAL_DATA
        self.vehicle_models = VEHICLE_MODEL_DATA
        self.business_segments = BUSINESS_SEGMENTS
        
    def calculate_driver_based_revenue(self, drivers: Dict, year: int) -> Dict:
        """
        PHASE 1: Calculate revenue using driver-based approach (Deliveries × ASP)
        """
        revenue_breakdown = {
            "automotive_revenue_by_model": {},
            "total_automotive_revenue": 0,
            "energy_revenue": 0,
            "services_revenue": 0,
            "total_revenue": 0
        }
        
        # Calculate automotive revenue by model (Volume × ASP approach)
        total_automotive_revenue = 0
        
        for model_key, delivery_count in drivers["projected_deliveries"].items():
            if model_key in self.vehicle_models["models"]:
                model_data = self.vehicle_models["models"][model_key]
                
                # Calculate ASP with scenario adjustments
                base_asp = model_data["base_asp"]
                adjusted_asp = base_asp * drivers["asp_multiplier"]
                
                # Apply year-over-year pricing trends (slight decline due to scale)
                years_from_2024 = max(0, year - 2024)
                pricing_trend = 0.98 ** years_from_2024  # 2% annual decline due to scale
                final_asp = adjusted_asp * pricing_trend
                
                # Calculate model revenue
                model_revenue = float(delivery_count * final_asp)
                
                revenue_breakdown["automotive_revenue_by_model"][model_key] = {
                    "deliveries": int(delivery_count),
                    "asp": float(final_asp),
                    "revenue": float(model_revenue),
                    "model_name": model_data["name"]
                }
                
                total_automotive_revenue += model_revenue
        
        revenue_breakdown["total_automotive_revenue"] = float(total_automotive_revenue)
        
        # Calculate Energy & Storage revenue (based on growth rates)
        if year <= 2023:
            energy_revenue = float(self.historical_data["historical_revenue"][year]["energy"] * 1000000)
        else:
            # Start from 2023 base and compound
            base_energy_2023 = self.historical_data["historical_revenue"][2023]["energy"] * 1000000
            years_from_2023 = year - 2023
            energy_revenue = float(base_energy_2023 * ((1 + drivers["energy_growth_rate"]) ** years_from_2023))
        
        revenue_breakdown["energy_revenue"] = energy_revenue
        
        # Calculate Services revenue (based on growth rates)
        if year <= 2023:
            services_revenue = float(self.historical_data["historical_revenue"][year]["services"] * 1000000)
        else:
            base_services_2023 = self.historical_data["historical_revenue"][2023]["services"] * 1000000
            years_from_2023 = year - 2023
            services_revenue = float(base_services_2023 * ((1 + drivers["services_growth_rate"]) ** years_from_2023))
        
        revenue_breakdown["services_revenue"] = services_revenue
        
        # Total revenue
        revenue_breakdown["total_revenue"] = float(
            total_automotive_revenue + energy_revenue + services_revenue
        )
        
        return revenue_breakdown
    
    def calculate_enhanced_margins(self, revenue_breakdown: Dict, drivers: Dict, year: int) -> Dict:
        """
        Calculate margins with vehicle model and segment specificity
        """
        margins = {}
        
        # Automotive margins by model (different models have different margins)
        automotive_cogs = 0
        
        for model_key, model_revenue_data in revenue_breakdown["automotive_revenue_by_model"].items():
            if model_key in self.vehicle_models["models"]:
                model_data = self.vehicle_models["models"][model_key]
                
                # Base margin for this model
                base_margin = model_data["margin_premium"]
                
                # Apply scenario-based margin improvements
                adjusted_margin = base_margin + drivers["automotive_margin_improvement"]
                
                # Scale benefits (higher volumes = better margins)
                delivery_count = model_revenue_data["deliveries"]
                scale_benefit = min(0.05, delivery_count / 500000 * 0.02)  # Up to 5% benefit
                final_margin = min(0.35, adjusted_margin + scale_benefit)  # Cap at 35%
                
                # Calculate COGS
                model_revenue = model_revenue_data["revenue"]
                model_cogs = model_revenue * (1 - final_margin)
                automotive_cogs += model_cogs
                
                margins[f"{model_key}_margin"] = final_margin
        
        # Energy segment margins (higher margin business)
        energy_margin = 0.22  # Based on historical analysis
        energy_cogs = revenue_breakdown["energy_revenue"] * (1 - energy_margin)
        
        # Services margins (highest margin business)
        services_margin = 0.45  # High-margin software and services
        services_cogs = revenue_breakdown["services_revenue"] * (1 - services_margin)
        
        margins.update({
            "automotive_cogs": automotive_cogs,
            "automotive_gross_profit": revenue_breakdown["total_automotive_revenue"] - automotive_cogs,
            "automotive_margin": (revenue_breakdown["total_automotive_revenue"] - automotive_cogs) / revenue_breakdown["total_automotive_revenue"] if revenue_breakdown["total_automotive_revenue"] > 0 else 0,
            
            "energy_cogs": energy_cogs,
            "energy_gross_profit": revenue_breakdown["energy_revenue"] - energy_cogs,
            "energy_margin": energy_margin,
            
            "services_cogs": services_cogs,
            "services_gross_profit": revenue_breakdown["services_revenue"] - services_cogs,
            "services_margin": services_margin,
            
            "total_cogs": automotive_cogs + energy_cogs + services_cogs,
            "total_gross_profit": (revenue_breakdown["total_automotive_revenue"] - automotive_cogs + 
                                 revenue_breakdown["energy_revenue"] - energy_cogs + 
                                 revenue_breakdown["services_revenue"] - services_cogs),
        })
        
        margins["total_margin"] = margins["total_gross_profit"] / revenue_breakdown["total_revenue"] if revenue_breakdown["total_revenue"] > 0 else 0
        
        return margins
    
    def calculate_enhanced_income_statement(self, scenario: ScenarioType, year: int) -> Dict:
        """
        Enhanced income statement with driver-based calculations
        """
        # Get enhanced drivers for this scenario and year
        drivers = get_enhanced_tesla_drivers(scenario, year)
        
        # Calculate revenue using driver-based approach
        revenue_breakdown = self.calculate_driver_based_revenue(drivers, year)
        
        # Calculate enhanced margins
        margins = self.calculate_enhanced_margins(revenue_breakdown, drivers, year)
        
        # Operating expenses with efficiency improvements
        total_revenue = revenue_breakdown["total_revenue"]
        
        rd_expense = total_revenue * drivers["rd_efficiency"]
        sga_expense = total_revenue * drivers["sga_efficiency"]
        total_opex = rd_expense + sga_expense
        
        # Operating income
        operating_income = margins["total_gross_profit"] - total_opex
        operating_margin = operating_income / total_revenue if total_revenue > 0 else 0
        
        # Other income/expenses
        interest_income = 28000000000 * 0.03  # Interest on cash
        interest_expense = 5230000000 * drivers["cost_of_debt"]  # Interest on debt
        
        # Pre-tax income
        pretax_income = operating_income + interest_income - interest_expense
        
        # Taxes
        tax_expense = max(0, pretax_income * drivers["tax_rate"]) if pretax_income > 0 else 0
        
        # Net income
        net_income = pretax_income - tax_expense
        net_margin = net_income / total_revenue if total_revenue > 0 else 0
        
        # Shares outstanding (assume gradual increase due to employee stock plans)
        base_shares = 3178000000
        shares_growth = (year - 2023) * 0.02  # 2% annual increase
        shares_outstanding = base_shares * (1 + shares_growth)
        
        eps = net_income / shares_outstanding
        
        return {
            "scenario": scenario.value,  # Convert enum to string
            "year": int(year),
            "revenue_breakdown": revenue_breakdown,
            "margins": margins,
            "drivers": {k: float(v) if isinstance(v, (int, float)) else v for k, v in drivers.items()},
            
            # Income statement items
            "automotive_revenue": float(revenue_breakdown["total_automotive_revenue"]),
            "energy_revenue": float(revenue_breakdown["energy_revenue"]),
            "services_revenue": float(revenue_breakdown["services_revenue"]),
            "total_revenue": float(total_revenue),
            
            "automotive_cogs": float(margins["automotive_cogs"]),
            "energy_cogs": float(margins["energy_cogs"]),
            "services_cogs": float(margins["services_cogs"]),
            "total_cogs": float(margins["total_cogs"]),
            
            "automotive_gross_profit": float(margins["automotive_gross_profit"]),
            "energy_gross_profit": float(margins["energy_gross_profit"]),
            "services_gross_profit": float(margins["services_gross_profit"]),
            "total_gross_profit": float(margins["total_gross_profit"]),
            "gross_margin": float(margins["total_margin"]),
            
            "research_development": float(rd_expense),
            "selling_general_admin": float(sga_expense),
            "total_operating_expenses": float(total_opex),
            
            "operating_income": float(operating_income),
            "operating_margin": float(operating_margin),
            
            "interest_income": float(interest_income),
            "interest_expense": float(interest_expense),
            "other_income": 0.0,
            
            "pretax_income": float(pretax_income),
            "income_tax_expense": float(tax_expense),
            "effective_tax_rate": float(drivers["tax_rate"]),
            
            "net_income": float(net_income),
            "net_margin": float(net_margin),
            
            "shares_outstanding": float(shares_outstanding),
            "earnings_per_share": float(eps)
        }
    
    def calculate_enhanced_working_capital(self, income_statement: Dict, drivers: Dict) -> Dict:
        """
        PHASE 2: Enhanced working capital modeling with targets
        """
        total_revenue = income_statement["total_revenue"]
        total_cogs = income_statement["total_cogs"]
        
        # Calculate working capital components using targets
        accounts_receivable = (drivers["dso_target"] / 365) * total_revenue
        inventory = (drivers["dio_target"] / 365) * total_cogs
        accounts_payable = (60 / 365) * total_cogs  # Assume 60-day payment terms
        
        net_working_capital = accounts_receivable + inventory - accounts_payable
        
        return {
            "accounts_receivable": accounts_receivable,
            "inventory": inventory,
            "accounts_payable": accounts_payable,
            "net_working_capital": net_working_capital,
            "dso_actual": drivers["dso_target"],
            "dio_actual": drivers["dio_target"],
            "dpo_actual": 60,
            "working_capital_as_percent_revenue": net_working_capital / total_revenue if total_revenue > 0 else 0
        }
    
    def calculate_enhanced_dcf_valuation(self, scenario: ScenarioType, income_statements: List[Dict]) -> Dict:
        """
        PHASE 2: Enhanced DCF with 10-year forecasts and better sensitivity
        """
        if not income_statements:
            return {}
        
        # Calculate WACC using market-based inputs from Tesla 2.xlsx
        drivers = income_statements[0]["drivers"]
        
        # Cost of equity = Risk-free rate + Beta × Market risk premium
        cost_of_equity = drivers["risk_free_rate"] + drivers["beta"] * drivers["market_risk_premium"]
        
        # After-tax cost of debt
        cost_of_debt = drivers["cost_of_debt"] * (1 - drivers["tax_rate"])
        
        # Market values (simplified using book values)
        market_value_debt = 5230000000  # Current debt
        market_value_equity = 800000000000  # Approximate market cap
        total_capital = market_value_debt + market_value_equity
        
        weight_debt = market_value_debt / total_capital
        weight_equity = market_value_equity / total_capital
        
        wacc = weight_equity * cost_of_equity + weight_debt * cost_of_debt
        
        # Calculate free cash flows for 10 years
        free_cash_flows = []
        
        for income_stmt in income_statements:
            # Calculate FCF = Operating Income × (1 - Tax Rate) + Depreciation - CapEx - ΔWorking Capital
            after_tax_operating_income = income_stmt["operating_income"] * (1 - drivers["tax_rate"])
            
            # Depreciation (assume 12% of automotive revenue for simplicity)
            depreciation = income_stmt["automotive_revenue"] * 0.12
            
            # CapEx (based on driver)
            capex = income_stmt["automotive_revenue"] * drivers["capex_rate"]
            
            # Working capital change (simplified)
            wc_change = income_stmt["total_revenue"] * 0.01  # Assume 1% of revenue WC growth
            
            fcf = after_tax_operating_income + depreciation - capex - wc_change
            free_cash_flows.append(fcf)
        
        # Terminal value calculation (using 2033 as terminal year)
        terminal_growth_rate = 0.025  # 2.5% perpetual growth
        terminal_fcf = free_cash_flows[-1] * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (wacc - terminal_growth_rate)
        
        # Present value calculations
        present_value_cash_flows = 0
        for i, fcf in enumerate(free_cash_flows):
            pv_fcf = fcf / ((1 + wacc) ** (i + 1))
            present_value_cash_flows += pv_fcf
        
        present_value_terminal = terminal_value / ((1 + wacc) ** len(free_cash_flows))
        
        # Enterprise and equity value
        enterprise_value = present_value_cash_flows + present_value_terminal
        net_cash = 28000000000  # Approximate net cash position
        equity_value = enterprise_value + net_cash
        
        shares_outstanding = income_statements[-1]["shares_outstanding"]
        price_per_share = equity_value / shares_outstanding
        
        # Enhanced sensitivity analysis (5x5 matrix)
        sensitivity_growth_rates = [0.015, 0.020, 0.025, 0.030, 0.035]
        sensitivity_wacc_rates = [wacc - 0.01, wacc - 0.005, wacc, wacc + 0.005, wacc + 0.01]
        
        sensitivity_matrix = []
        for growth in sensitivity_growth_rates:
            row = []
            for discount_rate in sensitivity_wacc_rates:
                if discount_rate <= growth:
                    row.append(0)
                else:
                    sens_terminal_fcf = free_cash_flows[-1] * (1 + growth)
                    sens_terminal_value = sens_terminal_fcf / (discount_rate - growth)
                    sens_pv_terminal = sens_terminal_value / ((1 + discount_rate) ** len(free_cash_flows))
                    
                    sens_pv_cf = 0
                    for i, fcf in enumerate(free_cash_flows):
                        sens_pv_cf += fcf / ((1 + discount_rate) ** (i + 1))
                    
                    sens_enterprise_value = sens_pv_cf + sens_pv_terminal
                    sens_equity_value = sens_enterprise_value + net_cash
                    sens_price = sens_equity_value / shares_outstanding
                    row.append(sens_price)
            sensitivity_matrix.append(row)
        
        return {
            "scenario": scenario.value,
            "cost_of_equity": float(cost_of_equity),
            "cost_of_debt": float(cost_of_debt),
            "wacc": float(wacc),
            "projected_free_cash_flows": [float(fcf) for fcf in free_cash_flows],
            "terminal_growth_rate": float(terminal_growth_rate),
            "terminal_value": float(terminal_value),
            "present_value_terminal": float(present_value_terminal),
            "present_value_cash_flows": float(present_value_cash_flows),
            "enterprise_value": float(enterprise_value),
            "net_cash": float(net_cash),
            "equity_value": float(equity_value),
            "shares_outstanding": float(shares_outstanding),
            "price_per_share": float(price_per_share),
            "sensitivity_growth_rates": [float(x) for x in sensitivity_growth_rates],
            "sensitivity_wacc_rates": [float(x) for x in sensitivity_wacc_rates],
            "sensitivity_matrix": [[float(x) for x in row] for row in sensitivity_matrix]
        }
    
    def build_enhanced_financial_model(self, scenario: ScenarioType, forecast_years: List[int] = None) -> Dict:
        """
        Build enhanced 10-year financial model with driver-based calculations
        """
        if forecast_years is None:
            forecast_years = list(range(2024, 2034))  # 10-year forecast
        
        # Calculate enhanced income statements for all years
        income_statements = []
        for year in forecast_years:
            income_stmt = self.calculate_enhanced_income_statement(scenario, year)
            income_statements.append(income_stmt)
        
        # Calculate enhanced DCF valuation
        dcf_valuation = self.calculate_enhanced_dcf_valuation(scenario, income_statements)
        
        return {
            "scenario": scenario.value,
            "model_name": "Enhanced Tesla Financial Model with Driver-Based Forecasting",
            "forecast_years": [int(year) for year in forecast_years],
            "income_statements": income_statements,
            "dcf_valuation": dcf_valuation,
            "model_features": [
                "Vehicle model granularity",
                "Driver-based revenue forecasting", 
                "10-year forecast period",
                "Business segment analysis",
                "Enhanced DCF with sensitivity analysis"
            ]
        }