"""
Tesla Historical and Projected Data
Based on latest financial reports and industry analysis
"""

from models.financial_models import TeslaAssumptions, ScenarioType

# Tesla Base Year (2024) Historical Data
TESLA_BASE_YEAR_DATA = {
    "total_deliveries": 1808581,  # 2024 actual deliveries
    "total_revenue": 96773000000,  # $96.77B in 2024
    "automotive_revenue": 82419000000,  # 85% of total revenue
    "services_revenue": 14354000000,  # Energy & Services
    "net_income": 15000000000,  # Approx $15B net income
    "total_assets": 106618000000,  # Total assets
    "cash_and_equivalents": 28700000000,  # Cash position
    "total_debt": 9570000000,  # Total debt
    "shares_outstanding": 3178000000,  # Shares outstanding
}

# Industry and Macro Assumptions (Updated for 2025)
MACRO_ASSUMPTIONS = {
    "global_ev_market_growth": 0.25,  # 25% annual growth
    "inflation_rate": 0.045,  # 4.5% inflation in 2025
    "risk_free_rate": 0.043,  # 10-year Treasury at 4.3%
    "market_risk_premium": 0.06,  # 6% equity risk premium
    "tesla_beta": 2.3,  # Tesla's equity beta
}

def get_tesla_assumptions(scenario: ScenarioType, year: int) -> dict:
    """
    Generate Tesla assumptions for specific scenario and year
    Based on EV market trends and Tesla's strategic position
    """
    
    # Base growth rates by year
    base_delivery_growth = {
        2025: 0.20,  # 20% growth
        2026: 0.18,  # Moderating growth
        2027: 0.15,
        2028: 0.12,
        2029: 0.10
    }
    
    # Scenario adjustments
    scenario_multipliers = {
        ScenarioType.BEST: {
            "delivery_growth_multiplier": 1.3,  # 30% higher growth
            "asp_premium": 1.05,  # 5% ASP premium
            "margin_improvement": 0.02,  # 200bps margin improvement
            "cost_inflation_discount": 0.8  # 20% less cost inflation
        },
        ScenarioType.BASE: {
            "delivery_growth_multiplier": 1.0,
            "asp_premium": 1.0,
            "margin_improvement": 0.0,
            "cost_inflation_discount": 1.0
        },
        ScenarioType.WORST: {
            "delivery_growth_multiplier": 0.7,  # 30% lower growth
            "asp_premium": 0.95,  # 5% ASP discount
            "margin_improvement": -0.02,  # 200bps margin compression
            "cost_inflation_discount": 1.2  # 20% more cost inflation
        }
    }
    
    multiplier = scenario_multipliers[scenario]
    base_growth = base_delivery_growth[year]
    
    # Calculate scenario-specific assumptions
    assumptions = {
        "scenario": scenario,
        "year": year,
        
        # Production & Sales
        "total_deliveries": int(TESLA_BASE_YEAR_DATA["total_deliveries"] * 
                              (1 + base_growth * multiplier["delivery_growth_multiplier"])),
        "average_selling_price": 53500 * multiplier["asp_premium"],  # Base ASP $53.5k
        "automotive_revenue_growth": base_growth * multiplier["delivery_growth_multiplier"],
        "services_revenue_growth": 0.35,  # Energy business growing 35% annually
        
        # Margins
        "gross_margin_automotive": min(0.25, 0.19 + multiplier["margin_improvement"]),  # Base 19%, cap at 25%
        "gross_margin_services": 0.22,  # Services margin
        "cogs_inflation_rate": MACRO_ASSUMPTIONS["inflation_rate"] * multiplier["cost_inflation_discount"],
        
        # Operating Expenses
        "rd_as_percent_revenue": max(0.025, 0.035 - (year - 2025) * 0.002),  # R&D efficiency over time
        "sga_as_percent_revenue": max(0.04, 0.055 - (year - 2025) * 0.003),  # SG&A leverage
        
        # Working Capital
        "days_sales_outstanding": 15,  # Tesla's efficient collection
        "days_inventory_outstanding": 56,  # Lean inventory model
        "days_payable_outstanding": 60,  # Supplier payment terms
        
        # CapEx
        "capex_as_percent_revenue": max(0.06, 0.09 - (year - 2025) * 0.005),  # CapEx efficiency
        "depreciation_rate": 0.12,  # 12% annual depreciation
        
        # Financial
        "tax_rate": 0.21 if scenario == ScenarioType.BEST else 0.25,  # Tax optimization
        "interest_rate_on_debt": 0.05,  # Tesla's low cost of debt
        
        # WACC Components
        "risk_free_rate": MACRO_ASSUMPTIONS["risk_free_rate"],
        "market_risk_premium": MACRO_ASSUMPTIONS["market_risk_premium"],
        "beta": MACRO_ASSUMPTIONS["tesla_beta"],
        "debt_to_equity_ratio": 0.1,  # Tesla's conservative debt policy
    }
    
    return assumptions

# Predefined assumptions for all scenarios and years
def generate_all_tesla_assumptions():
    """Generate complete assumption set for all scenarios and years"""
    all_assumptions = []
    
    for scenario in [ScenarioType.BEST, ScenarioType.BASE, ScenarioType.WORST]:
        for year in [2025, 2026, 2027, 2028, 2029]:
            assumptions_dict = get_tesla_assumptions(scenario, year)
            assumptions = TeslaAssumptions(**assumptions_dict)
            all_assumptions.append(assumptions)
    
    return all_assumptions

# Tesla Peer Comparison Data (for context)
TESLA_PEER_METRICS = {
    "traditional_oems": {
        "average_ev_margin": 0.05,  # Traditional OEMs struggle with EV margins
        "average_pe_ratio": 8.5,
        "average_revenue_growth": 0.02
    },
    "ev_pure_plays": {
        "average_ev_margin": 0.12,
        "average_pe_ratio": 25.0,
        "average_revenue_growth": 0.45
    },
    "tesla_advantages": {
        "vertical_integration_margin_benefit": 0.08,  # 800bps advantage
        "software_margin_premium": 0.15,  # High-margin software/services
        "brand_premium": 0.05  # Premium pricing power
    }
}

# Market Share and Competitive Positioning
TESLA_MARKET_DATA = {
    "global_ev_market_size_2024": 1200000000000,  # $1.2T market
    "tesla_market_share_2024": 0.20,  # 20% global EV market share
    "projected_market_share": {
        2025: 0.18,  # Competition intensifying
        2026: 0.16,
        2027: 0.15,
        2028: 0.14,
        2029: 0.13
    },
    "addressable_markets": {
        "automotive": 3000000000000,  # $3T global auto market
        "energy_storage": 120000000000,  # $120B energy storage
        "autonomous_driving": 800000000000,  # $800B future AV market
    }
} 
