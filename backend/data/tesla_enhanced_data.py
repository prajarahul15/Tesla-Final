"""
Enhanced Tesla Data with Historical Actuals and Driver-Based Forecasting
Extracted from Tesla 2.xlsx professional model
"""

import pandas as pd
from typing import Dict, List
from models.financial_models import ScenarioType

# PHASE 1: Historical Actuals (2020-2023) - Real Tesla Data
TESLA_HISTORICAL_DATA = {
    "historical_years": [2020, 2021, 2022, 2023],
    "forecast_years": [2024, 2025, 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033],
    
    # Historical Revenue Data (in millions)
    "historical_revenue": {
        2020: {"automotive": 27236, "energy": 1994, "services": 2306, "total": 31536},
        2021: {"automotive": 47232, "energy": 2789, "services": 3802, "total": 53823},
        2022: {"automotive": 71462, "energy": 3909, "services": 6091, "total": 81462},
        2023: {"automotive": 82419, "energy": 6035, "services": 8319, "total": 96773}
    },
    
    # Historical Cost Data (in millions)
    "historical_costs": {
        2020: {"automotive_cogs": 20259, "energy_cogs": 1976, "services_cogs": 2671},
        2021: {"automotive_cogs": 33393, "energy_cogs": 2918, "services_cogs": 3906},
        2022: {"automotive_cogs": 51108, "energy_cogs": 3621, "services_cogs": 5880},
        2023: {"automotive_cogs": 66389, "energy_cogs": 4894, "services_cogs": 7830}
    },
    
    # Vehicle Delivery Data (actual numbers)
    "historical_deliveries": {
        2020: {"model_s": 19095, "model_x": 37945, "model_3": 358441, "model_y": 84167, "cybertruck": 0, "semi": 0},
        2021: {"model_s": 14950, "model_x": 9985, "model_3": 476356, "model_y": 434790, "cybertruck": 0, "semi": 0},
        2022: {"model_s": 29256, "model_x": 37449, "model_3": 485984, "model_y": 761162, "cybertruck": 0, "semi": 0},
        2023: {"model_s": 27389.5, "model_x": 41484.5, "model_3": 626938, "model_y": 1112769, "cybertruck": 1163, "semi": 1200}
    },
    
    # Working Capital Metrics (actual)
    "historical_working_capital": {
        2020: {"dso": 21.53, "dio": 45.2, "dpo": 65.8},
        2021: {"dso": 12.80, "dio": 38.9, "dpo": 58.7},
        2022: {"dso": 13.05, "dio": 56.8, "dpo": 67.4},
        2023: {"dso": 13.05, "dio": 50.9, "dpo": 54.2}
    }
}

# PHASE 1: Vehicle Model Breakdown and Driver-Based Assumptions
VEHICLE_MODEL_DATA = {
    "models": {
        "model_s": {
            "name": "Model S",
            "segment": "luxury_sedan",
            "base_asp": 95000,  # Average Selling Price
            "margin_premium": 0.25,  # Higher margin luxury model
            "max_capacity": 50000,  # Annual production capacity
            "growth_trajectory": "mature"
        },
        "model_x": {
            "name": "Model X", 
            "segment": "luxury_suv",
            "base_asp": 105000,
            "margin_premium": 0.22,
            "max_capacity": 80000,
            "growth_trajectory": "mature"
        },
        "model_3": {
            "name": "Model 3",
            "segment": "mass_market_sedan", 
            "base_asp": 42000,
            "margin_premium": 0.18,
            "max_capacity": 800000,
            "growth_trajectory": "stable"
        },
        "model_y": {
            "name": "Model Y",
            "segment": "mass_market_suv",
            "base_asp": 52000,
            "margin_premium": 0.20,
            "max_capacity": 1500000,
            "growth_trajectory": "high_growth"
        },
        "cybertruck": {
            "name": "Cybertruck",
            "segment": "pickup_truck",
            "base_asp": 75000,
            "margin_premium": 0.28,
            "max_capacity": 500000,
            "growth_trajectory": "ramp_up"
        },
        "semi": {
            "name": "Tesla Semi",
            "segment": "commercial",
            "base_asp": 200000,
            "margin_premium": 0.15,
            "max_capacity": 100000,
            "growth_trajectory": "early_stage"
        }
    }
}

# PHASE 1: Driver-Based Forecast Assumptions by Scenario
def get_enhanced_tesla_drivers(scenario: ScenarioType, year: int) -> Dict:
    """
    Enhanced driver-based assumptions with vehicle model granularity
    """
    
    # Base delivery growth assumptions by model and scenario
    delivery_growth_assumptions = {
        ScenarioType.BEST: {
            "model_s": {2024: 0.15, 2025: 0.12, 2026: 0.10, 2027: 0.08, 2028: 0.06, 2029: 0.05},
            "model_x": {2024: 0.18, 2025: 0.15, 2026: 0.12, 2027: 0.10, 2028: 0.08, 2029: 0.06},
            "model_3": {2024: 0.08, 2025: 0.06, 2026: 0.05, 2027: 0.04, 2028: 0.03, 2029: 0.02},
            "model_y": {2024: 0.25, 2025: 0.20, 2026: 0.15, 2027: 0.12, 2028: 0.10, 2029: 0.08},
            "cybertruck": {2024: 8.0, 2025: 2.5, 2026: 1.8, 2027: 1.4, 2028: 1.2, 2029: 1.0},
            "semi": {2024: 50.0, 2025: 5.0, 2026: 3.0, 2027: 2.0, 2028: 1.5, 2029: 1.2}
        },
        ScenarioType.BASE: {
            "model_s": {2024: 0.10, 2025: 0.08, 2026: 0.06, 2027: 0.05, 2028: 0.04, 2029: 0.03},
            "model_x": {2024: 0.12, 2025: 0.10, 2026: 0.08, 2027: 0.06, 2028: 0.05, 2029: 0.04},
            "model_3": {2024: 0.05, 2025: 0.04, 2026: 0.03, 2027: 0.02, 2028: 0.02, 2029: 0.01},
            "model_y": {2024: 0.18, 2025: 0.15, 2026: 0.12, 2027: 0.10, 2028: 0.08, 2029: 0.06},
            "cybertruck": {2024: 5.0, 2025: 2.0, 2026: 1.5, 2027: 1.2, 2028: 1.0, 2029: 0.8},
            "semi": {2024: 20.0, 2025: 3.0, 2026: 2.0, 2027: 1.5, 2028: 1.2, 2029: 1.0}
        },
        ScenarioType.WORST: {
            "model_s": {2024: 0.05, 2025: 0.03, 2026: 0.02, 2027: 0.01, 2028: 0.01, 2029: 0.00},
            "model_x": {2024: 0.06, 2025: 0.04, 2026: 0.03, 2027: 0.02, 2028: 0.01, 2029: 0.01},
            "model_3": {2024: 0.02, 2025: 0.01, 2026: 0.00, 2027: -0.01, 2028: -0.01, 2029: -0.02},
            "model_y": {2024: 0.10, 2025: 0.08, 2026: 0.06, 2027: 0.04, 2028: 0.03, 2029: 0.02},
            "cybertruck": {2024: 3.0, 2025: 1.2, 2026: 0.8, 2027: 0.6, 2028: 0.5, 2029: 0.4},
            "semi": {2024: 10.0, 2025: 1.5, 2026: 1.0, 2027: 0.8, 2028: 0.6, 2029: 0.5}
        }
    }
    
    # ASP (Average Selling Price) assumptions by scenario
    asp_adjustments = {
        ScenarioType.BEST: 1.05,  # 5% premium pricing
        ScenarioType.BASE: 1.00,  # Base pricing
        ScenarioType.WORST: 0.95  # 5% pricing pressure
    }
    
    # Get delivery growth for this year and scenario
    growth_rates = delivery_growth_assumptions[scenario]
    asp_multiplier = asp_adjustments[scenario]
    
    # Calculate projected deliveries based on 2023 base
    base_deliveries_2023 = TESLA_HISTORICAL_DATA["historical_deliveries"][2023]
    projected_deliveries = {}
    
    for model, base_count in base_deliveries_2023.items():
        if year in growth_rates[model]:
            # Compound growth from 2023
            years_from_base = year - 2023
            cumulative_growth = 1.0
            for y in range(2024, year + 1):
                if y in growth_rates[model]:
                    cumulative_growth *= (1 + growth_rates[model][y])
            
            projected_deliveries[model] = max(0, int(base_count * cumulative_growth))
        else:
            projected_deliveries[model] = base_count
    
    # Business segment growth rates
    energy_growth = {
        ScenarioType.BEST: 0.45,
        ScenarioType.BASE: 0.35,
        ScenarioType.WORST: 0.25
    }
    
    services_growth = {
        ScenarioType.BEST: 0.40,
        ScenarioType.BASE: 0.30,
        ScenarioType.WORST: 0.20
    }
    
    return {
        "scenario": scenario,
        "year": year,
        "projected_deliveries": projected_deliveries,
        "asp_multiplier": asp_multiplier,
        "energy_growth_rate": energy_growth[scenario],
        "services_growth_rate": services_growth[scenario],
        
        # Enhanced financial drivers
        "automotive_margin_improvement": {
            ScenarioType.BEST: 0.02,
            ScenarioType.BASE: 0.01,
            ScenarioType.WORST: -0.01
        }[scenario],
        
        # Working capital efficiency
        "dso_target": {
            ScenarioType.BEST: 12.0,
            ScenarioType.BASE: 13.0,
            ScenarioType.WORST: 15.0
        }[scenario],
        
        "dio_target": {
            ScenarioType.BEST: 40.0,
            ScenarioType.BASE: 56.0,
            ScenarioType.WORST: 55.0
        }[scenario],
        
        # CapEx as % of automotive revenue
        "capex_rate": {
            ScenarioType.BEST: 0.08,
            ScenarioType.BASE: 0.10,
            ScenarioType.WORST: 0.12
        }[scenario],
        
        # R&D and SG&A efficiency
        "rd_efficiency": max(0.025, 0.040 - (year - 2024) * 0.002),
        "sga_efficiency": max(0.035, 0.055 - (year - 2024) * 0.003),
        
        # WACC components (market-based from Tesla 2.xlsx)
        "risk_free_rate": 0.039,
        "beta": 2.29,
        "market_risk_premium": 0.05,
        "tax_rate": 0.25,
        "cost_of_debt": 0.053
    }

# PHASE 2: Business Segment Definitions
BUSINESS_SEGMENTS = {
    "automotive": {
        "name": "Automotive",
        "description": "Vehicle sales and automotive services",
        "models": ["model_s", "model_x", "model_3", "model_y", "cybertruck", "semi"],
        "margin_profile": "primary_revenue"
    },
    "energy": {
        "name": "Energy Generation & Storage",
        "description": "Solar panels, energy storage, and grid services",
        "products": ["solar_panels", "powerwall", "megapack", "grid_services"],
        "margin_profile": "high_growth"
    },
    "services": {
        "name": "Services & Other",
        "description": "Supercharging, insurance, software, and other services",
        "products": ["supercharging", "insurance", "fsd", "software", "parts"],
        "margin_profile": "high_margin"
    }
}

# PHASE 3: Price-Volume-Mix Analysis Framework
PRICE_VOLUME_MIX_DATA = {
    "historical_asp_trends": {
        "model_s": {2020: 98000, 2021: 102000, 2022: 108000, 2023: 89000},  # Price cuts in 2023
        "model_x": {2020: 108000, 2021: 112000, 2022: 120000, 2023: 99000},
        "model_3": {2020: 49000, 2021: 52000, 2022: 54000, 2023: 42000},
        "model_y": {2020: 58000, 2021: 60000, 2022: 65000, 2023: 52000}
    },
    
    "volume_price_elasticity": {
        "model_s": -0.8,  # Luxury segment less price sensitive
        "model_x": -0.7,
        "model_3": -1.2,  # Mass market more price sensitive
        "model_y": -1.0,
        "cybertruck": -0.6,  # Unique product, less elastic
        "semi": -0.5  # Commercial, focus on TCO
    }
}

# Bridge Analysis Data Structure
BRIDGE_ANALYSIS_FRAMEWORK = {
    "revenue_bridges": ["volume_effect", "price_effect", "mix_effect", "fx_effect"],
    "margin_bridges": ["volume_leverage", "cost_inflation", "productivity", "mix_shift"],
    "cash_flow_bridges": ["operating_performance", "working_capital", "capex", "financing"]
}