from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime
import uuid
from enum import Enum

class ScenarioType(str, Enum):
    BEST = "best"
    BASE = "base"
    WORST = "worst"

class TeslaAssumptions(BaseModel):
    """Core Tesla financial assumptions and drivers"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario: ScenarioType
    year: int
    
    # Production & Sales Assumptions
    total_deliveries: int  # Annual vehicle deliveries
    average_selling_price: float  # Average ASP across all models
    automotive_revenue_growth: float  # Y/Y growth rate
    services_revenue_growth: float  # Energy & Services growth
    
    # Cost Assumptions
    gross_margin_automotive: float  # Automotive gross margin %
    gross_margin_services: float  # Services gross margin %
    cogs_inflation_rate: float  # Cost inflation rate
    
    # Operating Expenses
    rd_as_percent_revenue: float  # R&D as % of revenue
    sga_as_percent_revenue: float  # SG&A as % of revenue
    
    # Working Capital Assumptions
    days_sales_outstanding: int  # DSO in days
    days_inventory_outstanding: int  # DIO in days
    days_payable_outstanding: int  # DPO in days
    
    # Capital Expenditure
    capex_as_percent_revenue: float  # CapEx as % of revenue
    depreciation_rate: float  # Annual depreciation rate
    
    # Financial Assumptions
    tax_rate: float  # Effective tax rate
    interest_rate_on_debt: float  # Cost of debt
    
    # WACC Components
    risk_free_rate: float  # 10-year Treasury yield
    market_risk_premium: float  # Equity risk premium
    beta: float  # Tesla's equity beta
    debt_to_equity_ratio: float  # Target D/E ratio
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class IncomeStatement(BaseModel):
    """Tesla Income Statement"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario: ScenarioType
    year: int
    
    # Revenue
    automotive_revenue: float
    services_revenue: float  # Energy & Services
    total_revenue: float
    
    # Cost of Goods Sold
    automotive_cogs: float
    services_cogs: float
    total_cogs: float
    
    # Gross Profit
    automotive_gross_profit: float
    services_gross_profit: float
    total_gross_profit: float
    gross_margin: float
    
    # Operating Expenses
    research_development: float
    selling_general_admin: float
    total_operating_expenses: float
    
    # Operating Income
    operating_income: float
    operating_margin: float
    
    # Other Income/Expenses
    interest_income: float
    interest_expense: float
    other_income: float
    
    # Pre-tax Income
    pretax_income: float
    
    # Taxes
    income_tax_expense: float
    effective_tax_rate: float
    
    # Net Income
    net_income: float
    net_margin: float
    
    # Per Share Data
    shares_outstanding: float
    earnings_per_share: float
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class BalanceSheet(BaseModel):
    """Tesla Balance Sheet"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario: ScenarioType
    year: int
    
    # Current Assets
    cash_and_equivalents: float
    accounts_receivable: float
    inventory: float
    prepaid_expenses: float
    other_current_assets: float
    total_current_assets: float
    
    # Non-Current Assets
    property_plant_equipment: float
    accumulated_depreciation: float
    net_ppe: float
    intangible_assets: float
    other_non_current_assets: float
    total_non_current_assets: float
    
    # Total Assets
    total_assets: float
    
    # Current Liabilities
    accounts_payable: float
    accrued_liabilities: float
    current_portion_debt: float
    other_current_liabilities: float
    total_current_liabilities: float
    
    # Non-Current Liabilities
    long_term_debt: float
    other_non_current_liabilities: float
    total_non_current_liabilities: float
    
    # Total Liabilities
    total_liabilities: float
    
    # Shareholders' Equity
    common_stock: float
    other_equity: float
    retained_earnings: float
    total_shareholders_equity: float
    
    # Total Liabilities & Equity
    total_liab_and_equity: float
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class CashFlowStatement(BaseModel):
    """Tesla Cash Flow Statement"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario: ScenarioType
    year: int
    
    # Operating Cash Flow
    net_income: float
    depreciation_amortization: float
    stock_based_compensation: float
    change_accounts_receivable: float
    change_inventory: float
    change_accounts_payable: float
    change_other_working_capital: float
    other_operating_activities: float
    operating_cash_flow: float
    
    # Investing Cash Flow
    capital_expenditures: float
    acquisitions: float
    investments: float
    other_investing_activities: float
    investing_cash_flow: float
    
    # Free Cash Flow (OCF - CapEx)
    free_cash_flow: float
    
    # Financing Cash Flow
    debt_proceeds: float
    debt_repayments: float
    equity_proceeds: float
    share_repurchases: float
    dividends_paid: float
    other_financing_activities: float
    financing_cash_flow: float
    
    # Net Change in Cash
    net_change_cash: float
    beginning_cash: float
    ending_cash: float
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class DCFValuation(BaseModel):
    """DCF Valuation Results"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario: ScenarioType
    
    # WACC Components
    cost_of_equity: float
    cost_of_debt: float
    wacc: float
    
    # Cash Flow Projections (5 years)
    projected_free_cash_flows: List[float]
    
    # Terminal Value
    terminal_growth_rate: float
    terminal_value: float
    present_value_terminal: float
    
    # Valuation
    present_value_cash_flows: float
    enterprise_value: float
    net_cash: float
    equity_value: float
    shares_outstanding: float
    price_per_share: float
    
    # Sensitivity Analysis
    sensitivity_growth_rates: List[float]
    sensitivity_wacc_rates: List[float]
    sensitivity_matrix: List[List[float]]
    
    created_at: datetime = Field(default_factory=datetime.utcnow)

class FinancialModel(BaseModel):
    """Complete Tesla Financial Model"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scenario: ScenarioType
    model_name: str = "Tesla 5-Year Financial Model"
    
    # Assumptions
    assumptions: List[TeslaAssumptions]
    
    # Financial Statements (5 years: 2025-2029)
    income_statements: List[IncomeStatement]
    balance_sheets: List[BalanceSheet] 
    cash_flow_statements: List[CashFlowStatement]
    
    # Valuation
    dcf_valuation: DCFValuation
    
    # Model Metadata
    base_year: int = 2024
    forecast_years: List[int] = [2025, 2026, 2027, 2028, 2029]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

class ModelInput(BaseModel):
    """Input for real-time model updates"""
    scenario: ScenarioType
    year: int
    field_name: str
    field_value: float

class ModelResponse(BaseModel):
    """Response for model calculations"""
    success: bool
    message: str
    financial_model: Optional[FinancialModel] = None
    updated_statements: Optional[Dict] = None