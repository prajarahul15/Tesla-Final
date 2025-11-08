"""
Tesla Financial Calculator
Comprehensive financial modeling and DCF valuation engine
"""

import math
from typing import List, Dict, Tuple
from models.financial_models import *
from data.tesla_data import TESLA_BASE_YEAR_DATA, get_tesla_assumptions

class TeslaFinancialCalculator:
    """Main calculator for Tesla financial modeling"""
    
    def __init__(self):
        self.base_year_data = TESLA_BASE_YEAR_DATA
        
    def calculate_income_statement(self, assumptions: TeslaAssumptions, 
                                 previous_year_data: Dict = None) -> IncomeStatement:
        """Calculate complete income statement for given year and scenario"""
        
        # Revenue Calculations
        if previous_year_data:
            prev_auto_revenue = previous_year_data.get("automotive_revenue", self.base_year_data["automotive_revenue"])
            prev_services_revenue = previous_year_data.get("services_revenue", self.base_year_data["services_revenue"])
        else:
            prev_auto_revenue = self.base_year_data["automotive_revenue"]
            prev_services_revenue = self.base_year_data["services_revenue"]
        
        # Automotive Revenue = Previous Year * (1 + growth rate)
        automotive_revenue = prev_auto_revenue * (1 + assumptions.automotive_revenue_growth)
        
        # Services Revenue = Previous Year * (1 + services growth)
        services_revenue = prev_services_revenue * (1 + assumptions.services_revenue_growth)
        
        total_revenue = automotive_revenue + services_revenue
        
        # Cost of Goods Sold
        automotive_cogs = automotive_revenue * (1 - assumptions.gross_margin_automotive)
        services_cogs = services_revenue * (1 - assumptions.gross_margin_services)
        total_cogs = automotive_cogs + services_cogs
        
        # Gross Profit
        automotive_gross_profit = automotive_revenue - automotive_cogs
        services_gross_profit = services_revenue - services_cogs
        total_gross_profit = automotive_gross_profit + services_gross_profit
        gross_margin = total_gross_profit / total_revenue
        
        # Operating Expenses
        research_development = total_revenue * assumptions.rd_as_percent_revenue
        selling_general_admin = total_revenue * assumptions.sga_as_percent_revenue
        total_operating_expenses = research_development + selling_general_admin
        
        # Operating Income
        operating_income = total_gross_profit - total_operating_expenses
        operating_margin = operating_income / total_revenue
        
        # Other Income/Expenses
        interest_income = 0.005 * total_revenue  # Interest income ~0.5% of revenue
        
        interest_expense = assumptions.interest_rate_on_debt * 9570000000  # Interest on debt
        
        # Pre-tax Income
        pretax_income = operating_income + interest_income - interest_expense
        
        # Taxes
        if pretax_income > 0:
            income_tax_expense = pretax_income * assumptions.tax_rate
            effective_tax_rate = assumptions.tax_rate
        else:
            income_tax_expense = 0
            effective_tax_rate = 0
        
        # Net Income
        net_income = pretax_income - income_tax_expense
        net_margin = net_income / total_revenue if total_revenue > 0 else 0
        
        # Per Share Data
        shares_outstanding = 3178000000  # Relatively stable
        earnings_per_share = net_income / shares_outstanding
        
        # DEBUG: Print interest_income just before returning
        print(f"[RETURN] Year {assumptions.year}: Returning IncomeStatement with interest_income = ${interest_income/1e6:.1f}M")
        
        return IncomeStatement(
            scenario=assumptions.scenario,
            year=assumptions.year,
            automotive_revenue=automotive_revenue,
            services_revenue=services_revenue,
            total_revenue=total_revenue,
            automotive_cogs=automotive_cogs,
            services_cogs=services_cogs,
            total_cogs=total_cogs,
            automotive_gross_profit=automotive_gross_profit,
            services_gross_profit=services_gross_profit,
            total_gross_profit=total_gross_profit,
            gross_margin=gross_margin,
            research_development=research_development,
            selling_general_admin=selling_general_admin,
            total_operating_expenses=total_operating_expenses,
            operating_income=operating_income,
            operating_margin=operating_margin,
            interest_income=interest_income,
            interest_expense=interest_expense,
            other_income=0.0,
            pretax_income=pretax_income,
            income_tax_expense=income_tax_expense,
            effective_tax_rate=effective_tax_rate,
            net_income=net_income,
            net_margin=net_margin,
            shares_outstanding=shares_outstanding,
            earnings_per_share=earnings_per_share
        )
    
    def calculate_balance_sheet(self, assumptions: TeslaAssumptions, 
                              income_stmt: IncomeStatement,
                              previous_balance_sheet: BalanceSheet = None,
                              cash_flow_stmt: CashFlowStatement = None) -> BalanceSheet:
        """Calculate balance sheet with working capital and asset projections"""
        
        # Starting values (from previous year or base year)
        if previous_balance_sheet:
            prev_cash = previous_balance_sheet.cash_and_equivalents
            prev_ppe = previous_balance_sheet.net_ppe
            prev_debt = previous_balance_sheet.long_term_debt
            prev_retained_earnings = previous_balance_sheet.retained_earnings
        else:
            prev_cash = self.base_year_data["cash_and_equivalents"]
            prev_ppe = 50000000000  # Estimated base year net PPE
            prev_debt = self.base_year_data["total_debt"]
            prev_retained_earnings = 20000000000  # Estimated base retained earnings
        
        # Current Assets
        # Cash - updated from cash flow statement if available
        if cash_flow_stmt:
            cash_and_equivalents = cash_flow_stmt.ending_cash
        else:
            cash_and_equivalents = prev_cash
        
        # Accounts Receivable = (DSO / 365) * Revenue
        accounts_receivable = (assumptions.days_sales_outstanding / 365) * income_stmt.total_revenue
        
        # Inventory = (DIO / 365) * COGS
        inventory = (assumptions.days_inventory_outstanding / 365) * income_stmt.total_cogs
        
        prepaid_expenses = income_stmt.total_revenue * 0.01  # 1% of revenue
        other_current_assets = income_stmt.total_revenue * 0.02  # 2% of revenue
        total_current_assets = (cash_and_equivalents + accounts_receivable + 
                              inventory + prepaid_expenses + other_current_assets)
        
        # Non-Current Assets
        # CapEx and Depreciation
        annual_capex = income_stmt.total_revenue * assumptions.capex_as_percent_revenue
        annual_depreciation = prev_ppe * assumptions.depreciation_rate
        
        # Net PPE = Previous PPE + CapEx - Depreciation
        net_ppe = prev_ppe + annual_capex - annual_depreciation
        accumulated_depreciation = annual_depreciation  # Simplified - would track cumulative
        property_plant_equipment = net_ppe + accumulated_depreciation
        
        intangible_assets = 2000000000  # Relatively stable
        other_non_current_assets = 5000000000  # Investments, etc.
        total_non_current_assets = net_ppe + intangible_assets + other_non_current_assets
        
        # Total Assets
        total_assets = total_current_assets + total_non_current_assets
        
        # Current Liabilities
        # Accounts Payable = (DPO / 365) * COGS
        accounts_payable = (assumptions.days_payable_outstanding / 365) * income_stmt.total_cogs
        
        accrued_liabilities = income_stmt.total_revenue * 0.03  # 3% of revenue
        current_portion_debt = prev_debt * 0.1  # 10% of debt is current
        other_current_liabilities = income_stmt.total_revenue * 0.02  # 2% of revenue
        total_current_liabilities = (accounts_payable + accrued_liabilities + 
                                   current_portion_debt + other_current_liabilities)
        
        # Non-Current Liabilities
        long_term_debt = prev_debt - current_portion_debt  # Remaining debt
        other_non_current_liabilities = 3000000000  # Relatively stable
        total_non_current_liabilities = long_term_debt + other_non_current_liabilities
        
        # Total Liabilities
        total_liabilities = total_current_liabilities + total_non_current_liabilities
        
        # Shareholders' Equity
        # Retained Earnings = Previous Retained Earnings + Net Income
        retained_earnings = prev_retained_earnings + income_stmt.net_income
        
        # Total Shareholders' Equity = Common Stock + Other Equity + Retained Earnings
        common_stock = 0.0  # Simplified - no new stock issuance
        other_equity = 0.0  # Simplified
        total_shareholders_equity = common_stock + other_equity + retained_earnings
        
        # Balance Check
        total_liab_and_equity = total_liabilities + total_shareholders_equity
        
        return BalanceSheet(
            scenario=assumptions.scenario,
            year=assumptions.year,
            cash_and_equivalents=cash_and_equivalents,
            accounts_receivable=accounts_receivable,
            inventory=inventory,
            prepaid_expenses=prepaid_expenses,
            other_current_assets=other_current_assets,
            total_current_assets=total_current_assets,
            property_plant_equipment=property_plant_equipment,
            accumulated_depreciation=accumulated_depreciation,
            net_ppe=net_ppe,
            intangible_assets=intangible_assets,
            other_non_current_assets=other_non_current_assets,
            total_non_current_assets=total_non_current_assets,
            total_assets=total_assets,
            accounts_payable=accounts_payable,
            accrued_liabilities=accrued_liabilities,
            current_portion_debt=current_portion_debt,
            other_current_liabilities=other_current_liabilities,
            total_current_liabilities=total_current_liabilities,
            long_term_debt=long_term_debt,
            other_non_current_liabilities=other_non_current_liabilities,
            total_non_current_liabilities=total_non_current_liabilities,
            total_liabilities=total_liabilities,
            common_stock=common_stock,
            other_equity=other_equity,
            retained_earnings=retained_earnings,
            total_shareholders_equity=total_shareholders_equity,
            total_liab_and_equity=total_liab_and_equity
        )
    
    def calculate_cash_flow_statement(self, assumptions: TeslaAssumptions,
                                    income_stmt: IncomeStatement,
                                    current_balance_sheet: BalanceSheet,
                                    previous_balance_sheet: BalanceSheet = None) -> CashFlowStatement:
        """Calculate cash flow statement using indirect method"""
        
        # Starting cash
        if previous_balance_sheet:
            beginning_cash = previous_balance_sheet.cash_and_equivalents
        else:
            beginning_cash = self.base_year_data["cash_and_equivalents"]
        
        # Operating Cash Flow
        net_income = income_stmt.net_income
        
        # Non-cash items
        annual_capex = income_stmt.total_revenue * assumptions.capex_as_percent_revenue
        depreciation_amortization = (previous_balance_sheet.net_ppe if previous_balance_sheet 
                                   else 50000000000) * assumptions.depreciation_rate
        
        # Working Capital Items (shown as absolute values with cash flow impact sign)
        # AR = -(DSO/365) × Revenue (negative because it's a use of cash)
        # Inventory = -(DIO/365) × COGS (negative because it's a use of cash)
        # AP = (DPO/365) × COGS (positive because it's a source of cash)
        # Note: These are absolute values, not deltas, but signed for cash flow impact
        
        # Always use formula-based absolute values with cash flow signs
        change_accounts_receivable = -current_balance_sheet.accounts_receivable
        change_inventory = -current_balance_sheet.inventory
        change_accounts_payable = current_balance_sheet.accounts_payable
        
        # Operating Cash Flow = Simple sum of all signed values
        operating_cash_flow = (net_income + depreciation_amortization +
                             change_accounts_receivable + change_inventory + 
                             change_accounts_payable)
        
        # Investing Cash Flow
        capital_expenditures = -annual_capex  # Negative cash flow
        acquisitions = 0  # No major acquisitions assumed
        investments = 0  # Net investment activity
        other_investing_activities = 0
        
        investing_cash_flow = (capital_expenditures + acquisitions + 
                             investments + other_investing_activities)
        
        # Financing Cash Flow
        debt_proceeds = 0  # No new debt issuance
        debt_repayments = 0  # No major debt repayments
        
        # Total Financing Cash Flow = Debt Proceeds - Debt Repayments
        financing_cash_flow = debt_proceeds - debt_repayments
        
        # Free Cash Flow = Operating Cash Flow - Capital Expenditures
        # Note: capital_expenditures is negative, so we subtract it (which adds its absolute value)
        free_cash_flow = operating_cash_flow - abs(capital_expenditures)
        print(f"[FCF] Year {assumptions.year}: OCF={operating_cash_flow/1e9:.2f}B, CapEx={capital_expenditures/1e9:.2f}B, FCF={free_cash_flow/1e9:.2f}B")
        
        # Net Change in Cash
        net_change_cash = operating_cash_flow + investing_cash_flow + financing_cash_flow
        ending_cash = beginning_cash + net_change_cash
        
        return CashFlowStatement(
            scenario=assumptions.scenario,
            year=assumptions.year,
            net_income=net_income,
            depreciation_amortization=depreciation_amortization,
            stock_based_compensation=0.0,
            change_accounts_receivable=change_accounts_receivable,
            change_inventory=change_inventory,
            change_accounts_payable=change_accounts_payable,
            change_other_working_capital=0.0,
            other_operating_activities=0.0,
            operating_cash_flow=operating_cash_flow,
            capital_expenditures=capital_expenditures,
            acquisitions=acquisitions,
            investments=investments,
            other_investing_activities=other_investing_activities,
            investing_cash_flow=investing_cash_flow,
            free_cash_flow=free_cash_flow,
            debt_proceeds=debt_proceeds,
            debt_repayments=debt_repayments,
            equity_proceeds=0.0,
            share_repurchases=0.0,
            dividends_paid=0.0,
            other_financing_activities=0.0,
            financing_cash_flow=financing_cash_flow,
            net_change_cash=net_change_cash,
            beginning_cash=beginning_cash,
            ending_cash=ending_cash
        )
    
    def calculate_wacc(self, assumptions: TeslaAssumptions, 
                      current_balance_sheet: BalanceSheet) -> Tuple[float, float, float]:
        """Calculate WACC components and overall WACC"""
        
        # Cost of Equity = Risk-free rate + Beta * Market risk premium
        cost_of_equity = (assumptions.risk_free_rate + 
                         assumptions.beta * assumptions.market_risk_premium)
        
        # After-tax Cost of Debt = Interest rate * (1 - Tax rate)
        cost_of_debt = assumptions.interest_rate_on_debt * (1 - assumptions.tax_rate)
        
        # Market values (simplified - using book values)
        total_debt = current_balance_sheet.long_term_debt + current_balance_sheet.current_portion_debt
        market_value_equity = current_balance_sheet.total_shareholders_equity
        total_capital = total_debt + market_value_equity
        
        # Weights
        weight_debt = total_debt / total_capital if total_capital > 0 else 0
        weight_equity = market_value_equity / total_capital if total_capital > 0 else 1
        
        # WACC = (Weight of Equity * Cost of Equity) + (Weight of Debt * After-tax Cost of Debt)
        wacc = (weight_equity * cost_of_equity) + (weight_debt * cost_of_debt)
        
        return cost_of_equity, cost_of_debt, wacc
    
    def calculate_dcf_valuation(self, scenario: ScenarioType, 
                               cash_flow_statements: List[CashFlowStatement],
                               final_year_assumptions: TeslaAssumptions,
                               final_balance_sheet: BalanceSheet) -> DCFValuation:
        """Calculate DCF valuation with terminal value"""
        
        # Calculate free cash flows (2025-2029)
        # FCF = Operating Cash Flow + Capital Expenditures (CapEx is negative)
        projected_free_cash_flows = [
            cf.operating_cash_flow + cf.capital_expenditures 
            for cf in cash_flow_statements
        ]
        
        # Calculate WACC using final year data
        cost_of_equity, cost_of_debt, wacc = self.calculate_wacc(final_year_assumptions, final_balance_sheet)
        
        # Terminal Value Calculation
        terminal_growth_rate = 0.025  # 2.5% perpetual growth
        final_year_fcf = projected_free_cash_flows[-1]
        terminal_fcf = final_year_fcf * (1 + terminal_growth_rate)
        terminal_value = terminal_fcf / (wacc - terminal_growth_rate)
        
        # Present Value of Terminal Value (discounted back 5 years)
        present_value_terminal = terminal_value / ((1 + wacc) ** 5)
        
        # Present Value of Projected Cash Flows
        present_value_cash_flows = 0
        for i, fcf in enumerate(projected_free_cash_flows):
            pv_fcf = fcf / ((1 + wacc) ** (i + 1))
            present_value_cash_flows += pv_fcf
        
        # Enterprise Value
        enterprise_value = present_value_cash_flows + present_value_terminal
        
        # Equity Value = Enterprise Value + Net Cash
        net_cash = final_balance_sheet.cash_and_equivalents - (
            final_balance_sheet.long_term_debt + final_balance_sheet.current_portion_debt)
        equity_value = enterprise_value + net_cash
        
        # Price per Share
        shares_outstanding = 3178000000  # Tesla shares outstanding
        price_per_share = equity_value / shares_outstanding
        
        # Sensitivity Analysis
        sensitivity_growth_rates = [0.015, 0.020, 0.025, 0.030, 0.035]  # 1.5% to 3.5%
        sensitivity_wacc_rates = [wacc - 0.01, wacc - 0.005, wacc, wacc + 0.005, wacc + 0.01]
        
        sensitivity_matrix = []
        for growth in sensitivity_growth_rates:
            row = []
            for discount_rate in sensitivity_wacc_rates:
                if discount_rate <= growth:
                    row.append(0)  # Invalid (WACC must be > growth rate)
                else:
                    sens_terminal_fcf = final_year_fcf * (1 + growth)
                    sens_terminal_value = sens_terminal_fcf / (discount_rate - growth)
                    sens_pv_terminal = sens_terminal_value / ((1 + discount_rate) ** 5)
                    
                    sens_pv_cf = 0
                    for i, fcf in enumerate(projected_free_cash_flows):
                        sens_pv_cf += fcf / ((1 + discount_rate) ** (i + 1))
                    
                    sens_enterprise_value = sens_pv_cf + sens_pv_terminal
                    sens_equity_value = sens_enterprise_value + net_cash
                    sens_price = sens_equity_value / shares_outstanding
                    row.append(sens_price)
            sensitivity_matrix.append(row)
        
        return DCFValuation(
            scenario=scenario,
            cost_of_equity=cost_of_equity,
            cost_of_debt=cost_of_debt,
            wacc=wacc,
            projected_free_cash_flows=projected_free_cash_flows,
            terminal_growth_rate=terminal_growth_rate,
            terminal_value=terminal_value,
            present_value_terminal=present_value_terminal,
            present_value_cash_flows=present_value_cash_flows,
            enterprise_value=enterprise_value,
            net_cash=net_cash,
            equity_value=equity_value,
            shares_outstanding=shares_outstanding,
            price_per_share=price_per_share,
            sensitivity_growth_rates=sensitivity_growth_rates,
            sensitivity_wacc_rates=sensitivity_wacc_rates,
            sensitivity_matrix=sensitivity_matrix
        )
    
    def build_complete_financial_model(self, scenario: ScenarioType) -> FinancialModel:
        """Build complete 5-year financial model for given scenario"""
        
        # Generate all yearly assumptions for scenario
        all_assumptions = []
        for year in [2025, 2026, 2027, 2028, 2029]:
            assumptions_dict = get_tesla_assumptions(scenario, year)
            assumptions = TeslaAssumptions(**assumptions_dict)
            all_assumptions.append(assumptions)
        
        # Calculate financial statements year by year
        income_statements = []
        balance_sheets = []
        cash_flow_statements = []
        
        previous_income = None
        previous_balance = None
        
        for assumptions in all_assumptions:
            # Income Statement
            prev_data = {}
            if previous_income:
                prev_data = {
                    "automotive_revenue": previous_income.automotive_revenue,
                    "services_revenue": previous_income.services_revenue
                }
            
            income_stmt = self.calculate_income_statement(assumptions, prev_data)
            income_statements.append(income_stmt)
            
            # Balance Sheet (needs income statement)
            balance_sheet = self.calculate_balance_sheet(
                assumptions, income_stmt, previous_balance)
            balance_sheets.append(balance_sheet)
            
            # Cash Flow Statement (needs both current and previous balance sheet)
            cash_flow = self.calculate_cash_flow_statement(
                assumptions, income_stmt, balance_sheet, previous_balance)
            
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
            
            cash_flow_statements.append(cash_flow)
            
            # Set for next iteration
            previous_income = income_stmt
            previous_balance = balance_sheet
        
        # DCF Valuation
        dcf_valuation = self.calculate_dcf_valuation(
            scenario, cash_flow_statements, all_assumptions[-1], balance_sheets[-1])
        
        return FinancialModel(
            scenario=scenario,
            assumptions=all_assumptions,
            income_statements=income_statements,
            balance_sheets=balance_sheets,
            cash_flow_statements=cash_flow_statements,
            dcf_valuation=dcf_valuation
        ) 


