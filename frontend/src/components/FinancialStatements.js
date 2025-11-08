import { formatCompactNumber } from '../lib/utils';
import React, { useState } from 'react';
import CrossStatementSimulation from './CrossStatementSimulation';
import MetricForecasting from './MetricForecasting';
import DCFValuation from './DCFValuation';
import ScenarioComparison from './ScenarioComparison';

const FinancialStatements = ({ scenario, model, generateModel, loading, models = {}, generateAllScenarios = () => {} }) => {
  const [activeStatement, setActiveStatement] = useState('income');
  const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
  const API = `${BACKEND_URL}/api`;

  const years = [2025, 2026, 2027, 2028, 2029];

  const formatCurrency = (value) => `$${formatCompactNumber(value)}`;

  const formatPercent = (value) => {
    if (!value) return '0.0%';
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatNumber = (value) => formatCompactNumber(value);


  if (!model) {
    return (
      <div className="text-center py-12">
        <div className="text-gray-600 mb-4">No financial model generated for {scenario} scenario</div>
        <button
          onClick={() => generateModel(scenario)}
          disabled={loading}
          className="bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white px-6 py-3 rounded-lg font-medium transition-colors"
        >
          {loading ? 'Generating...' : 'Generate Financial Model'}
        </button>
      </div>
    );
  }

  const statements = [
    { id: 'income', name: 'Income Statement', data: model.income_statements },
    { id: 'balance', name: 'Balance Sheet', data: model.balance_sheets },
    { id: 'cashflow', name: 'Cash Flow', data: model.cash_flow_statements },
    { id: 'cross-simulation', name: 'Cross-Statement Simulation', data: null },
    { id: 'metric-forecasting', name: 'Metric Forecasting', data: null },
    { id: 'dcf-valuation', name: 'DCF Valuation', data: null },
    { id: 'scenario-comparison', name: 'Scenario Comparison', data: null }
  ];



  const renderIncomeStatement = () => (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Income Statement
            </th>
            {years.map(year => (
              <th key={year} className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                {year}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          <tr className="bg-blue-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Total Revenue</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-blue-600">
                {formatCurrency(stmt.total_revenue)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Automotive Revenue</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(stmt.automotive_revenue)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Services Revenue</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(stmt.services_revenue)}
              </td>
            ))}
          </tr>
          <tr className="bg-red-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Total COGS</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-red-600">
                ({formatCurrency(stmt.total_cogs)})
              </td>
            ))}
          </tr>
          <tr className="bg-green-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Gross Profit</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-green-600">
                {formatCurrency(stmt.total_gross_profit)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Gross Margin</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatPercent(stmt.gross_margin)}
              </td>
            ))}
          </tr>
          <tr className="bg-gray-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Operating Expenses</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-600">
                ({formatCurrency(stmt.total_operating_expenses)})
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• R&D</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                ({formatCurrency(stmt.research_development)})
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• SG&A</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                ({formatCurrency(stmt.selling_general_admin)})
              </td>
            ))}
          </tr>
          <tr className="bg-purple-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Operating Income</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-purple-600">
                {formatCurrency(stmt.operating_income)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Operating Margin</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatPercent(stmt.operating_margin)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Interest Income</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(stmt.interest_income)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Interest Expense</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                ({formatCurrency(stmt.interest_expense)})
              </td>
            ))}
          </tr>
          <tr className="bg-gray-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Pre-Tax Income</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold text-gray-900">
                {formatCurrency(stmt.pretax_income)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Income Tax Expense</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                ({formatCurrency(stmt.income_tax_expense)})
              </td>
            ))}
          </tr>
          <tr className="bg-yellow-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Net Income</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-yellow-700">
                {formatCurrency(stmt.net_income)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Net Margin</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatPercent(stmt.net_margin)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">EPS</td>
            {model.income_statements.map((stmt, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                ${stmt.earnings_per_share?.toFixed(2)}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );

  const renderBalanceSheet = () => (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Balance Sheet
            </th>
            {years.map(year => (
              <th key={year} className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                {year}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          <tr className="bg-blue-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">ASSETS</td>
            <td colSpan={5}></td>
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Current Assets</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold">
                {formatCurrency(bs.total_current_assets)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Cash & Equivalents</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.cash_and_equivalents)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Accounts Receivable</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.accounts_receivable)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Inventory</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.inventory)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Prepaid Expenses</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.prepaid_expenses)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Other Current Assets</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.other_current_assets)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Non-Current Assets</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold">
                {formatCurrency(bs.total_non_current_assets)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Net PP&E</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.net_ppe)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Intangible Assets</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.intangible_assets)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Other Non-Current Assets</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.other_non_current_assets)}
              </td>
            ))}
          </tr>
          <tr className="bg-green-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Total Assets</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-green-600">
                {formatCurrency(bs.total_assets)}
              </td>
            ))}
          </tr>
          <tr className="bg-red-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">LIABILITIES</td>
            <td colSpan={5}></td>
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Current Liabilities</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold">
                {formatCurrency(bs.total_current_liabilities)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Accounts Payable</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.accounts_payable)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Accrued Liabilities</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.accrued_liabilities)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Current Portion of Debt</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.current_portion_debt)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Other Current Liabilities</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.other_current_liabilities)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Non-Current Liabilities</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold">
                {formatCurrency(bs.total_non_current_liabilities)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Long-Term Debt</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.long_term_debt)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Other Non-Current Liabilities</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.other_non_current_liabilities)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Total Liabilities</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold">
                {formatCurrency(bs.total_liabilities)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">Shareholders' Equity</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-semibold">
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">• Retained Earnings</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(bs.retained_earnings)}
              </td>
            ))}
          </tr>
          <tr className="bg-purple-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Total Shareholders' Equity</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-purple-600">
                {formatCurrency(bs.total_shareholders_equity)}
              </td>
            ))}
          </tr>
          <tr className="bg-green-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Total Liabilities & Equity</td>
            {model.balance_sheets.map((bs, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-green-600">
                {formatCurrency(bs.total_liab_and_equity)}
              </td>
            ))}
          </tr>
          <tr className="bg-yellow-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Balance Difference</td>
            {model.balance_sheets.map((bs, idx) => {
              const diff = bs.total_assets - bs.total_liab_and_equity;
              return (
                <td key={idx} className={`px-6 py-4 whitespace-nowrap text-sm text-right font-bold ${diff !== 0 ? 'text-red-600' : 'text-green-600'}`}>
                  {formatCurrency(diff)}
                </td>
              );
            })}
          </tr>
        </tbody>
      </table>
    </div>
  );

  const renderCashFlow = () => (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
              Cash Flow Statement
            </th>
            {years.map(year => (
              <th key={year} className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                {year}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          <tr className="bg-blue-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">OPERATING ACTIVITIES</td>
            <td colSpan={5}></td>
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Net Income</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(cf.net_income)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Depreciation & Amortization</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(cf.depreciation_amortization)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Accounts Receivable</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(cf.change_accounts_receivable)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Inventory</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(cf.change_inventory)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Accounts Payable</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(cf.change_accounts_payable)}
              </td>
            ))}
          </tr>
          <tr className="bg-green-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Operating Cash Flow</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-green-600">
                {formatCurrency(cf.operating_cash_flow)}
              </td>
            ))}
          </tr>
          <tr className="bg-blue-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">INVESTING ACTIVITIES</td>
            <td colSpan={5}></td>
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Capital Expenditures (CapEx)</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                ({formatCurrency(Math.abs(cf.capital_expenditures))})
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Acquisitions</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                ({formatCurrency(Math.abs(cf.acquisitions))})
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Investments</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                ({formatCurrency(Math.abs(cf.investments))})
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Other Investing Activities</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(cf.other_investing_activities)}
              </td>
            ))}
          </tr>
          <tr className="bg-red-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Investing Cash Flow</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-red-600">
                {formatCurrency(cf.investing_cash_flow)}
              </td>
            ))}
          </tr>
          <tr className="bg-yellow-50 border-t-2 border-yellow-400">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Free Cash Flow (OCF - CapEx)</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-yellow-700">
                {cf.free_cash_flow !== undefined && cf.free_cash_flow !== null 
                  ? formatCurrency(cf.free_cash_flow)
                  : formatCurrency(cf.operating_cash_flow - Math.abs(cf.capital_expenditures))}
              </td>
            ))}
          </tr>
          <tr className="bg-blue-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">FINANCING ACTIVITIES</td>
            <td colSpan={5}></td>
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Debt Proceeds</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(cf.debt_proceeds)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Debt Repayments</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                ({formatCurrency(Math.abs(cf.debt_repayments))})
              </td>
            ))}
          </tr>
          <tr className="bg-purple-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-bold text-gray-900">Financing Cash Flow</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right font-bold text-purple-600">
                {formatCurrency(cf.financing_cash_flow)}
              </td>
            ))}
          </tr>
          <tr className="bg-gray-50">
            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">SUMMARY</td>
            <td colSpan={5}></td>
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Net Change in Cash</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(cf.net_change_cash)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Beginning Cash</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(cf.beginning_cash)}
              </td>
            ))}
          </tr>
          <tr>
            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 pl-8">Ending Cash</td>
            {model.cash_flow_statements.map((cf, idx) => (
              <td key={idx} className="px-6 py-4 whitespace-nowrap text-sm text-right text-gray-700">
                {formatCurrency(cf.ending_cash)}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>
  );

  return (
    <div>
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-lg font-semibold text-gray-900">
          Financial Statements - {scenario.charAt(0).toUpperCase() + scenario.slice(1)} Case
        </h3>
      </div>

      {/* Statement Navigation */}
      <div className="flex justify-between items-center border-b border-gray-200 mb-6">
        <div className="flex">
        {statements.map((statement) => (
          <button
            key={statement.id}
            onClick={() => setActiveStatement(statement.id)}
            className={`px-4 py-2 font-medium text-sm border-b-2 ${
              activeStatement === statement.id
                ? 'border-red-500 text-red-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {statement.name}
          </button>
        ))}
        </div>
        
      </div>

      {/* Statement Content */}
      <div className="bg-white rounded-lg border shadow-sm">
        {activeStatement === 'income' && renderIncomeStatement()}
        {activeStatement === 'balance' && renderBalanceSheet()}
        {activeStatement === 'cashflow' && renderCashFlow()}
        {activeStatement === 'cross-simulation' && (
          <div className="p-6">
            <CrossStatementSimulation scenario={scenario} />
          </div>
        )}
        {activeStatement === 'metric-forecasting' && (
          <div className="p-6">
            <MetricForecasting scenario={scenario} />
          </div>
        )}
        {activeStatement === 'dcf-valuation' && (
          <div className="p-6">
            <DCFValuation scenario={scenario} model={model} generateModel={generateModel} loading={loading} />
          </div>
        )}
        {activeStatement === 'scenario-comparison' && (
          <div className="p-6">
            <ScenarioComparison models={models} generateAllScenarios={generateAllScenarios} loading={loading} />
          </div>
        )}
      </div>
    </div>
  );
};

export default FinancialStatements;