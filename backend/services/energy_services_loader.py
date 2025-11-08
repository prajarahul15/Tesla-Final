"""
Energy & Services Data Loader
Loads monthly data from CSV and provides aggregated summaries by year
"""

import pandas as pd
import os
from typing import Dict, Optional


class EnergyServicesLoader:
    """Load and process Energy & Storage and Services & Other monthly data"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.data = None
        self.load_data()
    
    def load_data(self) -> pd.DataFrame:
        """Load CSV data"""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Energy/Services CSV not found: {self.csv_path}")
        
        self.data = pd.read_csv(self.csv_path)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        
        print(f"✅ Loaded Energy/Services data: {len(self.data)} records from {self.data['year'].min()} to {self.data['year'].max()}")
        
        return self.data
    
    def get_available_years(self) -> list:
        """Get list of available years in dataset"""
        if self.data is None:
            self.load_data()
        return sorted(self.data['year'].unique().tolist())
    
    def get_summary_for_year(self, year: int) -> Dict:
        """
        Get aggregated summary for a specific year
        
        Returns:
            {
                "success": True,
                "year": 2025,
                "energy": {
                    "revenue": 9000000000,
                    "cogs": 6750000000,
                    "gross_profit": 2250000000,
                    "margin": 0.25,
                    "yoy_growth": 0.0588,
                    "cagr": 0.384
                },
                "services": {
                    "revenue": 10500000000,
                    "cogs": 8400000000,
                    "gross_profit": 2100000000,
                    "margin": 0.20,
                    "yoy_growth": -0.0278,
                    "cagr": 0.334
                }
            }
        """
        if self.data is None:
            self.load_data()
        
        # Filter data for the requested year
        year_data = self.data[self.data['year'] == year]
        
        if year_data.empty:
            return {
                "success": False,
                "error": f"No data available for year {year}",
                "available_years": self.get_available_years()
            }
        
        # Aggregate for the year (sum all months)
        energy_revenue = year_data['energy_revenue'].sum() * 1_000_000  # Convert to actual (millions to dollars)
        energy_cogs = year_data['energy_cogs'].sum() * 1_000_000
        services_revenue = year_data['services_revenue'].sum() * 1_000_000
        services_cogs = year_data['services_cogs'].sum() * 1_000_000
        
        # Calculate margins
        energy_margin = ((energy_revenue - energy_cogs) / energy_revenue) if energy_revenue > 0 else 0
        services_margin = ((services_revenue - services_cogs) / services_revenue) if services_revenue > 0 else 0
        
        # Calculate YoY Growth
        energy_yoy = self._calculate_yoy_growth(year, 'energy_revenue')
        services_yoy = self._calculate_yoy_growth(year, 'services_revenue')
        
        # Calculate CAGR (from first year to current year)
        first_year = self.data['year'].min()
        energy_cagr = self._calculate_cagr(first_year, year, 'energy_revenue')
        services_cagr = self._calculate_cagr(first_year, year, 'services_revenue')
        
        return {
            "success": True,
            "year": year,
            "months_count": len(year_data),
            "energy": {
                "revenue": round(energy_revenue, 2),
                "cogs": round(energy_cogs, 2),
                "gross_profit": round(energy_revenue - energy_cogs, 2),
                "margin": round(energy_margin, 4),
                "yoy_growth": round(energy_yoy, 4) if energy_yoy is not None else None,
                "cagr": round(energy_cagr, 4) if energy_cagr is not None else None
            },
            "services": {
                "revenue": round(services_revenue, 2),
                "cogs": round(services_cogs, 2),
                "gross_profit": round(services_revenue - services_cogs, 2),
                "margin": round(services_margin, 4),
                "yoy_growth": round(services_yoy, 4) if services_yoy is not None else None,
                "cagr": round(services_cagr, 4) if services_cagr is not None else None
            }
        }
    
    def _calculate_yoy_growth(self, year: int, metric: str) -> Optional[float]:
        """Calculate Year-over-Year growth"""
        prev_year = year - 1
        
        current_year_data = self.data[self.data['year'] == year]
        prev_year_data = self.data[self.data['year'] == prev_year]
        
        if prev_year_data.empty:
            return None  # No previous year data
        
        current_total = current_year_data[metric].sum()
        prev_total = prev_year_data[metric].sum()
        
        if prev_total == 0:
            return None
        
        yoy_growth = (current_total - prev_total) / prev_total
        return yoy_growth
    
    def _calculate_cagr(self, start_year: int, end_year: int, metric: str) -> Optional[float]:
        """Calculate Compound Annual Growth Rate"""
        if start_year == end_year:
            return None  # Cannot calculate CAGR for single year
        
        start_data = self.data[self.data['year'] == start_year]
        end_data = self.data[self.data['year'] == end_year]
        
        if start_data.empty or end_data.empty:
            return None
        
        start_value = start_data[metric].sum()
        end_value = end_data[metric].sum()
        
        if start_value <= 0:
            return None
        
        years = end_year - start_year
        cagr = (end_value / start_value) ** (1 / years) - 1
        
        return cagr
    
    def get_monthly_data(self, year: int) -> Dict:
        """Get month-by-month breakdown for a specific year"""
        if self.data is None:
            self.load_data()
        
        year_data = self.data[self.data['year'] == year]
        
        if year_data.empty:
            return {
                "success": False,
                "error": f"No data available for year {year}"
            }
        
        # Convert to list of dictionaries
        monthly_records = []
        for _, row in year_data.iterrows():
            monthly_records.append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "month": int(row['month']),
                "energy_revenue": round(row['energy_revenue'] * 1_000_000, 2),
                "energy_cogs": round(row['energy_cogs'] * 1_000_000, 2),
                "services_revenue": round(row['services_revenue'] * 1_000_000, 2),
                "services_cogs": round(row['services_cogs'] * 1_000_000, 2)
            })
        
        return {
            "success": True,
            "year": year,
            "months": monthly_records
        }


# Singleton instance
_energy_services_loader = None


def get_energy_services_loader() -> EnergyServicesLoader:
    """Get or create singleton instance of EnergyServicesLoader"""
    global _energy_services_loader
    
    if _energy_services_loader is None:
        csv_path = os.getenv(
            'ENERGY_SERVICES_CSV',
            'data/Energy_Services_Monthly_2018_2025.csv'
        )
        # If relative path, resolve it relative to this file's directory
        if not os.path.isabs(csv_path):
            backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(backend_dir, csv_path)
        
        _energy_services_loader = EnergyServicesLoader(csv_path)
    
    return _energy_services_loader

