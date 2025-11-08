"""
Advanced Analytics Engine for Professional Dashboard
Handles data analysis, forecasting, and economic variable processing
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AnalyticsEngine:
    """Main analytics engine for dashboard"""
    
    def __init__(self):
        self.sample_data = None
        self.mv_parameters = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare all data files"""
        try:
            from pathlib import Path
            # Use relative paths for deployment compatibility
            backend_dir = Path(__file__).parent.parent
            data_dir = backend_dir / "data"
            
            # Load sample data with basic columns
            sample_data_path = data_dir / "Sample_data_N.csv"
            if not sample_data_path.exists():
                # Fallback to alternative filename if exists
                alt_path = data_dir / "Tesla_Monthly_Model_Production_Delivery_2018_2025.csv"
                sample_data_path = alt_path if alt_path.exists() else sample_data_path
            
            self.sample_data = pd.read_csv(sample_data_path)
            self.sample_data['DATE'] = pd.to_datetime(self.sample_data['DATE'], format='%d-%m-%Y')
            self.sample_data = self.sample_data.sort_values(['Lineup', 'DATE'])
            
            # Load multivariate parameters with economic variables
            mv_params_path = data_dir / "MV Parameter.xlsx"
            if mv_params_path.exists():
                self.mv_parameters = pd.read_excel(mv_params_path)
                self.mv_parameters['DATE'] = pd.to_datetime(self.mv_parameters['DATE'])
                self.mv_parameters = self.mv_parameters.sort_values(['Lineup', 'DATE'])
            else:
                # If MV Parameter.xlsx doesn't exist, use sample_data as fallback
                self.mv_parameters = self.sample_data.copy()
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_data_overview_metrics(self):
        """Calculate comprehensive data overview metrics"""
        if self.sample_data is None:
            return None
            
        metrics = {}
        
        # Overall metrics
        metrics['total_actual'] = int(self.sample_data['Actual'].sum())
        metrics['total_plan'] = int(self.sample_data['Plan'].sum())
        metrics['total_variance'] = float(((metrics['total_actual'] - metrics['total_plan']) / metrics['total_plan'] * 100) if metrics['total_plan'] > 0 else 0)
        
        # Profile level breakdown
        profile_metrics = []
        for profile in self.sample_data['Profile'].unique():
            profile_data = self.sample_data[self.sample_data['Profile'] == profile]
            profile_actual = profile_data['Actual'].sum()
            profile_plan = profile_data['Plan'].sum()
            profile_variance = ((profile_actual - profile_plan) / profile_plan * 100) if profile_plan > 0 else 0
            
            profile_metrics.append({
                'profile': profile,
                'actual': int(profile_actual),
                'plan': int(profile_plan),
                'variance': float(profile_variance),
                'line_items': int(profile_data['Line_Item'].nunique()),
                'records': int(len(profile_data))
            })
        
        metrics['profile_breakdown'] = profile_metrics
        
        # Lineup level breakdown
        lineup_metrics = []
        for lineup in self.sample_data['Lineup'].unique():
            lineup_data = self.sample_data[self.sample_data['Lineup'] == lineup]
            lineup_actual = lineup_data['Actual'].sum()
            lineup_plan = lineup_data['Plan'].sum()
            lineup_variance = ((lineup_actual - lineup_plan) / lineup_plan * 100) if lineup_plan > 0 else 0
            
            # Monthly trends
            monthly_data = lineup_data.groupby(lineup_data['DATE'].dt.to_period('M')).agg({
                'Actual': 'sum',
                'Plan': 'sum'
            }).reset_index()
            monthly_data['DATE'] = monthly_data['DATE'].astype(str)
            
            lineup_metrics.append({
                'lineup': lineup,
                'actual': int(lineup_actual),
                'plan': int(lineup_plan),
                'variance': float(lineup_variance),
                'monthly_trends': monthly_data.to_dict('records'),
                'records': int(len(lineup_data))
            })
        
        metrics['lineup_breakdown'] = lineup_metrics
        
        # Time series data for charts
        monthly_totals = self.sample_data.groupby(self.sample_data['DATE'].dt.to_period('M')).agg({
            'Actual': 'sum',
            'Plan': 'sum'
        }).reset_index()
        monthly_totals['DATE'] = monthly_totals['DATE'].astype(str)
        monthly_totals['Variance'] = ((monthly_totals['Actual'] - monthly_totals['Plan']) / monthly_totals['Plan'] * 100).round(2)
        
        metrics['monthly_trends'] = monthly_totals.to_dict('records')
        
        return metrics
    
    def get_economic_variables_data(self):
        """Get economic variables data for analysis"""
        if self.mv_parameters is None:
            return None
            
        # Get unique economic variables
        economic_vars = [
            'Consumer Price Index', 'Dow_Jones_Bank', 'S&P Index', 
            'FED_FUND_RATE', 'NASDAQ_TECH', 'KBW_FINTECH', 'FIS Price', 'FIS_Volue'
        ]
        
        # Aggregate by month for cleaner visualization
        monthly_econ_data = self.mv_parameters.groupby(self.mv_parameters['DATE'].dt.to_period('M')).agg({
            var: 'mean' for var in economic_vars
        }).reset_index()
        monthly_econ_data['DATE'] = monthly_econ_data['DATE'].astype(str)
        
        return {
            'variables': economic_vars,
            'monthly_data': monthly_econ_data.to_dict('records')
        }
    
    def create_advanced_features(self, df, lineup):
        """Create advanced features for machine learning"""
        lineup_data = df[df['Lineup'] == lineup].copy().sort_values('DATE')
        
        if len(lineup_data) < 12:
            return None
            
        # Time-based features
        lineup_data['month'] = lineup_data['DATE'].dt.month
        lineup_data['year'] = lineup_data['DATE'].dt.year
        lineup_data['quarter'] = lineup_data['DATE'].dt.quarter
        lineup_data['time_index'] = range(len(lineup_data))
        
        # Lag features
        for lag in [1, 2, 3, 6, 12]:
            lineup_data[f'lag_{lag}'] = lineup_data['Actual'].shift(lag)
        
        # Rolling statistics
        for window in [3, 6, 12]:
            lineup_data[f'rolling_mean_{window}'] = lineup_data['Actual'].rolling(window=window, min_periods=1).mean()
            lineup_data[f'rolling_std_{window}'] = lineup_data['Actual'].rolling(window=window, min_periods=1).std()
        
        # Seasonal features
        lineup_data['sin_month'] = np.sin(2 * np.pi * lineup_data['month'] / 12)
        lineup_data['cos_month'] = np.cos(2 * np.pi * lineup_data['month'] / 12)
        
        # Trend features
        lineup_data['trend'] = lineup_data['time_index']
        lineup_data['trend_squared'] = lineup_data['time_index'] ** 2
        
        # Fill NaN values
        lineup_data = lineup_data.fillna(method='bfill').fillna(method='ffill')
        
        return lineup_data
    
    def train_univariate_model(self, lineup):
        """Train univariate forecasting model using only Actual values"""
        if self.sample_data is None:
            return None
            
        # Create features
        features_data = self.create_advanced_features(self.sample_data, lineup)
        if features_data is None:
            return None
        
        # Define feature columns (excluding economic variables)
        feature_columns = [
            'time_index', 'month', 'year', 'quarter',
            'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
            'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
            'rolling_std_3', 'rolling_std_6', 'rolling_std_12',
            'sin_month', 'cos_month', 'trend', 'trend_squared'
        ]
        
        # Prepare training data
        X = features_data[feature_columns].fillna(0)
        y = features_data['Actual']
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        model.fit(X, y)
        
        # Calculate feature importance
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        
        return {
            'model': model,
            'feature_columns': feature_columns,
            'feature_importance': feature_importance,
            'last_data': features_data.iloc[-1].to_dict(),
            'model_type': 'univariate'
        }
    
    def train_multivariate_model(self, lineup):
        """Train multivariate forecasting model using economic variables"""
        if self.mv_parameters is None:
            return None
            
        lineup_data = self.mv_parameters[self.mv_parameters['Lineup'] == lineup].copy().sort_values('DATE')
        
        if len(lineup_data) < 12:
            return None
        
        # Create base features
        features_data = self.create_advanced_features(lineup_data, lineup)
        if features_data is None:
            return None
        
        # Add economic variables
        economic_vars = [
            'Consumer Price Index', 'Dow_Jones_Bank', 'S&P Index', 
            'FED_FUND_RATE', 'NASDAQ_TECH', 'KBW_FINTECH', 'FIS Price', 'FIS_Volue'
        ]
        
        # Include economic variables in features
        feature_columns = [
            'time_index', 'month', 'year', 'quarter',
            'lag_1', 'lag_2', 'lag_3', 'lag_6', 'lag_12',
            'rolling_mean_3', 'rolling_mean_6', 'rolling_mean_12',
            'rolling_std_3', 'rolling_std_6', 'rolling_std_12',
            'sin_month', 'cos_month', 'trend', 'trend_squared'
        ] + economic_vars
        
        # Prepare training data
        X = features_data[feature_columns].fillna(0)
        y = features_data['Actual']
        
        # Scale features for better performance with economic variables
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        model = RandomForestRegressor(n_estimators=150, random_state=42, max_depth=15)
        model.fit(X_scaled, y)
        
        # Calculate feature importance
        feature_importance = dict(zip(feature_columns, model.feature_importances_))
        
        return {
            'model': model,
            'scaler': self.scaler,
            'feature_columns': feature_columns,
            'feature_importance': feature_importance,
            'last_data': features_data.iloc[-1].to_dict(),
            'model_type': 'multivariate'
        }
    
    def generate_forecast(self, lineup, forecast_type='univariate', months_ahead=12):
        """Generate forecast for specified lineup and type"""
        
        # Train appropriate model
        if forecast_type == 'univariate':
            model_info = self.train_univariate_model(lineup)
        else:
            model_info = self.train_multivariate_model(lineup)
        
        if model_info is None:
            return None
        
        model = model_info['model']
        feature_columns = model_info['feature_columns']
        last_data = model_info['last_data']
        
        # Generate future predictions
        forecasts = []
        last_date = pd.to_datetime(last_data['DATE'])
        
        for i in range(1, months_ahead + 1):
            # Calculate future date
            future_date = last_date + pd.DateOffset(months=i)
            
            # Create future features
            future_features = self._create_future_features(
                last_data, i, future_date, feature_columns, model_info['model_type']
            )
            
            # Scale features if multivariate
            if model_info['model_type'] == 'multivariate':
                future_features_scaled = model_info['scaler'].transform([future_features])
                prediction = model.predict(future_features_scaled)[0]
            else:
                prediction = model.predict([future_features])[0]
            
            forecasts.append({
                'date': future_date.strftime('%Y-%m-%d'),
                'forecast': max(0, prediction),  # Ensure non-negative
                'month_ahead': i
            })
        
        return {
            'lineup': lineup,
            'forecast_type': forecast_type,
            'forecasts': forecasts,
            'model_metrics': {
                'feature_importance': model_info['feature_importance'],
                'model_type': model_info['model_type']
            }
        }
    
    def _create_future_features(self, last_data, months_ahead, future_date, feature_columns, model_type):
        """Create features for future prediction"""
        features = {}
        
        # Time-based features
        features['time_index'] = last_data['time_index'] + months_ahead
        features['month'] = future_date.month
        features['year'] = future_date.year
        features['quarter'] = (future_date.month - 1) // 3 + 1
        
        # Lag features (use last known values)
        features['lag_1'] = last_data['Actual']
        features['lag_2'] = last_data.get('lag_1', last_data['Actual'])
        features['lag_3'] = last_data.get('lag_2', last_data['Actual'])
        features['lag_6'] = last_data.get('lag_5', last_data['Actual'])
        features['lag_12'] = last_data.get('lag_11', last_data['Actual'])
        
        # Rolling statistics
        features['rolling_mean_3'] = last_data.get('rolling_mean_3', last_data['Actual'])
        features['rolling_mean_6'] = last_data.get('rolling_mean_6', last_data['Actual'])
        features['rolling_mean_12'] = last_data.get('rolling_mean_12', last_data['Actual'])
        features['rolling_std_3'] = last_data.get('rolling_std_3', 0)
        features['rolling_std_6'] = last_data.get('rolling_std_6', 0)
        features['rolling_std_12'] = last_data.get('rolling_std_12', 0)
        
        # Seasonal features
        features['sin_month'] = np.sin(2 * np.pi * future_date.month / 12)
        features['cos_month'] = np.cos(2 * np.pi * future_date.month / 12)
        
        # Trend features
        features['trend'] = features['time_index']
        features['trend_squared'] = features['time_index'] ** 2
        
        # Economic variables (for multivariate model)
        if model_type == 'multivariate':
            economic_vars = [
                'Consumer Price Index', 'Dow_Jones_Bank', 'S&P Index', 
                'FED_FUND_RATE', 'NASDAQ_TECH', 'KBW_FINTECH', 'FIS Price', 'FIS_Volue'
            ]
            
            for var in economic_vars:
                # Use last known value with slight trend adjustment
                last_value = last_data.get(var, 0)
                # Simple trend extrapolation
                features[var] = last_value * (1 + 0.001 * months_ahead)  # 0.1% monthly growth assumption
        
        # Return features in the correct order
        return [features.get(col, 0) for col in feature_columns]
    
    def compare_forecast_methods(self, lineup, months_ahead=12):
        """Compare univariate vs multivariate forecasting"""
        
        # Generate both forecasts
        univariate_result = self.generate_forecast(lineup, 'univariate', months_ahead)
        multivariate_result = self.generate_forecast(lineup, 'multivariate', months_ahead)
        
        if not univariate_result or not multivariate_result:
            return None
        
        # Combine results for comparison
        comparison_data = []
        for i in range(months_ahead):
            comparison_data.append({
                'date': univariate_result['forecasts'][i]['date'],
                'univariate_forecast': univariate_result['forecasts'][i]['forecast'],
                'multivariate_forecast': multivariate_result['forecasts'][i]['forecast'],
                'difference': multivariate_result['forecasts'][i]['forecast'] - univariate_result['forecasts'][i]['forecast'],
                'month_ahead': i + 1
            })
        
        return {
            'lineup': lineup,
            'comparison_data': comparison_data,
            'univariate_importance': univariate_result['model_metrics']['feature_importance'],
            'multivariate_importance': multivariate_result['model_metrics']['feature_importance']
        }