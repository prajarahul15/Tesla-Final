"""
Metric Forecasting Service for Tesla Financial Data
Provides univariate and multivariate forecasting capabilities
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Models
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Some forecasting features will be limited.")
    SKLEARN_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Warning: statsmodels not available. ARIMA forecasting will use simple trend.")
    STATSMODELS_AVAILABLE = False

class MetricForecastingService:
    """
    Advanced forecasting service for Tesla's financial metrics
    """
    
    def __init__(self):
        self.monthly_data = None
        self.macro_indicators = None

    def load_data(self):
        """Load Tesla monthly financial data"""
        try:
            # Load the exact monthly data we created
            self.monthly_data = pd.read_csv('data/tesla_monthly_financial_exact.csv')
            self.monthly_data['date'] = pd.to_datetime(self.monthly_data['date'])
            self.monthly_data = self.monthly_data.sort_values('date').reset_index(drop=True)
            
            # Create synthetic macro indicators for demonstration
            self.macro_indicators = self._create_synthetic_macro_data()
            
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def _create_synthetic_macro_data(self) -> pd.DataFrame:
        """Create synthetic macroeconomic indicators"""
        if self.monthly_data is None:
            return pd.DataFrame()
            
        dates = self.monthly_data['date'].tolist()
        n_periods = len(dates)
        
        # Create realistic macro indicators
        np.random.seed(42)  # For reproducibility
        
        macro_data = pd.DataFrame({
            'date': dates,
            'gdp_growth': np.random.normal(2.5, 0.8, n_periods),  # GDP growth %
            'inflation_rate': np.random.normal(3.2, 1.2, n_periods),  # Inflation %
            'interest_rate': np.random.normal(4.5, 1.5, n_periods),  # Interest rate %
            'unemployment_rate': np.random.normal(4.0, 1.0, n_periods),  # Unemployment %
            'consumer_confidence': np.random.normal(100, 15, n_periods),  # Consumer confidence index
            'oil_price': np.random.normal(75, 20, n_periods),  # Oil price per barrel
            'ev_market_size': np.linspace(50, 200, n_periods) + np.random.normal(0, 10, n_periods)  # EV market size (billions)
        })
        
        return macro_data
    
    def univariate_forecast(
        self, 
        metric: str, 
        horizon: int = 12, 
        test_months: int = 6
    ) -> Dict:
        """
        Generate univariate forecast for a single metric
        
        Args:
            metric: The metric to forecast (e.g., 'revenue_millions')
            horizon: Number of months to forecast into the future
            test_months: Number of months to use for testing
            
        Returns:
            Dictionary containing forecast results, accuracy metrics, and plot data
        """
        if self.monthly_data is None:
            self.load_data()
            
        # Prepare time series data
        ts_data = self.monthly_data[['date', metric]].copy()
        ts_data = ts_data.dropna()
        
        # Split data
        total_months = len(ts_data)
        train_end = total_months - test_months
        
        train_data = ts_data.iloc[:train_end]
        test_data = ts_data.iloc[train_end:]
        
        # Fit ARIMA model (simple implementation)
        train_values = train_data[metric].values
        
        if STATSMODELS_AVAILABLE:
            try:
                # Fit ARIMA model
                model = ARIMA(train_values, order=(2, 1, 2))
                fitted_model = model.fit()

                # Test period forecast
                test_forecast = fitted_model.forecast(steps=test_months)

                # Future forecast
                future_forecast = fitted_model.forecast(steps=horizon)

            except Exception as e:
                print(f"ARIMA failed, using simple trend: {e}")
                # Fallback to simple trend
                test_forecast = self._simple_trend_forecast(train_values, test_months)
                future_forecast = self._simple_trend_forecast(train_values, horizon)
        else:
            # Use simple trend if statsmodels not available
            test_forecast = self._simple_trend_forecast(train_values, test_months)
            future_forecast = self._simple_trend_forecast(train_values, horizon)
            model_name = "Simple Trend"
        
        # Calculate accuracy metrics
        actual_test = test_data[metric].values
        if SKLEARN_AVAILABLE:
            mae = mean_absolute_error(actual_test, test_forecast)
            rmse = np.sqrt(mean_squared_error(actual_test, test_forecast))
            r2 = r2_score(actual_test, test_forecast) if len(actual_test) > 1 else 0
        else:
            # Simple accuracy calculations without sklearn
            mae = np.mean(np.abs(actual_test - test_forecast))
            rmse = np.sqrt(np.mean((actual_test - test_forecast) ** 2))
            ss_res = np.sum((actual_test - test_forecast) ** 2)
            ss_tot = np.sum((actual_test - np.mean(actual_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        mape = np.mean(np.abs((actual_test - test_forecast) / actual_test)) * 100
        
        # Prepare plot data
        plot_data = self._prepare_plot_data(
            ts_data, train_end, test_forecast, future_forecast, horizon, self.macro_indicators
        )
        
        # Prepare forecast table - start from end of actual data
        actual_end_date = ts_data.iloc[-1]['date']  # Last actual data point (Dec 2024)
        forecast_table = self._prepare_forecast_table(
            actual_end_date, future_forecast, horizon, train_end
        )
        
        return {
            'success': True,
            'metric': metric,
            'forecast_type': 'univariate',
            'model': 'ARIMA(2,1,2)',
            'accuracy': {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2': float(r2)
            },
            'plot_data': plot_data,
            'forecast_table': forecast_table,
            'test_period_months': test_months,
            'forecast_horizon': horizon
        }
    
    def multivariate_forecast(
        self, 
        metrics: List[str], 
        horizon: int = 12, 
        test_months: int = 6
    ) -> Dict:
        """
        Generate multivariate forecast for multiple metrics
        
        Args:
            metrics: List of metrics to forecast
            horizon: Number of months to forecast into the future
            test_months: Number of months to use for testing
            
        Returns:
            Dictionary containing forecast results for all metrics and feature importance
        """
        if self.monthly_data is None:
            self.load_data()
            
        # Prepare data with macro indicators
        data = self.monthly_data.merge(self.macro_indicators, on='date', how='left')
        
        results = {}
        feature_importance_data = []
        
        for metric in metrics:
            # Prepare features (macro indicators + lagged values)
            features = self._prepare_multivariate_features(data, metric)
            target = data[metric].values[2:]  # Skip first 2 rows due to lags
            
            # Split data
            total_samples = len(target)
            train_end = total_samples - test_months
            
            X_train = features[:train_end]
            y_train = target[:train_end]
            X_test = features[train_end:]
            y_test = target[train_end:]
            
            if SKLEARN_AVAILABLE:
                # Fit Random Forest model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)

                # Test period predictions
                test_predictions = model.predict(X_test)
                
                # Feature importance
                feature_names = self._get_feature_names()
                importance_scores = model.feature_importances_
                
                # Future predictions (simplified - using last known values)
                last_features = features[-1:].repeat(horizon, axis=0)
                future_predictions = model.predict(last_features)
            else:
                # Simple linear trend if sklearn not available
                test_predictions = self._simple_trend_forecast(y_train, len(y_test))
                future_predictions = self._simple_trend_forecast(y_train, horizon)
                feature_names = self._get_feature_names()
                importance_scores = np.ones(len(feature_names)) / len(feature_names)  # Equal importance
                model_type = "Simple Trend"
            
            for fname, score in zip(feature_names, importance_scores):
                feature_importance_data.append({
                    'metric': metric,
                    'feature': fname,
                    'importance': float(score)
                })
            
            # Future predictions with sequential lag updates for realistic forecasts
            if len(features) > 0:
                future_predictions = []
                last_features = features[-1].copy()
                
                # Keep track of recent values for lag features
                recent_values = list(y_train[-2:])  # Last 2 actual values for lag calculation
                
                for i in range(horizon):
                    # Create features for this prediction step
                    current_features = last_features.copy()
                    
                    # Update lag features with recent predictions/values
                    if len(current_features) >= 2:
                        current_features[0] = recent_values[-1]  # Lag 1
                        if len(recent_values) >= 2:
                            current_features[1] = recent_values[-2]  # Lag 2
                    
                    # Add slight variations to economic indicators to simulate uncertainty
                    if len(current_features) > 2:
                        # Economic indicators evolve slightly over time
                        trend_factor = 1 + (i * 0.003)  # 0.3% monthly trend
                        seasonal_factor = 1 + 0.05 * np.sin(i * np.pi / 6)  # 6-month cycle
                        noise_factor = np.random.normal(1.0, 0.02)  # 2% random noise
                        
                        # Apply to economic indicators (skip lag features)
                        current_features[2:] = current_features[2:] * trend_factor * seasonal_factor * noise_factor
                    
                    # Make prediction for this step
                    pred = model.predict(current_features.reshape(1, -1))[0]
                    future_predictions.append(pred)
                    
                    # Update recent values for next iteration
                    recent_values.append(pred)
                    if len(recent_values) > 3:  # Keep only last 3 values
                        recent_values.pop(0)
                
                future_predictions = np.array(future_predictions)
            else:
                future_predictions = self._simple_trend_forecast(y_train, horizon)
            
            # Calculate accuracy
            if SKLEARN_AVAILABLE:
                mae = mean_absolute_error(y_test, test_predictions)
                rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
                r2 = r2_score(y_test, test_predictions) if len(y_test) > 1 else 0
            else:
                mae = np.mean(np.abs(y_test - test_predictions))
                rmse = np.sqrt(np.mean((y_test - test_predictions) ** 2))
                ss_res = np.sum((y_test - test_predictions) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            mape = np.mean(np.abs((y_test - test_predictions) / y_test)) * 100 if np.all(y_test != 0) else 0
            
            # Prepare plot data for this metric
            metric_data = data[['date', metric]].iloc[2:].reset_index(drop=True)
            plot_data = self._prepare_plot_data(
                metric_data, train_end, test_predictions, future_predictions, horizon, self.macro_indicators
            )
            
            # Forecast table - start from end of actual data
            actual_end_date = data.iloc[-1]['date']  # Last actual data point (Dec 2024)
            forecast_table = self._prepare_forecast_table(
                actual_end_date, future_predictions, horizon, train_end
            )
            
            results[metric] = {
                'accuracy': {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'mape': float(mape),
                    'r2': float(r2)
                },
                'plot_data': plot_data,
                'forecast_table': forecast_table
            }
        
        # Aggregate feature importance
        feature_importance = self._aggregate_feature_importance(feature_importance_data)
        
        return {
            'success': True,
            'metrics': metrics,
            'forecast_type': 'multivariate',
            'model': 'Random Forest',
            'results': results,
            'feature_importance': feature_importance,
            'test_period_months': test_months,
            'forecast_horizon': horizon
        }
    
    def _simple_trend_forecast(self, values: np.ndarray, periods: int) -> np.ndarray:
        """Simple trend-based forecast as fallback"""
        if len(values) < 2:
            return np.full(periods, values[-1] if len(values) > 0 else 0)
        
        # Calculate simple trend
        x = np.arange(len(values))
        trend = np.polyfit(x, values, 1)
        
        # Project trend forward
        future_x = np.arange(len(values), len(values) + periods)
        forecast = np.polyval(trend, future_x)
        
        return forecast
    
    def _prepare_multivariate_features(self, data: pd.DataFrame, target_metric: str) -> np.ndarray:
        """Prepare features for multivariate forecasting"""
        feature_cols = [
            'gdp_growth', 'inflation_rate', 'interest_rate', 'unemployment_rate',
            'consumer_confidence', 'oil_price', 'ev_market_size'
        ]
        
        # Add lagged versions of the target metric
        data[f'{target_metric}_lag1'] = data[target_metric].shift(1)
        data[f'{target_metric}_lag2'] = data[target_metric].shift(2)
        
        feature_cols.extend([f'{target_metric}_lag1', f'{target_metric}_lag2'])
        
        # Create feature matrix (skip first 2 rows due to lags)
        features = data[feature_cols].iloc[2:].fillna(method='ffill').values
        
        return features
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for importance analysis"""
        return [
            'GDP Growth', 'Inflation Rate', 'Interest Rate', 'Unemployment Rate',
            'Consumer Confidence', 'Oil Price', 'EV Market Size', 'Lag 1', 'Lag 2'
        ]
    
    def _prepare_plot_data(
        self, 
        ts_data: pd.DataFrame, 
        train_end: int, 
        test_forecast: np.ndarray, 
        future_forecast: np.ndarray, 
        horizon: int,
        macro_data: pd.DataFrame = None
    ) -> List[Dict]:
        """Prepare data for plotting"""
        plot_data = []
        
        # Historical actual data
        for i, row in ts_data.iterrows():
            # Always include actual data if available
            actual_val = float(row[ts_data.columns[1]]) if pd.notna(row[ts_data.columns[1]]) else None
            test_forecast_val = None
            
            # Set test_forecast for test period indices
            if i >= train_end and i - train_end < len(test_forecast):
                test_forecast_val = float(test_forecast[i - train_end])
            
            # Get macro indicators for this date if available
            macro_indicators = {}
            if macro_data is not None and not macro_data.empty:
                date_str = row['date'].strftime('%Y-%m')
                macro_row = macro_data[macro_data['date'].dt.strftime('%Y-%m') == date_str]
                if not macro_row.empty:
                    macro_indicators = {
                        'gdp_growth': float(macro_row.iloc[0]['gdp_growth']) if pd.notna(macro_row.iloc[0]['gdp_growth']) else None,
                        'inflation_rate': float(macro_row.iloc[0]['inflation_rate']) if pd.notna(macro_row.iloc[0]['inflation_rate']) else None,
                        'interest_rate': float(macro_row.iloc[0]['interest_rate']) if pd.notna(macro_row.iloc[0]['interest_rate']) else None,
                        'unemployment_rate': float(macro_row.iloc[0]['unemployment_rate']) if pd.notna(macro_row.iloc[0]['unemployment_rate']) else None,
                        'consumer_confidence': float(macro_row.iloc[0]['consumer_confidence']) if pd.notna(macro_row.iloc[0]['consumer_confidence']) else None,
                        'oil_price': float(macro_row.iloc[0]['oil_price']) if pd.notna(macro_row.iloc[0]['oil_price']) else None,
                        'ev_market_size': float(macro_row.iloc[0]['ev_market_size']) if pd.notna(macro_row.iloc[0]['ev_market_size']) else None
                    }
            
            plot_data.append({
                'date': row['date'].strftime('%Y-%m'),
                'actual': actual_val,  # Always include actual data when available
                'test_forecast': test_forecast_val,
                'future_forecast': None,
                **macro_indicators
            })
        
        # Future forecast data - start from month after actual data ends
        actual_end_date = ts_data.iloc[-1]['date']  # Last actual data point (Dec 2024)
        for i in range(horizon):
            # Start from January 2025, then add months
            future_date = pd.Timestamp(year=2025, month=1, day=1) + pd.DateOffset(months=i)
            
            # Generate synthetic macro indicators for future periods
            future_macro_indicators = {}
            if macro_data is not None and not macro_data.empty:
                # Use the last known values as base and add some variation
                last_macro = macro_data.iloc[-1]
                future_macro_indicators = {
                    'gdp_growth': float(last_macro['gdp_growth']) + np.random.normal(0, 0.1),
                    'inflation_rate': float(last_macro['inflation_rate']) + np.random.normal(0, 0.1),
                    'interest_rate': float(last_macro['interest_rate']) + np.random.normal(0, 0.1),
                    'unemployment_rate': float(last_macro['unemployment_rate']) + np.random.normal(0, 0.1),
                    'consumer_confidence': float(last_macro['consumer_confidence']) + np.random.normal(0, 2),
                    'oil_price': float(last_macro['oil_price']) + np.random.normal(0, 2),
                    'ev_market_size': float(last_macro['ev_market_size']) + np.random.normal(0, 5)
                }
            
            plot_data.append({
                'date': future_date.strftime('%Y-%m'),
                'actual': None,
                'test_forecast': None,
                'future_forecast': float(future_forecast[i]),
                **future_macro_indicators
            })
        
        return plot_data
    
    def _prepare_forecast_table(
        self, 
        last_date: pd.Timestamp, 
        forecast: np.ndarray, 
        horizon: int,
        train_end: int = None
    ) -> List[Dict]:
        """Prepare forecast table data"""
        table_data = []
        
        # Use actual end date (before test period) if train_end is provided
        start_date = last_date
        if train_end is not None and train_end > 0:
            # This will be updated in the calling method
            pass
        
        for i in range(horizon):
            # Start from January 2025, then add months
            future_date = pd.Timestamp(year=2025, month=1, day=1) + pd.DateOffset(months=i)
            forecast_value = float(forecast[i])
            
            # Simple confidence intervals (±10%)
            lower_ci = forecast_value * 0.9
            upper_ci = forecast_value * 1.1
            
            table_data.append({
                'date': future_date.strftime('%Y-%m'),
                'forecast': forecast_value,
                'lower_ci': lower_ci,
                'upper_ci': upper_ci
            })
        
        return table_data
    
    def _aggregate_feature_importance(self, importance_data: List[Dict]) -> List[Dict]:
        """Aggregate feature importance across metrics"""
        feature_scores = {}
        
        for item in importance_data:
            feature = item['feature']
            importance = item['importance']
            
            if feature not in feature_scores:
                feature_scores[feature] = []
            feature_scores[feature].append(importance)
        
        # Calculate average importance
        aggregated = []
        for feature, scores in feature_scores.items():
            avg_importance = np.mean(scores)
            aggregated.append({
                'feature': feature,
                'importance': float(avg_importance)
            })
        
        # Sort by importance
        aggregated.sort(key=lambda x: x['importance'], reverse=True)
        
        return aggregated

# Global instance
metric_forecasting_service = MetricForecastingService()
