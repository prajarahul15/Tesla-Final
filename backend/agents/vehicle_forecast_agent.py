"""
Autonomous Vehicle Forecast Agent
==================================
This agent runs independently to generate forecasts for all vehicle models
without user intervention. It pre-computes both univariate and multivariate
forecasts and stores them in a cache for instant retrieval.

Key Features:
- Autonomous execution (on startup + scheduled intervals)
- Batch processing for all models
- Both univariate & multivariate methods
- Cache storage with timestamps
- Production & deliveries forecasting
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelForecast:
    """Container for a model's forecast data"""
    model_key: str
    model_name: str
    forecast_date: datetime
    univariate: Dict
    multivariate: Dict
    metadata: Dict = field(default_factory=dict)


class ForecastCacheManager:
    """
    Manages in-memory cache of pre-computed forecasts
    TTL: 24 hours (configurable)
    """
    
    def __init__(self, ttl_hours: int = 24):
        self.cache: Dict[str, ModelForecast] = {}
        self.ttl = timedelta(hours=ttl_hours)
        self.last_update: Optional[datetime] = None
        logger.info(f"ForecastCacheManager initialized with TTL={ttl_hours} hours")
    
    def set(self, model_key: str, forecast: ModelForecast):
        """Store forecast for a model"""
        self.cache[model_key] = forecast
        logger.info(f"Cached forecast for {model_key}")
    
    def get(self, model_key: str) -> Optional[ModelForecast]:
        """Retrieve forecast if available and not expired"""
        if model_key not in self.cache:
            logger.warning(f"No cached forecast for {model_key}")
            return None
        
        forecast = self.cache[model_key]
        age = datetime.now() - forecast.forecast_date
        
        if age > self.ttl:
            logger.warning(f"Cached forecast for {model_key} expired (age: {age})")
            del self.cache[model_key]
            return None
        
        logger.info(f"Retrieved cached forecast for {model_key} (age: {age})")
        return forecast
    
    def get_all(self) -> Dict[str, ModelForecast]:
        """Get all non-expired forecasts"""
        now = datetime.now()
        valid_forecasts = {}
        expired_keys = []
        
        for key, forecast in self.cache.items():
            age = now - forecast.forecast_date
            if age > self.ttl:
                expired_keys.append(key)
            else:
                valid_forecasts[key] = forecast
        
        # Clean up expired
        for key in expired_keys:
            del self.cache[key]
            logger.info(f"Removed expired forecast for {key}")
        
        return valid_forecasts
    
    def clear(self):
        """Clear all cached forecasts"""
        count = len(self.cache)
        self.cache.clear()
        self.last_update = None
        logger.info(f"Cleared {count} cached forecasts")
    
    def is_stale(self) -> bool:
        """Check if cache needs refresh"""
        if not self.last_update:
            return True
        age = datetime.now() - self.last_update
        return age > self.ttl


class VehicleForecastAgent:
    """
    Autonomous agent that generates forecasts for all vehicle models
    - Runs automatically on startup
    - Can be triggered manually for refresh
    - Supports both univariate and multivariate forecasting
    - Forecasts production and deliveries (sold quantity)
    """
    
    def __init__(self, vehicle_engine):
        """
        Args:
            vehicle_engine: Instance of VehicleForecastEngine with loaded data
        """
        self.engine = vehicle_engine
        self.cache = ForecastCacheManager(ttl_hours=24)
        self.is_running = False
        self.default_horizon = 12  # 12 months ahead
        self.test_window = 6  # 6 months for test evaluation
        logger.info("VehicleForecastAgent initialized")
    
    async def generate_all_forecasts(self) -> Dict[str, Dict]:
        """
        Main method: Generate forecasts for all available models
        Returns summary of forecasting results
        """
        if self.is_running:
            logger.warning("Forecast generation already in progress")
            return {"status": "already_running"}
        
        self.is_running = True
        start_time = datetime.now()
        results = {
            "status": "running",
            "start_time": start_time.isoformat(),
            "models_processed": [],
            "models_failed": [],
            "total_models": 0
        }
        
        try:
            logger.info("🚀 Starting autonomous forecast generation for all models...")
            
            # Get available models
            models = self.engine.get_available_models()
            results["total_models"] = len(models)
            
            if not models:
                logger.warning("No vehicle models found in data")
                results["status"] = "no_models"
                return results
            
            logger.info(f"Found {len(models)} models to forecast")
            
            # Process each model
            for model_info in models:
                model_key = model_info.model_key
                try:
                    logger.info(f"📊 Processing {model_key}...")
                    
                    # Generate univariate multi-target forecast (deliveries, production, sold)
                    univariate_result = self.engine.generate_multi_target_forecast(
                        model_key=model_key,
                        forecast_type="univariate",
                        months_ahead=self.default_horizon,
                        test_window=self.test_window
                    )
                    
                    # Generate multivariate multi-target forecast (deliveries, production, sold)
                    multivariate_result = self.engine.generate_multi_target_forecast(
                        model_key=model_key,
                        forecast_type="multivariate",
                        months_ahead=self.default_horizon,
                        test_window=self.test_window
                    )
                    
                    if univariate_result and multivariate_result:
                        # Store in cache
                        forecast = ModelForecast(
                            model_key=model_key,
                            model_name=model_info.display_name,
                            forecast_date=datetime.now(),
                            univariate=univariate_result,
                            multivariate=multivariate_result,
                            metadata={
                                "horizon_months": self.default_horizon,
                                "test_window": self.test_window,
                                "data_range": f"{model_info.first_date} to {model_info.last_date}",
                                "records_count": model_info.records,
                                "available_targets": univariate_result.get("available_targets", [])
                            }
                        )
                        
                        self.cache.set(model_key, forecast)
                        
                        # Calculate average metrics across all targets
                        uni_targets = univariate_result.get("forecasts_by_target", {})
                        multi_targets = multivariate_result.get("forecasts_by_target", {})
                        
                        uni_mae = sum(t["model_metrics"]["mae"] for t in uni_targets.values()) / len(uni_targets) if uni_targets else 0
                        uni_mape = sum(t["model_metrics"]["mape"] for t in uni_targets.values()) / len(uni_targets) if uni_targets else 0
                        multi_mae = sum(t["model_metrics"]["mae"] for t in multi_targets.values()) / len(multi_targets) if multi_targets else 0
                        multi_mape = sum(t["model_metrics"]["mape"] for t in multi_targets.values()) / len(multi_targets) if multi_targets else 0
                        
                        results["models_processed"].append({
                            "model_key": model_key,
                            "model_name": model_info.display_name,
                            "available_targets": univariate_result.get("available_targets", []),
                            "univariate_mae_avg": uni_mae,
                            "multivariate_mae_avg": multi_mae,
                            "univariate_mape_avg": uni_mape,
                            "multivariate_mape_avg": multi_mape,
                        })
                        logger.info(f"✅ Successfully forecasted {model_key} for targets: {univariate_result.get('available_targets', [])}")
                    else:
                        logger.error(f"❌ Failed to generate forecast for {model_key}")
                        results["models_failed"].append({
                            "model_key": model_key,
                            "reason": "Forecast generation returned None"
                        })
                
                except Exception as e:
                    logger.error(f"❌ Error processing {model_key}: {str(e)}")
                    results["models_failed"].append({
                        "model_key": model_key,
                        "reason": str(e)
                    })
            
            # Update cache timestamp
            self.cache.last_update = datetime.now()
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            results["status"] = "completed"
            results["end_time"] = end_time.isoformat()
            results["duration_seconds"] = duration
            results["success_count"] = len(results["models_processed"])
            results["failure_count"] = len(results["models_failed"])
            
            logger.info(f"🎉 Forecast generation completed in {duration:.2f}s")
            logger.info(f"   ✅ Success: {results['success_count']}/{results['total_models']}")
            logger.info(f"   ❌ Failed: {results['failure_count']}/{results['total_models']}")
            
            return results
        
        except Exception as e:
            logger.error(f"Critical error in forecast generation: {str(e)}")
            results["status"] = "error"
            results["error"] = str(e)
            return results
        
        finally:
            self.is_running = False
    
    def get_forecast(self, model_key: str) -> Optional[ModelForecast]:
        """
        Retrieve pre-computed forecast for a specific model
        Returns None if not available or expired
        """
        return self.cache.get(model_key)
    
    def get_all_forecasts(self) -> Dict[str, ModelForecast]:
        """
        Retrieve all valid forecasts
        """
        return self.cache.get_all()
    
    def force_refresh(self, model_key: Optional[str] = None):
        """
        Force refresh of forecasts
        If model_key is provided, only refresh that model
        Otherwise, refresh all models
        """
        if model_key:
            logger.info(f"Force refresh requested for {model_key}")
            # Remove from cache to force regeneration
            if model_key in self.cache.cache:
                del self.cache.cache[model_key]
        else:
            logger.info("Force refresh requested for all models")
            self.cache.clear()
    
    def get_status(self) -> Dict:
        """
        Get agent status and cache statistics
        """
        all_forecasts = self.cache.get_all()
        
        return {
            "is_running": self.is_running,
            "cache_size": len(all_forecasts),
            "last_update": self.cache.last_update.isoformat() if self.cache.last_update else None,
            "is_stale": self.cache.is_stale(),
            "ttl_hours": self.cache.ttl.total_seconds() / 3600,
            "cached_models": [
                {
                    "model_key": key,
                    "model_name": forecast.model_name,
                    "forecast_date": forecast.forecast_date.isoformat(),
                    "age_hours": (datetime.now() - forecast.forecast_date).total_seconds() / 3600
                }
                for key, forecast in all_forecasts.items()
            ]
        }


# Singleton instance (will be initialized in server.py)
vehicle_forecast_agent: Optional[VehicleForecastAgent] = None


def get_vehicle_forecast_agent() -> Optional[VehicleForecastAgent]:
    """Get the global vehicle forecast agent instance"""
    return vehicle_forecast_agent


def initialize_vehicle_forecast_agent(vehicle_engine) -> VehicleForecastAgent:
    """
    Initialize the global vehicle forecast agent
    Should be called once on application startup
    """
    global vehicle_forecast_agent
    vehicle_forecast_agent = VehicleForecastAgent(vehicle_engine)
    logger.info("Global vehicle forecast agent created")
    return vehicle_forecast_agent

