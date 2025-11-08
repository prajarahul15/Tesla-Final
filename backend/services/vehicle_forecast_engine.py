"""
Vehicle-level monthly forecasting engine
- Loads Excel with monthly production/delivery per model
- Trains simple ML models per model (univariate baseline; multivariate optional)
- Generates H-month forecasts using autoregressive feature construction

This module is self-contained to avoid impacting existing analytics modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class VehicleModelInfo:
    model_key: str
    display_name: str
    first_date: str
    last_date: str
    records: int


class VehicleForecastEngine:
    def __init__(self, xlsx_path: Optional[str] = None):
        self.xlsx_path: Optional[str] = xlsx_path or os.environ.get("VEHICLE_DATA_XLSX")
        self.monthly_data: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()
        # Optional monthly economic variables (DATE + columns)
        self.econ_monthly_df: Optional[pd.DataFrame] = None
        self._econ_columns: List[str] = []

    # ---------------------- Data Loading ----------------------
    def load_data(self) -> bool:
        """Load and normalize Excel into a canonical monthly dataframe.
        Expected canonical columns: ['DATE','model_key','deliveries','production']
        """
        if not self.xlsx_path:
            return False
        try:
            xl = pd.ExcelFile(self.xlsx_path)
            frames: List[pd.DataFrame] = []
            for sheet_name in xl.sheet_names:
                try:
                    df = xl.parse(sheet_name)
                    if df is None or df.empty:
                        continue
                    frames.append(self._normalize_sheet(df))
                except Exception:
                    # Skip sheets that fail to parse
                    continue
            if not frames:
                return False
            monthly = pd.concat(frames, ignore_index=True)
            # Drop rows with missing essentials
            monthly = monthly.dropna(subset=["DATE", "model_key"]).copy()
            # Coerce types
            monthly["DATE"] = pd.to_datetime(monthly["DATE"]).dt.to_period("M").dt.to_timestamp()
            for col in ["deliveries", "production", "sold", "revenue", "asp"]:
                if col in monthly.columns:
                    monthly[col] = pd.to_numeric(monthly[col], errors="coerce")
            # Derive revenue if not present but ASP and deliveries exist
            if "revenue" not in monthly.columns and "asp" in monthly.columns and "deliveries" in monthly.columns:
                try:
                    monthly["revenue"] = monthly["asp"].fillna(0) * monthly["deliveries"].fillna(0)
                except Exception:
                    pass
            # Aggregate duplicates
            agg_map = {"deliveries": "sum", "production": "sum"}
            if "sold" in monthly.columns:
                agg_map["sold"] = "sum"
            if "revenue" in monthly.columns:
                agg_map["revenue"] = "sum"
            if "asp" in monthly.columns:
                # Average ASP within same month/model rows
                agg_map["asp"] = "mean"
            monthly = (
                monthly.groupby(["model_key", "DATE"], as_index=False)
                .agg(agg_map)
            )
            self.monthly_data = monthly.sort_values(["model_key", "DATE"]).reset_index(drop=True)
            return True
        except Exception:
            return False

    def _normalize_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        # Standardize column names
        cols = {str(c).strip(): str(c).strip().lower() for c in df.columns}
        df = df.rename(columns=cols)
        # Heuristic detection of key columns
        date_col = next((c for c in df.columns if c.lower() in ["date", "month", "ds"]), None)
        model_col = next((c for c in df.columns if any(k in c.lower() for k in ["model", "vehicle", "name"])), None)
        deliv_col = next((c for c in df.columns if any(k in c.lower() for k in ["deliveries", "delivery", "delivered", "units"])) , None)
        prod_col = next((c for c in df.columns if any(k in c.lower() for k in ["production", "produced", "prod"])) , None)
        sold_col = next((c for c in df.columns if any(k in c.lower() for k in ["sold", "units_sold", "quantity_sold", "qty_sold"])) , None)
        rev_col = next((c for c in df.columns if any(k in c.lower() for k in ["revenue", "sales"])) , None)
        asp_col = next((c for c in df.columns if any(k in c.lower() for k in ["asp", "avg selling", "average selling", "avg_price", "price"])) , None)

        out = pd.DataFrame()
        if date_col is not None:
            out["DATE"] = pd.to_datetime(df[date_col], errors="coerce")
        if model_col is not None:
            out["model_key"] = df[model_col].apply(self._to_model_key)
        if deliv_col is not None:
            out["deliveries"] = pd.to_numeric(df[deliv_col], errors="coerce")
        if prod_col is not None:
            out["production"] = pd.to_numeric(df[prod_col], errors="coerce")
        if sold_col is not None:
            out["sold"] = pd.to_numeric(df[sold_col], errors="coerce")
        if rev_col is not None:
            out["revenue"] = pd.to_numeric(df[rev_col], errors="coerce")
        if asp_col is not None:
            out["asp"] = pd.to_numeric(df[asp_col], errors="coerce")
        return out

    def _to_model_key(self, name: str) -> str:
        if not isinstance(name, str):
            return "unknown"
        n = name.strip().lower().replace("-", " ")
        mapping = {
            "model s": "model_s",
            "s": "model_s",
            "model x": "model_x",
            "x": "model_x",
            "model 3": "model_3",
            "m3": "model_3",
            "model y": "model_y",
            "y": "model_y",
            "cybertruck": "cybertruck",
            "semi": "semi",
        }
        return mapping.get(n, "_".join([p for p in n.split() if p.isalnum()]))

    # ---------------------- Optional Econ Integration ----------------------
    def set_economic_variables(self, econ_df: pd.DataFrame):
        """Set monthly economic variables dataframe.
        Requires a 'DATE' column at month start; other numeric columns will be used.
        """
        if econ_df is None or econ_df.empty or "DATE" not in econ_df.columns:
            self.econ_monthly_df = None
            self._econ_columns = []
            return
        df = econ_df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"]).dt.to_period("M").dt.to_timestamp()
        # Identify numeric econ columns
        econ_cols = [c for c in df.columns if c != "DATE"]
        for c in econ_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # Aggregate to month in case duplicates exist
        df = df.groupby("DATE", as_index=False).mean(numeric_only=True)
        self.econ_monthly_df = df
        self._econ_columns = [c for c in df.columns if c != "DATE"]

    def _merge_econ(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.econ_monthly_df is None or not self._econ_columns:
            return df
        merged = pd.merge(df, self.econ_monthly_df, on="DATE", how="left")
        # fill missing econ with ffill/bfill then 0
        for c in self._econ_columns:
            merged[c] = merged[c].fillna(method="ffill").fillna(method="bfill").fillna(0)
        return merged

    # ---------------------- Queries ----------------------
    def get_available_models(self) -> List[VehicleModelInfo]:
        if self.monthly_data is None:
            if not self.load_data():
                return []
        rows: List[VehicleModelInfo] = []
        for mk, grp in self.monthly_data.groupby("model_key"):
            if grp.empty:
                continue
            rows.append(
                VehicleModelInfo(
                    model_key=str(mk),
                    display_name=self._display_name(mk),
                    first_date=str(grp["DATE"].min().date()),
                    last_date=str(grp["DATE"].max().date()),
                    records=int(len(grp)),
                )
            )
        return rows

    def _display_name(self, model_key: str) -> str:
        return model_key.replace("_", " ").title()

    # ---------------------- Feature Engineering ----------------------
    def _make_features(self, df: pd.DataFrame, target_column: str = "deliveries") -> Tuple[pd.DataFrame, pd.Series, List[str], List[pd.Timestamp]]:
        """
        Create features for forecasting
        Args:
            df: DataFrame with vehicle data
            target_column: Column to forecast ('deliveries', 'production', or 'sold')
        """
        data = df.copy().sort_values("DATE")
        dates = list(pd.to_datetime(data["DATE"]))
        data["month"] = data["DATE"].dt.month
        data["year"] = data["DATE"].dt.year
        data["quarter"] = data["DATE"].dt.quarter
        data["time_index"] = np.arange(len(data))
        
        # Ensure target column exists
        if target_column not in data.columns:
            # Fallback to deliveries if target not available
            target_column = "deliveries"
        
        # Lags based on target
        for lag in [1, 2, 3, 6, 12]:
            data[f"lag_{lag}"] = data[target_column].shift(lag)
        # Rolling
        for window in [3, 6, 12]:
            data[f"rolling_mean_{window}"] = (
                data[target_column].rolling(window=window, min_periods=1).mean()
            )
            data[f"rolling_std_{window}"] = (
                data[target_column].rolling(window=window, min_periods=1).std().fillna(0)
            )
        # Seasonality
        data["sin_month"] = np.sin(2 * np.pi * data["month"] / 12)
        data["cos_month"] = np.cos(2 * np.pi * data["month"] / 12)
        
        # Cross-features: Include other metrics as features if available
        # For deliveries forecast, include production as feature
        if target_column == "deliveries" and "production" in data.columns:
            data["prod_feature"] = data["production"].fillna(method="ffill").fillna(0)
        # For production forecast, include deliveries as feature  
        elif target_column == "production" and "deliveries" in data.columns:
            data["deliv_feature"] = data["deliveries"].fillna(method="ffill").fillna(0)
        # For sold forecast, include both if available
        elif target_column == "sold":
            if "deliveries" in data.columns:
                data["deliv_feature"] = data["deliveries"].fillna(method="ffill").fillna(0)
            if "production" in data.columns:
                data["prod_feature"] = data["production"].fillna(method="ffill").fillna(0)
        
        feature_columns = [
            "time_index",
            "month",
            "year",
            "quarter",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_6",
            "lag_12",
            "rolling_mean_3",
            "rolling_mean_6",
            "rolling_mean_12",
            "rolling_std_3",
            "rolling_std_6",
            "rolling_std_12",
            "sin_month",
            "cos_month",
            "prod_feature" if "prod_feature" in data.columns else None,
            "deliv_feature" if "deliv_feature" in data.columns else None,
        ]
        feature_columns = [c for c in feature_columns if c is not None]
        # If econ variables exist in data, append them (only used when multivariate trains)
        econ_cols_in_df = [c for c in self._econ_columns if c in data.columns]
        feature_columns_extended = feature_columns + econ_cols_in_df
        X = data[feature_columns_extended].fillna(0)
        y = data[target_column].fillna(0)
        return X, y, feature_columns_extended, dates

    # ---------------------- Training ----------------------
    def _get_model_frame(self, model_key: str) -> Optional[pd.DataFrame]:
        if self.monthly_data is None:
            if not self.load_data():
                return None
        df = self.monthly_data[self.monthly_data["model_key"] == model_key].copy()
        if df.empty:
            return None
        # Require minimum history
        if len(df) < 12:
            return None
        return df

    def _split_train_test(self, X: pd.DataFrame, y: pd.Series, dates: List[pd.Timestamp], test_window: Optional[int] = None) -> Tuple:
        n = len(X)
        if test_window in (3, 6) and n > test_window:
            test_size = int(test_window)
        else:
            test_size = 6 if n >= 18 else max(3, n // 5)
        split = n - test_size
        return (
            X.iloc[:split], y.iloc[:split], dates[:split],
            X.iloc[split:], y.iloc[split:], dates[split:]
        )

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred))) if len(y_true) > 0 else 0.0
        denom = np.where(np.abs(y_true) < 1e-6, 1.0, np.abs(y_true))
        mape = float(np.mean(np.abs((y_true - y_pred) / denom))) if len(y_true) > 0 else 0.0
        return {"mae": mae, "mape": mape}

    def train_univariate(self, model_key: str, test_window: Optional[int] = None, target_column: str = "deliveries"):
        df = self._get_model_frame(model_key)
        if df is None:
            return None
        X_all, y_all, feature_columns, dates = self._make_features(df, target_column)
        X_train, y_train, dates_train, X_test, y_test, dates_test = self._split_train_test(X_all, y_all, dates, test_window)
        model = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=12)
        model.fit(X_train, y_train)
        # Evaluate on test
        y_pred_test = model.predict(X_test) if len(X_test) > 0 else np.array([])
        metrics = self._compute_metrics(y_test, y_pred_test) if len(y_pred_test) > 0 else {"mae": 0.0, "mape": 0.0}
        importance = dict(zip(feature_columns, model.feature_importances_))
        last_row = X_all.iloc[-1].to_dict()
        last_date = df["DATE"].max()
        return {
            "model": model,
            "feature_columns": feature_columns,
            "feature_importance": importance,
            "last_row": last_row,
            "last_date": last_date,
            "model_type": "univariate",
            "target_column": target_column,
            "history": {"dates": [str(pd.to_datetime(d).date()) for d in dates], "actuals": [float(v) for v in y_all.values]},
            "test": {
                "dates": [str(pd.to_datetime(d).date()) for d in dates_test],
                "actuals": [float(v) for v in y_test.values],
                "predictions": [float(v) for v in y_pred_test] if len(X_test) > 0 else [],
                "metrics": metrics,
            },
        }

    def train_multivariate(self, model_key: str, test_window: Optional[int] = None, target_column: str = "deliveries"):
        # Merge econ variables if available
        df = self._get_model_frame(model_key)
        if df is None:
            return None
        if self.econ_monthly_df is not None and self._econ_columns:
            df = self._merge_econ(df)
        X_all, y_all, feature_columns, dates = self._make_features(df, target_column)
        X_train, y_train, dates_train, X_test, y_test, dates_test = self._split_train_test(X_all, y_all, dates, test_window)
        X_scaled = self.scaler.fit_transform(X_train)
        model = RandomForestRegressor(n_estimators=160, random_state=42, max_depth=14)
        model.fit(X_scaled, y_train)
        # Evaluate
        X_test_scaled = self.scaler.transform(X_test) if len(X_test) > 0 else np.zeros((0, X_train.shape[1]))
        y_pred_test = model.predict(X_test_scaled) if len(X_test) > 0 else np.array([])
        metrics = self._compute_metrics(y_test, y_pred_test) if len(y_pred_test) > 0 else {"mae": 0.0, "mape": 0.0}
        importance = dict(zip(feature_columns, model.feature_importances_))
        last_row = X_all.iloc[-1].to_dict()
        last_date = df["DATE"].max()
        return {
            "model": model,
            "feature_columns": feature_columns,
            "feature_importance": importance,
            "last_row": last_row,
            "last_date": last_date,
            "model_type": "multivariate",
            "target_column": target_column,
            "scaler": self.scaler,
            "history": {"dates": [str(pd.to_datetime(d).date()) for d in dates], "actuals": [float(v) for v in y_all.values]},
            "test": {
                "dates": [str(pd.to_datetime(d).date()) for d in dates_test],
                "actuals": [float(v) for v in y_test.values],
                "predictions": [float(v) for v in y_pred_test] if len(X_test) > 0 else [],
                "metrics": metrics,
            },
        }

    # ---------------------- Forecasting ----------------------
    def generate_forecast(self, model_key: str, forecast_type: str, months_ahead: int = 12, test_window: Optional[int] = None) -> Optional[Dict]:
        if months_ahead < 1 or months_ahead > 24:
            months_ahead = 12
        if forecast_type == "multivariate":
            info = self.train_multivariate(model_key, test_window)
        else:
            info = self.train_univariate(model_key, test_window)
        if info is None:
            return None

        feature_columns = info["feature_columns"]
        model = info["model"]
        last_row = info["last_row"].copy()
        last_date = info["last_date"]

        forecasts: List[Dict] = []
        for i in range(1, months_ahead + 1):
            # Roll time features
            time_index = last_row.get("time_index", 0) + 1
            # Compute next month
            future_date = (last_date + pd.DateOffset(months=1)) if isinstance(last_date, pd.Timestamp) else pd.Timestamp.now()
            month = future_date.month
            year = future_date.year
            quarter = (month - 1) // 3 + 1
            sin_month = np.sin(2 * np.pi * month / 12)
            cos_month = np.cos(2 * np.pi * month / 12)

            # Lags: simulate using previous prediction
            lag_1 = forecasts[-1]["forecast"] if forecasts else last_row.get("lag_1", last_row.get("rolling_mean_3", 0))
            lag_2 = last_row.get("lag_1", lag_1)
            lag_3 = last_row.get("lag_2", lag_2)
            lag_6 = last_row.get("lag_5", lag_3)
            lag_12 = last_row.get("lag_11", lag_6)

            # Rolling means/stds: approximate with last known
            rolling_mean_3 = last_row.get("rolling_mean_3", lag_1)
            rolling_mean_6 = last_row.get("rolling_mean_6", lag_1)
            rolling_mean_12 = last_row.get("rolling_mean_12", lag_1)
            rolling_std_3 = last_row.get("rolling_std_3", 0)
            rolling_std_6 = last_row.get("rolling_std_6", 0)
            rolling_std_12 = last_row.get("rolling_std_12", 0)

            feat = {
                "time_index": time_index,
                "month": month,
                "year": year,
                "quarter": quarter,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_6": lag_6,
                "lag_12": lag_12,
                "rolling_mean_3": rolling_mean_3,
                "rolling_mean_6": rolling_mean_6,
                "rolling_mean_12": rolling_mean_12,
                "rolling_std_3": rolling_std_3,
                "rolling_std_6": rolling_std_6,
                "rolling_std_12": rolling_std_12,
                "sin_month": sin_month,
                "cos_month": cos_month,
            }
            # Optional prod feature passthrough
            if "prod" in feature_columns:
                feat["prod"] = last_row.get("prod", 0)
            # Econ features
            for econ_col in self._econ_columns:
                if econ_col in feature_columns:
                    feat[econ_col] = last_row.get(econ_col, 0)

            # Ordered vector
            vec = [feat.get(c, 0) for c in feature_columns]

            if info["model_type"] == "multivariate":
                vec = info["scaler"].transform([vec])
                yhat = float(model.predict(vec)[0])
            else:
                yhat = float(model.predict([vec])[0])

            yhat = max(0.0, yhat)
            forecasts.append({
                "date": future_date.strftime("%Y-%m-%d"),
                "forecast": yhat,
                "month_ahead": i,
            })

            # Update last_row/last_date for next step
            last_row.update(feat)
            last_row["lag_1"] = yhat
            last_date = future_date

        return {
            "model_key": model_key,
            "forecast_type": info["model_type"],
            "forecasts": forecasts,
            "history": info.get("history", {}),
            "test_evaluation": info.get("test", {}),
            "model_metrics": {
                "feature_importance": info["feature_importance"],
                "model_type": info["model_type"],
                "mae": info.get("test", {}).get("metrics", {}).get("mae", 0.0),
                "mape": info.get("test", {}).get("metrics", {}).get("mape", 0.0),
            },
        }
    
    def generate_multi_target_forecast(self, model_key: str, forecast_type: str, months_ahead: int = 12, test_window: Optional[int] = None) -> Optional[Dict]:
        """
        Generate forecasts for multiple targets: deliveries, production, and sold quantity
        Returns a comprehensive forecast with all three metrics
        """
        df = self._get_model_frame(model_key)
        if df is None:
            return None
        
        # Determine which targets are available in the data
        available_targets = []
        if "deliveries" in df.columns and df["deliveries"].notna().sum() > 12:
            available_targets.append("deliveries")
        if "production" in df.columns and df["production"].notna().sum() > 12:
            available_targets.append("production")
        if "sold" in df.columns and df["sold"].notna().sum() > 12:
            available_targets.append("sold")
        
        if not available_targets:
            return None
        
        # Generate forecast for each available target
        forecasts_by_target = {}
        
        for target in available_targets:
            forecast_result = self.generate_forecast(
                model_key=model_key,
                forecast_type=forecast_type,
                months_ahead=months_ahead,
                test_window=test_window
            )
            
            if forecast_result:
                # Retrain with specific target column
                if forecast_type == "multivariate":
                    info = self.train_multivariate(model_key, test_window, target)
                else:
                    info = self.train_univariate(model_key, test_window, target)
                
                if info:
                    # Generate forecast with the target-specific model
                    feature_columns = info["feature_columns"]
                    model = info["model"]
                    last_row = info["last_row"].copy()
                    last_date = info["last_date"]
                    
                    target_forecasts = []
                    for i in range(1, months_ahead + 1):
                        time_index = last_row.get("time_index", 0) + 1
                        future_date = (last_date + pd.DateOffset(months=1)) if isinstance(last_date, pd.Timestamp) else pd.Timestamp.now()
                        month = future_date.month
                        year = future_date.year
                        quarter = (month - 1) // 3 + 1
                        sin_month = np.sin(2 * np.pi * month / 12)
                        cos_month = np.cos(2 * np.pi * month / 12)
                        
                        lag_1 = target_forecasts[-1]["forecast"] if target_forecasts else last_row.get("lag_1", last_row.get("rolling_mean_3", 0))
                        lag_2 = last_row.get("lag_1", lag_1)
                        lag_3 = last_row.get("lag_2", lag_2)
                        lag_6 = last_row.get("lag_5", lag_3)
                        lag_12 = last_row.get("lag_11", lag_6)
                        
                        rolling_mean_3 = last_row.get("rolling_mean_3", lag_1)
                        rolling_mean_6 = last_row.get("rolling_mean_6", lag_1)
                        rolling_mean_12 = last_row.get("rolling_mean_12", lag_1)
                        rolling_std_3 = last_row.get("rolling_std_3", 0)
                        rolling_std_6 = last_row.get("rolling_std_6", 0)
                        rolling_std_12 = last_row.get("rolling_std_12", 0)
                        
                        feat = {
                            "time_index": time_index,
                            "month": month,
                            "year": year,
                            "quarter": quarter,
                            "lag_1": lag_1,
                            "lag_2": lag_2,
                            "lag_3": lag_3,
                            "lag_6": lag_6,
                            "lag_12": lag_12,
                            "rolling_mean_3": rolling_mean_3,
                            "rolling_mean_6": rolling_mean_6,
                            "rolling_mean_12": rolling_mean_12,
                            "rolling_std_3": rolling_std_3,
                            "rolling_std_6": rolling_std_6,
                            "rolling_std_12": rolling_std_12,
                            "sin_month": sin_month,
                            "cos_month": cos_month,
                        }
                        
                        if "prod_feature" in feature_columns:
                            feat["prod_feature"] = last_row.get("prod_feature", 0)
                        if "deliv_feature" in feature_columns:
                            feat["deliv_feature"] = last_row.get("deliv_feature", 0)
                        
                        for econ_col in self._econ_columns:
                            if econ_col in feature_columns:
                                feat[econ_col] = last_row.get(econ_col, 0)
                        
                        vec = [feat.get(c, 0) for c in feature_columns]
                        
                        if info["model_type"] == "multivariate":
                            vec = info["scaler"].transform([vec])
                            yhat = float(model.predict(vec)[0])
                        else:
                            yhat = float(model.predict([vec])[0])
                        
                        yhat = max(0.0, yhat)
                        target_forecasts.append({
                            "date": future_date.strftime("%Y-%m-%d"),
                            "forecast": yhat,
                            "month_ahead": i,
                        })
                        
                        last_row.update(feat)
                        last_row["lag_1"] = yhat
                        last_date = future_date
                    
                    forecasts_by_target[target] = {
                        "forecasts": target_forecasts,
                        "history": info.get("history", {}),
                        "test_evaluation": info.get("test", {}),
                        "model_metrics": {
                            "mae": info.get("test", {}).get("metrics", {}).get("mae", 0.0),
                            "mape": info.get("test", {}).get("metrics", {}).get("mape", 0.0),
                        }
                    }
        
        if not forecasts_by_target:
            return None
        
        return {
            "model_key": model_key,
            "forecast_type": forecast_type,
            "available_targets": available_targets,
            "forecasts_by_target": forecasts_by_target
        }
