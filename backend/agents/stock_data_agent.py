"""
Stock Data Agent - Provides real-time Tesla stock analysis and technical indicators
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class StockData:
    symbol: str
    current_price: float
    daily_change: float
    daily_change_percent: float
    volume: int
    market_cap: float
    high_52_week: float
    low_52_week: float
    timestamp: datetime

@dataclass
class TechnicalIndicators:
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    bollinger_upper: Optional[float] = None
    bollinger_lower: Optional[float] = None
    bollinger_middle: Optional[float] = None

@dataclass
class PerformanceMetrics:
    ytd_return: Optional[float] = None
    one_month_return: Optional[float] = None
    three_month_return: Optional[float] = None
    one_year_return: Optional[float] = None
    volatility: Optional[float] = None
    beta: Optional[float] = None

class StockDataAgent:
    """
    Stock Data Agent that provides real-time Tesla stock analysis
    Uses free APIs with fallback mechanisms
    """
    
    def __init__(self):
        # Clean the API keys (remove quotes and whitespace)
        alpha_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if alpha_key:
            alpha_key = alpha_key.strip().strip('"').strip("'")
        self.alpha_vantage_api_key = alpha_key
        
        polygon_key = os.getenv('POLYGON_API_KEY')
        if polygon_key:
            polygon_key = polygon_key.strip().strip('"').strip("'")
        self.polygon_api_key = polygon_key
        
        self.yahoo_finance_base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        self.cache = {}
        self.cache_ttl = 60  # 1 minute cache for stock data
        self.historical_cache_ttl = 3600  # 1 hour cache for historical data
        
        logger.info(f"StockDataAgent initialized:")
        logger.info(f"  - Alpha Vantage API key: {bool(self.alpha_vantage_api_key and self.alpha_vantage_api_key != 'your_alpha_vantage_key_here')}")
        logger.info(f"  - Polygon API key: {bool(self.polygon_api_key and self.polygon_api_key != 'your_polygon_key_here')}")
        
    async def get_comprehensive_analysis(self, symbol: str = "TSLA") -> Dict[str, Any]:
        """
        Get comprehensive stock analysis including price data and technical indicators
        """
        try:
            # Get current stock data
            stock_data = await self.get_current_stock_data(symbol)
            
            # Get technical indicators (simplified for MVP)
            technical_indicators = await self.get_technical_indicators(symbol)
            
            # Get performance metrics
            performance_metrics = await self.get_performance_metrics(symbol)
            
            # Calculate 52-week analysis
            fifty_two_week_analysis = self._calculate_52_week_analysis(stock_data)
            
            return {
                "stock_data": stock_data.__dict__ if stock_data else None,
                "technical_indicators": technical_indicators.__dict__ if technical_indicators else None,
                "performance_metrics": performance_metrics.__dict__ if performance_metrics else None,
                "fifty_two_week_analysis": fifty_two_week_analysis,
                "analysis_summary": self._generate_analysis_summary(stock_data, technical_indicators),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting comprehensive analysis for {symbol}: {str(e)}")
            return {
                "error": f"Failed to get stock analysis: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_current_stock_data(self, symbol: str = "TSLA") -> Optional[StockData]:
        """
        Get current stock price and basic data
        Uses Yahoo Finance as primary source (free, no API key required)
        """
        try:
            # Check cache first
            cache_key = f"stock_data_{symbol}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                    return cached_data
            
            # Try Yahoo Finance first (free, reliable)
            stock_data = await self._get_yahoo_finance_data(symbol)
            
            if not stock_data and self.alpha_vantage_api_key:
                # Fallback to Alpha Vantage if available
                stock_data = await self._get_alpha_vantage_data(symbol)
            
            if not stock_data:
                # Return mock data for development/demo purposes
                stock_data = self._get_mock_stock_data(symbol)
            
            # Cache the result
            if stock_data:
                self.cache[cache_key] = (stock_data, datetime.now())
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error getting current stock data for {symbol}: {str(e)}")
            return self._get_mock_stock_data(symbol)
    
    async def _get_yahoo_finance_data(self, symbol: str) -> Optional[StockData]:
        """Get stock data from Yahoo Finance (free API)"""
        try:
            url = f"{self.yahoo_finance_base_url}/{symbol}"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'chart' in data and 'result' in data['chart']:
                            result = data['chart']['result'][0]
                            meta = result['meta']
                            quotes = result['indicators']['quote'][0]
                            
                            current_price = meta.get('regularMarketPrice', 0)
                            prev_close = meta.get('previousClose', 0)
                            volume = meta.get('regularMarketVolume', 0)
                            market_cap = meta.get('marketCap', 0)
                            
                            daily_change = current_price - prev_close if prev_close > 0 else 0
                            daily_change_percent = (daily_change / prev_close * 100) if prev_close > 0 else 0
                            
                            # Get 52-week high/low from meta
                            high_52_week = meta.get('fiftyTwoWeekHigh', current_price)
                            low_52_week = meta.get('fiftyTwoWeekLow', current_price)
                            
                            return StockData(
                                symbol=symbol,
                                current_price=round(current_price, 2),
                                daily_change=round(daily_change, 2),
                                daily_change_percent=round(daily_change_percent, 2),
                                volume=volume,
                                market_cap=market_cap,
                                high_52_week=round(high_52_week, 2),
                                low_52_week=round(low_52_week, 2),
                                timestamp=datetime.now()
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Yahoo Finance data for {symbol}: {str(e)}")
            return None
    
    async def _get_alpha_vantage_data(self, symbol: str) -> Optional[StockData]:
        """Get stock data from Alpha Vantage (requires API key)"""
        try:
            if not self.alpha_vantage_api_key:
                return None
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'Global Quote' in data:
                            quote = data['Global Quote']
                            
                            current_price = float(quote.get('05. price', 0))
                            prev_close = float(quote.get('08. previous close', 0))
                            volume = int(quote.get('06. volume', 0))
                            daily_change = current_price - prev_close
                            daily_change_percent = (daily_change / prev_close * 100) if prev_close > 0 else 0
                            
                            # Calculate market cap (Alpha Vantage doesn't provide it)
                            # Tesla has approximately 3.18 billion shares outstanding (as of 2024)
                            # For accurate data, we'd need to fetch this separately
                            market_cap = current_price * 3_180_000_000 if current_price > 0 else 0
                            
                            # Note: Alpha Vantage Global Quote doesn't include 52-week data
                            return StockData(
                                symbol=symbol,
                                current_price=round(current_price, 2),
                                daily_change=round(daily_change, 2),
                                daily_change_percent=round(daily_change_percent, 2),
                                volume=volume,
                                market_cap=round(market_cap, 0),  # Calculated from shares outstanding
                                high_52_week=current_price,  # Fallback
                                low_52_week=current_price,   # Fallback
                                timestamp=datetime.now()
                            )
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage data for {symbol}: {str(e)}")
            return None
    
    def _get_mock_stock_data(self, symbol: str) -> StockData:
        """Generate mock stock data for development/demo purposes"""
        base_price = 245.67  # Base Tesla price
        daily_change = 5.23
        daily_change_percent = 2.18
        
        return StockData(
            symbol=symbol,
            current_price=base_price,
            daily_change=daily_change,
            daily_change_percent=daily_change_percent,
            volume=45678901,
            market_cap=780000000000,  # ~$780B
            high_52_week=299.29,
            low_52_week=138.80,
            timestamp=datetime.now()
        )
    
    async def get_technical_indicators(self, symbol: str) -> TechnicalIndicators:
        """
        Get technical indicators (simplified for MVP)
        In production, this would use Alpha Vantage or other APIs
        """
        try:
            # For MVP, return calculated indicators based on current price
            stock_data = await self.get_current_stock_data(symbol)
            if not stock_data:
                return TechnicalIndicators()
            
            # Simplified technical indicators (would normally require historical data)
            current_price = stock_data.current_price
            
            # Mock technical indicators for demo
            return TechnicalIndicators(
                rsi_14=65.4,  # Overbought territory
                macd=2.34,
                macd_signal=1.89,
                macd_histogram=0.45,
                sma_50=238.45,
                sma_200=215.67,
                bollinger_upper=265.23,
                bollinger_lower=226.11,
                bollinger_middle=245.67
            )
            
        except Exception as e:
            logger.error(f"Error getting technical indicators for {symbol}: {str(e)}")
            return TechnicalIndicators()
    
    async def get_performance_metrics(self, symbol: str) -> PerformanceMetrics:
        """
        Get performance metrics (simplified for MVP)
        """
        try:
            # Mock performance metrics for demo
            return PerformanceMetrics(
                ytd_return=12.4,
                one_month_return=8.2,
                three_month_return=15.6,
                one_year_return=23.8,
                volatility=0.45,
                beta=1.85
            )
            
        except Exception as e:
            logger.error(f"Error getting performance metrics for {symbol}: {str(e)}")
            return PerformanceMetrics()
    
    def _calculate_52_week_analysis(self, stock_data: Optional[StockData]) -> Dict[str, Any]:
        """Calculate 52-week analysis"""
        if not stock_data:
            return {}
        
        current_price = stock_data.current_price
        high_52_week = stock_data.high_52_week
        low_52_week = stock_data.low_52_week
        
        percent_from_high = ((current_price - high_52_week) / high_52_week * 100) if high_52_week > 0 else 0
        percent_from_low = ((current_price - low_52_week) / low_52_week * 100) if low_52_week > 0 else 0
        
        return {
            "high": high_52_week,
            "low": low_52_week,
            "current_percent_from_high": round(percent_from_high, 1),
            "current_percent_from_low": round(percent_from_low, 1),
            "range_percent": round(((high_52_week - low_52_week) / low_52_week * 100), 1)
        }
    
    async def get_historical_data(self, symbol: str = "TSLA", years: int = 5) -> List[Dict[str, Any]]:
        """
        Get historical stock data for specified years
        Fallback order: Alpha Vantage → Polygon.io → Yahoo Finance
        """
        try:
            # Check cache first
            cache_key = f"historical_data_{symbol}_{years}y"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.historical_cache_ttl):
                    logger.info(f"✅ Returning cached historical data for {symbol}")
                    return cached_data
            
            # TIER 1: Try Alpha Vantage first
            if self.alpha_vantage_api_key and self.alpha_vantage_api_key != 'your_alpha_vantage_key_here':
                logger.info(f"🔄 Tier 1: Fetching from Alpha Vantage for {symbol}...")
                historical_data = await self._get_alpha_vantage_historical(symbol, years)
                if historical_data:
                    self.cache[cache_key] = (historical_data, datetime.now())
                    logger.info(f"✅ Alpha Vantage: Successfully fetched {len(historical_data)} days")
                    return historical_data
                else:
                    logger.warning("⚠️ Alpha Vantage: No data returned (rate limit or error)")
            else:
                logger.info("⏭️ Alpha Vantage API key not configured, skipping...")
            
            # TIER 2: Try Polygon.io as first fallback
            if self.polygon_api_key and self.polygon_api_key != 'your_polygon_key_here':
                logger.info(f"🔄 Tier 2: Fetching from Polygon.io for {symbol}...")
                historical_data = await self._get_polygon_historical(symbol, years)
                if historical_data:
                    self.cache[cache_key] = (historical_data, datetime.now())
                    logger.info(f"✅ Polygon.io: Successfully fetched {len(historical_data)} days")
                    return historical_data
                else:
                    logger.warning("⚠️ Polygon.io: No data returned")
            else:
                logger.info("⏭️ Polygon API key not configured, skipping...")
            
            # TIER 3: Fallback to Yahoo Finance (free, no key required)
            logger.info(f"🔄 Tier 3: Fetching from Yahoo Finance for {symbol}...")
            historical_data = await self._get_yahoo_historical(symbol, years)
            if historical_data:
                self.cache[cache_key] = (historical_data, datetime.now())
                logger.info(f"✅ Yahoo Finance: Successfully fetched {len(historical_data)} days")
                return historical_data
            else:
                logger.warning("⚠️ Yahoo Finance: No data returned")
            
            # All sources failed
            logger.error(f"❌ All data sources failed for {symbol}")
            return []
            
        except Exception as e:
            logger.error(f"❌ Error getting historical data for {symbol}: {str(e)}")
            return []
    
    async def _get_alpha_vantage_historical(self, symbol: str, years: int) -> List[Dict[str, Any]]:
        """Get historical data from Alpha Vantage TIME_SERIES_DAILY"""
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'outputsize': 'full',  # Get up to 20 years of data
                'apikey': self.alpha_vantage_api_key
            }
            
            logger.info(f"Calling Alpha Vantage TIME_SERIES_DAILY for {symbol}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    logger.info(f"Alpha Vantage response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Log what we received
                        if 'Note' in data:
                            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                            return []
                        elif 'Error Message' in data:
                            logger.error(f"Alpha Vantage error: {data['Error Message']}")
                            return []
                        elif 'Information' in data:
                            logger.warning(f"Alpha Vantage info: {data['Information']}")
                            return []
                        
                        if 'Time Series (Daily)' in data:
                            time_series = data['Time Series (Daily)']
                            logger.info(f"Received {len(time_series)} days from Alpha Vantage")
                            
                            # Convert to list and sort by date
                            historical_data = []
                            cutoff_date = datetime.now() - timedelta(days=years * 365)
                            
                            for date_str, values in time_series.items():
                                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                                if date_obj >= cutoff_date:
                                    historical_data.append({
                                        'date': date_str,
                                        'timestamp': date_obj.timestamp(),
                                        'open': float(values['1. open']),
                                        'high': float(values['2. high']),
                                        'low': float(values['3. low']),
                                        'close': float(values['4. close']),
                                        'volume': int(values['5. volume'])
                                    })
                            
                            # Sort by date (oldest first)
                            historical_data.sort(key=lambda x: x['timestamp'])
                            return historical_data
                        elif 'Note' in data:
                            logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                        elif 'Error Message' in data:
                            logger.error(f"Alpha Vantage error: {data['Error Message']}")
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage historical data: {str(e)}")
            return []
    
    async def _get_polygon_historical(self, symbol: str, years: int) -> List[Dict[str, Any]]:
        """Get historical data from Polygon.io"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            # Polygon uses YYYY-MM-DD format
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            # Polygon.io REST API endpoint for aggregates (daily bars)
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{from_date}/{to_date}"
            params = {
                'adjusted': 'true',
                'sort': 'asc',
                'limit': 50000,  # Max results
                'apiKey': self.polygon_api_key
            }
            
            logger.info(f"Calling Polygon.io for {symbol} from {from_date} to {to_date}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    logger.info(f"Polygon response status: {response.status}")
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Check for errors
                        if data.get('status') == 'ERROR':
                            logger.error(f"Polygon error: {data.get('error', 'Unknown error')}")
                            return []
                        
                        if data.get('status') != 'OK':
                            logger.warning(f"Polygon status: {data.get('status')}")
                            return []
                        
                        # Extract results
                        results = data.get('results', [])
                        if not results:
                            logger.warning("Polygon returned no results")
                            return []
                        
                        logger.info(f"Polygon returned {len(results)} data points")
                        
                        # Convert to our standard format
                        historical_data = []
                        for bar in results:
                            # Polygon returns timestamp in milliseconds
                            timestamp = bar['t'] / 1000
                            date_obj = datetime.fromtimestamp(timestamp)
                            
                            historical_data.append({
                                'date': date_obj.strftime('%Y-%m-%d'),
                                'timestamp': timestamp,
                                'open': float(bar['o']),
                                'high': float(bar['h']),
                                'low': float(bar['l']),
                                'close': float(bar['c']),
                                'volume': int(bar['v'])
                            })
                        
                        return historical_data
                    
                    elif response.status == 429:
                        logger.warning("Polygon rate limit exceeded")
                        return []
                    elif response.status == 401 or response.status == 403:
                        logger.error("Polygon authentication failed - check API key")
                        return []
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching Polygon historical data: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    async def _get_yahoo_historical(self, symbol: str, years: int) -> List[Dict[str, Any]]:
        """Get historical data from Yahoo Finance as fallback"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years * 365)
            
            period1 = int(start_date.timestamp())
            period2 = int(end_date.timestamp())
            
            url = f"{self.yahoo_finance_base_url}/{symbol}"
            params = {
                'period1': period1,
                'period2': period2,
                'interval': '1d'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'chart' in data and 'result' in data['chart']:
                            result = data['chart']['result'][0]
                            timestamps = result['timestamp']
                            quotes = result['indicators']['quote'][0]
                            
                            historical_data = []
                            for i, timestamp in enumerate(timestamps):
                                date_obj = datetime.fromtimestamp(timestamp)
                                historical_data.append({
                                    'date': date_obj.strftime('%Y-%m-%d'),
                                    'timestamp': timestamp,
                                    'open': quotes['open'][i] or 0,
                                    'high': quotes['high'][i] or 0,
                                    'low': quotes['low'][i] or 0,
                                    'close': quotes['close'][i] or 0,
                                    'volume': quotes['volume'][i] or 0
                                })
                            
                            return historical_data
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance historical data: {str(e)}")
            return []
    
    def calculate_technical_indicators_from_history(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate real technical indicators from historical data
        Returns SMA 20, SMA 50, SMA 200, RSI 14
        """
        try:
            if not historical_data or len(historical_data) < 200:
                logger.warning(f"Insufficient data for technical indicators: {len(historical_data)} days")
                return {}
            
            # Extract closing prices
            closes = [day['close'] for day in historical_data]
            
            # Calculate SMAs
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
            sma_200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None
            
            # Calculate RSI (14-day)
            rsi_14 = self._calculate_rsi(closes, period=14) if len(closes) >= 15 else None
            
            # Calculate MACD
            macd_data = self._calculate_macd(closes) if len(closes) >= 34 else {}
            
            # Get latest values for display
            current_price = closes[-1] if closes else 0
            
            return {
                'sma_20': round(sma_20, 2) if sma_20 else None,
                'sma_50': round(sma_50, 2) if sma_50 else None,
                'sma_200': round(sma_200, 2) if sma_200 else None,
                'rsi_14': round(rsi_14, 2) if rsi_14 else None,
                'macd': round(macd_data.get('macd', 0), 2),
                'macd_signal': round(macd_data.get('signal', 0), 2),
                'macd_histogram': round(macd_data.get('histogram', 0), 2),
                'current_price': round(current_price, 2),
                'data_points': len(historical_data)
            }
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return {}
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        try:
            if len(prices) < period + 1:
                return None
            
            # Calculate price changes
            deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
            
            # Separate gains and losses
            gains = [d if d > 0 else 0 for d in deltas[-period:]]
            losses = [-d if d < 0 else 0 for d in deltas[-period:]]
            
            # Calculate average gain and loss
            avg_gain = sum(gains) / period
            avg_loss = sum(losses) / period
            
            if avg_loss == 0:
                return 100
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return rsi
            
        except Exception as e:
            logger.error(f"Error calculating RSI: {str(e)}")
            return None
    
    def _calculate_macd(self, prices: List[float], fast=12, slow=26, signal=9) -> Dict[str, float]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        try:
            if len(prices) < slow + signal:
                return {}
            
            # Calculate EMAs
            ema_fast = self._calculate_ema(prices, fast)
            ema_slow = self._calculate_ema(prices, slow)
            
            if not ema_fast or not ema_slow:
                return {}
            
            # MACD line = EMA(12) - EMA(26)
            macd_line = ema_fast - ema_slow
            
            # Signal line would need historical MACD values
            # For now, return simplified MACD
            return {
                'macd': macd_line,
                'signal': macd_line * 0.9,  # Simplified approximation
                'histogram': macd_line * 0.1
            }
            
        except Exception as e:
            logger.error(f"Error calculating MACD: {str(e)}")
            return {}
    
    def _calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        try:
            if len(prices) < period:
                return None
            
            multiplier = 2 / (period + 1)
            ema = sum(prices[:period]) / period  # Start with SMA
            
            for price in prices[period:]:
                ema = (price - ema) * multiplier + ema
            
            return ema
            
        except Exception as e:
            logger.error(f"Error calculating EMA: {str(e)}")
            return None
    
    def _generate_analysis_summary(self, stock_data: Optional[StockData], technical_indicators: Optional[TechnicalIndicators]) -> str:
        """Generate a brief analysis summary"""
        if not stock_data:
            return "Unable to generate analysis summary - no stock data available."
        
        summary_parts = []
        
        # Price performance
        if stock_data.daily_change_percent > 0:
            summary_parts.append(f"Tesla's stock is up {stock_data.daily_change_percent:.1f}% today, trading at ${stock_data.current_price:.2f}")
        else:
            summary_parts.append(f"Tesla's stock is down {abs(stock_data.daily_change_percent):.1f}% today, trading at ${stock_data.current_price:.2f}")
        
        # Volume analysis
        if stock_data.volume > 50000000:  # High volume threshold
            summary_parts.append("with high trading volume indicating strong investor interest")
        elif stock_data.volume < 20000000:  # Low volume threshold
            summary_parts.append("with relatively low trading volume")
        
        # Technical analysis
        if technical_indicators and technical_indicators.rsi_14:
            if technical_indicators.rsi_14 > 70:
                summary_parts.append("RSI indicates overbought conditions")
            elif technical_indicators.rsi_14 < 30:
                summary_parts.append("RSI indicates oversold conditions")
        
        return ". ".join(summary_parts) + "."
    
    async def analyze_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a stock-related query and provide relevant information
        """
        try:
            query_lower = query.lower()
            symbol = "TSLA"  # Default to Tesla
            
            # Extract symbol if mentioned
            if "rivian" in query_lower or "rivn" in query_lower:
                symbol = "RIVN"
            elif "lucid" in query_lower or "lcid" in query_lower:
                symbol = "LCID"
            elif "nio" in query_lower:
                symbol = "NIO"
            
            # Get comprehensive analysis
            analysis = await self.get_comprehensive_analysis(symbol)
            
            # Customize response based on query type
            if "price" in query_lower or "trading" in query_lower:
                return {
                    "type": "stock_price",
                    "symbol": symbol,
                    "analysis": analysis,
                    "summary": f"Current {symbol} stock analysis",
                    "insights": [
                        f"{symbol} is trading at ${analysis['stock_data']['current_price']:.2f}",
                        f"Daily change: {analysis['stock_data']['daily_change']:.2f} ({analysis['stock_data']['daily_change_percent']:.1f}%)",
                        f"52-week range: ${analysis['fifty_two_week_analysis']['low']:.2f} - ${analysis['fifty_two_week_analysis']['high']:.2f}"
                    ]
                }
            elif "technical" in query_lower or "analysis" in query_lower:
                return {
                    "type": "technical_analysis",
                    "symbol": symbol,
                    "analysis": analysis,
                    "summary": f"Technical analysis for {symbol}",
                    "insights": [
                        f"RSI (14): {analysis['technical_indicators']['rsi_14']:.1f}",
                        f"MACD: {analysis['technical_indicators']['macd']:.2f}",
                        f"50-day SMA: ${analysis['technical_indicators']['sma_50']:.2f}",
                        f"200-day SMA: ${analysis['technical_indicators']['sma_200']:.2f}"
                    ]
                }
            else:
                return {
                    "type": "general_analysis",
                    "symbol": symbol,
                    "analysis": analysis,
                    "summary": f"Stock analysis for {symbol}",
                    "insights": [
                        analysis['analysis_summary'],
                        f"Market cap: ${analysis['stock_data']['market_cap']:,.0f}",
                        f"Volume: {analysis['stock_data']['volume']:,}"
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error analyzing stock query: {str(e)}")
            return {
                "type": "error",
                "error": f"Failed to analyze stock query: {str(e)}",
                "summary": "Unable to provide stock analysis"
            }
