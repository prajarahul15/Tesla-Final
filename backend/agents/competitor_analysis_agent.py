"""
Competitor Analysis Agent - Compares Tesla with other EV companies
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class CompetitorData:
    symbol: str
    name: str
    current_price: float
    market_cap: float
    pe_ratio: Optional[float]
    revenue_ttm: float
    profit_margin: float
    delivery_count: int
    market_share: float

@dataclass
class ComparisonMetrics:
    metric_name: str
    tesla_value: float
    competitor_values: Dict[str, float]
    tesla_rank: int
    industry_average: float

class CompetitorAnalysisAgent:
    """
    Competitor Analysis Agent that compares Tesla with other EV manufacturers
    """
    
    def __init__(self):
        # Initialize API keys
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip().strip('"').strip("'").split('#')[0]
        self.polygon_key = os.getenv("POLYGON_API_KEY", "").strip().strip('"').strip("'").split('#')[0]
        self.nasdaq_key = os.getenv("NASDAQ_API_KEY", "").strip().strip('"').strip("'").split('#')[0]
        self.tavily_key = os.getenv("TAVILY_API_KEY", "").strip().strip('"').strip("'").split('#')[0]
        self.serper_key = os.getenv("SERPER_API_KEY", "").strip().strip('"').strip("'").split('#')[0]
        
        logger.info(f"CompetitorAnalysisAgent initialized:")
        logger.info(f"  - Alpha Vantage API key: {bool(self.alpha_vantage_key)}")
        logger.info(f"  - Polygon API key: {bool(self.polygon_key)}")
        logger.info(f"  - NASDAQ API key: {bool(self.nasdaq_key)}")
        logger.info(f"  - Tavily API key: {bool(self.tavily_key)}")
        logger.info(f"  - Serper API key: {bool(self.serper_key)}")
        
        self.competitors = {
            "RIVN": {
                "name": "Rivian Automotive",
                "type": "EV Startup",
                "focus": "Electric Trucks & SUVs"
            },
            "LCID": {
                "name": "Lucid Motors",
                "type": "EV Startup",
                "focus": "Luxury Electric Vehicles"
            },
            "NIO": {
                "name": "NIO Inc.",
                "type": "Chinese EV Maker",
                "focus": "Premium Electric SUVs"
            },
            "XPEV": {
                "name": "XPeng",
                "type": "Chinese EV Maker",
                "focus": "Smart Electric Vehicles"
            },
            "LI": {
                "name": "Li Auto",
                "type": "Chinese EV Maker",
                "focus": "Extended-Range EVs"
            },
            "F": {
                "name": "Ford Motor Company",
                "type": "Traditional OEM",
                "focus": "F-150 Lightning, Mustang Mach-E"
            },
            "GM": {
                "name": "General Motors",
                "type": "Traditional OEM",
                "focus": "Ultium Platform EVs"
            }
        }
        
        self.cache = {}
        self.cache_ttl = 600  # 10 minutes cache
    
    async def get_comprehensive_analysis(self, symbol: str = "TSLA", competitors: List[str] = None) -> Dict[str, Any]:
        """
        Get comprehensive competitor analysis
        """
        try:
            # Default competitors if none specified
            if not competitors:
                competitors = ["RIVN", "LCID", "NIO"]
            
            # Check cache
            cache_key = f"competitor_analysis_{symbol}_{'_'.join(sorted(competitors))}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                    return cached_data
            
            # Get data for Tesla
            tesla_data = await self.get_company_data(symbol)
            
            # Get data for competitors
            competitor_data = {}
            for comp_symbol in competitors:
                competitor_data[comp_symbol] = await self.get_company_data(comp_symbol)
            
            # Perform comparative analysis
            comparison = self.perform_comparison(tesla_data, competitor_data)
            
            # Calculate market positioning
            market_position = self.calculate_market_position(tesla_data, competitor_data)
            
            # Generate insights
            insights = self.generate_competitive_insights(tesla_data, competitor_data, comparison)
            
            result = {
                "tesla_data": tesla_data,
                "competitor_data": competitor_data,
                "comparison_metrics": comparison,
                "market_positioning": market_position,
                "insights": insights,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            self.cache[cache_key] = (result, datetime.now())
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting competitor analysis: {str(e)}")
            return self._get_mock_competitor_data()
    
    async def _fetch_real_company_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch real-time company data from multiple API sources
        """
        import aiohttp
        
        company_data = {
            "symbol": symbol,
            "name": self.competitors.get(symbol, {}).get("name", symbol),
            "current_price": 0,
            "market_cap": 0,
            "pe_ratio": None,
            "revenue_ttm": 0,
            "profit_margin": 0,
            "annual_deliveries": 0,
            "delivery_growth_yoy": 0,
            "market_share": 0,
            "ev_models": 0,
            "production_capacity": 0,
            "average_asp": 0,
            "r_and_d_spending": 0,
            "data_sources": []  # Track data sources
        }
        
        try:
            # Tier 1: Try Alpha Vantage for stock price and market cap
            if self.alpha_vantage_key:
                try:
                    async with aiohttp.ClientSession() as session:
                        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={self.alpha_vantage_key}"
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status == 200:
                                data = await response.json()
                                if "Global Quote" in data and data["Global Quote"]:
                                    quote = data["Global Quote"]
                                    company_data["current_price"] = float(quote.get("05. price", 0))
                                    # Calculate market cap (approximate using shares outstanding)
                                    shares_outstanding = self._get_shares_outstanding(symbol)
                                    if company_data["current_price"] > 0 and shares_outstanding > 0:
                                        company_data["market_cap"] = company_data["current_price"] * shares_outstanding
                                    company_data["data_sources"].append({
                                        "source": "Alpha Vantage",
                                        "url": "https://www.alphavantage.co/",
                                        "metrics": ["stock_price", "market_cap"]
                                    })
                                    logger.info(f"Alpha Vantage: Got price {company_data['current_price']} for {symbol}")
                except Exception as e:
                    logger.warning(f"Alpha Vantage failed for {symbol}: {str(e)}")
            
            # Tier 2: Try Polygon for additional data
            if self.polygon_key and company_data["current_price"] == 0:
                try:
                    async with aiohttp.ClientSession() as session:
                        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={self.polygon_key}"
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status == 200:
                                data = await response.json()
                                if "results" in data and data["results"]:
                                    result = data["results"][0]
                                    company_data["current_price"] = float(result.get("c", 0))
                                    shares_outstanding = self._get_shares_outstanding(symbol)
                                    if company_data["current_price"] > 0 and shares_outstanding > 0:
                                        company_data["market_cap"] = company_data["current_price"] * shares_outstanding
                                    company_data["data_sources"].append({
                                        "source": "Polygon.io",
                                        "url": "https://polygon.io/",
                                        "metrics": ["stock_price", "market_cap"]
                                    })
                                    logger.info(f"Polygon: Got price {company_data['current_price']} for {symbol}")
                except Exception as e:
                    logger.warning(f"Polygon failed for {symbol}: {str(e)}")
            
            # Tier 3: Try NASDAQ Data Link for fundamental metrics
            if self.nasdaq_key:
                try:
                    async with aiohttp.ClientSession() as session:
                        # Try to get fundamental data from NASDAQ
                        url = f"https://data.nasdaq.com/api/v3/datasets/WIKI/{symbol}.json?api_key={self.nasdaq_key}&rows=1"
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status == 200:
                                data = await response.json()
                                if "dataset" in data and "data" in data["dataset"]:
                                    dataset = data["dataset"]["data"]
                                    if dataset and len(dataset) > 0:
                                        latest = dataset[0]
                                        # NASDAQ data format: [Date, Open, High, Low, Close, Volume, ...]
                                        if len(latest) >= 5:
                                            company_data["current_price"] = float(latest[4])  # Close price
                                            shares_outstanding = self._get_shares_outstanding(symbol)
                                            if company_data["current_price"] > 0 and shares_outstanding > 0:
                                                company_data["market_cap"] = company_data["current_price"] * shares_outstanding
                                            company_data["data_sources"].append({
                                                "source": "NASDAQ Data Link",
                                                "url": "https://data.nasdaq.com/",
                                                "metrics": ["stock_price"]
                                            })
                                            logger.info(f"NASDAQ: Got price {company_data['current_price']} for {symbol}")
                except Exception as e:
                    logger.warning(f"NASDAQ failed for {symbol}: {str(e)}")
            
            # Tier 4: Try Tavily for comprehensive company research
            if self.tavily_key and company_data.get("annual_deliveries", 0) == 0:
                try:
                    async with aiohttp.ClientSession() as session:
                        url = "https://api.tavily.com/search"
                        headers = {
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "api_key": self.tavily_key,
                            "query": f"{self.competitors.get(symbol, {}).get('name', symbol)} annual deliveries revenue 2024 market share",
                            "search_depth": "advanced",
                            "max_results": 5,
                            "include_answer": True,
                            "include_raw_content": False
                        }
                        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as response:
                            if response.status == 200:
                                data = await response.json()
                                logger.info(f"Tavily: Got research results for {symbol}")
                                
                                # Track Tavily as a source
                                tavily_metrics = []
                                tavily_urls = []
                                
                                # Collect URLs from search results
                                if "results" in data:
                                    for result in data.get("results", [])[:3]:  # Top 3 sources
                                        if "url" in result:
                                            tavily_urls.append(result["url"])
                                
                                # Parse Tavily answer for key metrics
                                if "answer" in data and data["answer"]:
                                    answer = data["answer"].lower()
                                    # Try to extract delivery numbers from the answer
                                    import re
                                    
                                    # Look for delivery/vehicle numbers
                                    delivery_patterns = [
                                        r'(\d{1,3}(?:,\d{3})*)\s*(?:vehicles?|deliveries|units)',
                                        r'delivered?\s*(\d{1,3}(?:,\d{3})*)',
                                        r'sold?\s*(\d{1,3}(?:,\d{3})*)'
                                    ]
                                    
                                    for pattern in delivery_patterns:
                                        matches = re.findall(pattern, answer)
                                        if matches:
                                            try:
                                                # Convert "123,456" to 123456
                                                deliveries = int(matches[0].replace(',', ''))
                                                if deliveries > 1000:  # Sanity check
                                                    company_data["annual_deliveries"] = deliveries
                                                    logger.info(f"Tavily: Extracted {deliveries} deliveries for {symbol}")
                                                    tavily_metrics.append("deliveries")
                                                    break
                                            except ValueError:
                                                pass
                                    
                                    # Look for revenue numbers
                                    revenue_patterns = [
                                        r'\$(\d+(?:\.\d+)?)\s*(?:billion|b)',
                                        r'revenue.*?\$(\d+(?:\.\d+)?)\s*(?:billion|b)',
                                        r'\$(\d+(?:\.\d+)?)\s*(?:million|m)'
                                    ]
                                    
                                    for pattern in revenue_patterns:
                                        matches = re.findall(pattern, answer)
                                        if matches:
                                            try:
                                                revenue = float(matches[0])
                                                # Convert to actual numbers
                                                if 'billion' in answer or ' b' in answer:
                                                    revenue *= 1_000_000_000
                                                elif 'million' in answer or ' m' in answer:
                                                    revenue *= 1_000_000
                                                
                                                if revenue > 1_000_000:  # Sanity check
                                                    company_data["revenue_ttm"] = revenue
                                                    logger.info(f"Tavily: Extracted ${revenue/1e9:.2f}B revenue for {symbol}")
                                                    tavily_metrics.append("revenue")
                                                    break
                                            except ValueError:
                                                pass
                                    
                                    # Look for market share
                                    share_patterns = [
                                        r'(\d+(?:\.\d+)?)\s*%\s*market\s*share',
                                        r'market\s*share.*?(\d+(?:\.\d+)?)\s*%'
                                    ]
                                    
                                    for pattern in share_patterns:
                                        matches = re.findall(pattern, answer)
                                        if matches:
                                            try:
                                                market_share = float(matches[0])
                                                if 0 < market_share < 100:  # Sanity check
                                                    company_data["market_share"] = market_share
                                                    logger.info(f"Tavily: Extracted {market_share}% market share for {symbol}")
                                                    tavily_metrics.append("market_share")
                                                    break
                                            except ValueError:
                                                pass
                                    
                                    # Add Tavily as a data source if we extracted any metrics
                                    if tavily_metrics:
                                        company_data["data_sources"].append({
                                            "source": "Tavily AI Research",
                                            "url": "https://tavily.com/",
                                            "metrics": tavily_metrics,
                                            "references": tavily_urls
                                        })
                                
                except Exception as e:
                    logger.warning(f"Tavily failed for {symbol}: {str(e)}")
            
            # Tier 5: Try Serper for company news and recent data
            if self.serper_key:
                try:
                    async with aiohttp.ClientSession() as session:
                        url = "https://google.serper.dev/search"
                        headers = {
                            "X-API-KEY": self.serper_key,
                            "Content-Type": "application/json"
                        }
                        payload = {
                            "q": f"{symbol} stock price market cap deliveries 2024",
                            "num": 5
                        }
                        async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                            if response.status == 200:
                                data = await response.json()
                                logger.info(f"Serper: Got search results for {symbol}")
                                
                                # Collect search result URLs
                                serper_urls = []
                                if "organic" in data:
                                    for result in data.get("organic", [])[:3]:  # Top 3 results
                                        if "link" in result:
                                            serper_urls.append(result["link"])
                                
                                # Add Serper as a data source
                                if serper_urls:
                                    company_data["data_sources"].append({
                                        "source": "Serper (Google Search)",
                                        "url": "https://serper.dev/",
                                        "metrics": ["news", "company_updates"],
                                        "references": serper_urls
                                    })
                except Exception as e:
                    logger.warning(f"Serper failed for {symbol}: {str(e)}")
            
            # Return data if we got at least the price
            if company_data["current_price"] > 0:
                return company_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching real data for {symbol}: {str(e)}")
            return None
    
    def _get_shares_outstanding(self, symbol: str) -> float:
        """Get approximate shares outstanding for market cap calculation"""
        shares_map = {
            "TSLA": 3.18e9,  # 3.18 billion shares
            "RIVN": 1.0e9,   # ~1 billion shares
            "LCID": 2.3e9,   # ~2.3 billion shares
            "NIO": 1.8e9,    # ~1.8 billion shares
            "XPEV": 0.85e9,  # ~850 million shares
            "LI": 1.05e9,    # ~1.05 billion shares
            "F": 4.0e9,      # ~4 billion shares
            "GM": 1.2e9      # ~1.2 billion shares
        }
        return shares_map.get(symbol, 1.0e9)
    
    async def get_company_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get financial and operational data for a company
        Uses real-time APIs with fallback to mock data
        """
        try:
            # Try to fetch real data first
            logger.info(f"Attempting to fetch real data for {symbol}")
            real_data = await self._fetch_real_company_data(symbol)
            
            if real_data and real_data["current_price"] > 0:
                logger.info(f"Using real-time data for {symbol}")
                return real_data
            
            # Fallback to mock data
            logger.warning(f"Using mock data fallback for {symbol}")
            # In production, would use financial data APIs
            # For now, return mock data based on symbol
            
            if symbol == "TSLA":
                return {
                    "symbol": "TSLA",
                    "name": "Tesla, Inc.",
                    "current_price": 245.67,
                    "market_cap": 780000000000,  # $780B
                    "pe_ratio": 65.4,
                    "revenue_ttm": 96770000000,  # $96.77B
                    "profit_margin": 15.5,
                    "annual_deliveries": 1808581,
                    "delivery_growth_yoy": 38.0,
                    "market_share": 19.5,
                    "ev_models": 5,
                    "production_capacity": 2200000,
                    "average_asp": 50000,
                    "r_and_d_spending": 3969000000
                }
            elif symbol == "RIVN":
                return {
                    "symbol": "RIVN",
                    "name": "Rivian Automotive",
                    "current_price": 12.45,
                    "market_cap": 12000000000,  # $12B
                    "pe_ratio": None,  # Not profitable yet
                    "revenue_ttm": 4434000000,  # $4.43B
                    "profit_margin": -185.0,
                    "annual_deliveries": 57232,
                    "delivery_growth_yoy": 169.0,
                    "market_share": 0.6,
                    "ev_models": 2,
                    "production_capacity": 150000,
                    "average_asp": 75000,
                    "r_and_d_spending": 1500000000
                }
            elif symbol == "LCID":
                return {
                    "symbol": "LCID",
                    "name": "Lucid Motors",
                    "current_price": 3.25,
                    "market_cap": 7500000000,  # $7.5B
                    "pe_ratio": None,
                    "revenue_ttm": 608000000,  # $608M
                    "profit_margin": -320.0,
                    "annual_deliveries": 6001,
                    "delivery_growth_yoy": 15.0,
                    "market_share": 0.06,
                    "ev_models": 2,
                    "production_capacity": 90000,
                    "average_asp": 90000,
                    "r_and_d_spending": 1200000000
                }
            elif symbol == "NIO":
                return {
                    "symbol": "NIO",
                    "name": "NIO Inc.",
                    "current_price": 5.85,
                    "market_cap": 10500000000,  # $10.5B
                    "pe_ratio": None,
                    "revenue_ttm": 7107000000,  # $7.1B
                    "profit_margin": -8.2,
                    "annual_deliveries": 160038,
                    "delivery_growth_yoy": 30.7,
                    "market_share": 1.7,
                    "ev_models": 6,
                    "production_capacity": 300000,
                    "average_asp": 52000,
                    "r_and_d_spending": 900000000
                }
            else:
                return self._get_generic_company_data(symbol)
                
        except Exception as e:
            logger.error(f"Error getting company data for {symbol}: {str(e)}")
            return {}
    
    def perform_comparison(self, tesla_data: Dict, competitor_data: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        Perform metric-by-metric comparison
        """
        try:
            comparisons = []
            
            # Market Capitalization
            competitor_market_caps = {k: v.get('market_cap', 0) for k, v in competitor_data.items()}
            comparisons.append({
                "metric": "Market Capitalization",
                "tesla": tesla_data.get('market_cap', 0),
                "competitors": competitor_market_caps,
                "unit": "$",
                "format": "billions"
            })
            
            # Annual Deliveries
            competitor_deliveries = {k: v.get('annual_deliveries', 0) for k, v in competitor_data.items()}
            comparisons.append({
                "metric": "Annual Deliveries",
                "tesla": tesla_data.get('annual_deliveries', 0),
                "competitors": competitor_deliveries,
                "unit": "units",
                "format": "number"
            })
            
            # Market Share
            competitor_shares = {k: v.get('market_share', 0) for k, v in competitor_data.items()}
            comparisons.append({
                "metric": "Market Share",
                "tesla": tesla_data.get('market_share', 0),
                "competitors": competitor_shares,
                "unit": "%",
                "format": "percentage"
            })
            
            # Profit Margin
            competitor_margins = {k: v.get('profit_margin', 0) for k, v in competitor_data.items()}
            comparisons.append({
                "metric": "Profit Margin",
                "tesla": tesla_data.get('profit_margin', 0),
                "competitors": competitor_margins,
                "unit": "%",
                "format": "percentage"
            })
            
            # Delivery Growth YoY
            competitor_growth = {k: v.get('delivery_growth_yoy', 0) for k, v in competitor_data.items()}
            comparisons.append({
                "metric": "Delivery Growth YoY",
                "tesla": tesla_data.get('delivery_growth_yoy', 0),
                "competitors": competitor_growth,
                "unit": "%",
                "format": "percentage"
            })
            
            # Average Selling Price
            competitor_asp = {k: v.get('average_asp', 0) for k, v in competitor_data.items()}
            comparisons.append({
                "metric": "Average Selling Price",
                "tesla": tesla_data.get('average_asp', 0),
                "competitors": competitor_asp,
                "unit": "$",
                "format": "currency"
            })
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error performing comparison: {str(e)}")
            return []
    
    def calculate_market_position(self, tesla_data: Dict, competitor_data: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Calculate Tesla's market positioning relative to competitors
        """
        try:
            all_companies = {"TSLA": tesla_data, **competitor_data}
            
            # Calculate rankings
            rankings = {}
            
            # Market Cap Rank
            market_caps = {k: v.get('market_cap', 0) for k, v in all_companies.items()}
            sorted_market_caps = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
            tesla_market_cap_rank = next(i for i, (k, v) in enumerate(sorted_market_caps, 1) if k == "TSLA")
            rankings['market_cap'] = tesla_market_cap_rank
            
            # Deliveries Rank
            deliveries = {k: v.get('annual_deliveries', 0) for k, v in all_companies.items()}
            sorted_deliveries = sorted(deliveries.items(), key=lambda x: x[1], reverse=True)
            tesla_deliveries_rank = next(i for i, (k, v) in enumerate(sorted_deliveries, 1) if k == "TSLA")
            rankings['deliveries'] = tesla_deliveries_rank
            
            # Market Share
            market_share = tesla_data.get('market_share', 0)
            total_market_share = sum(v.get('market_share', 0) for v in all_companies.values())
            
            # Competitive Strengths
            strengths = []
            if tesla_market_cap_rank == 1:
                strengths.append("Highest market capitalization")
            if tesla_deliveries_rank == 1:
                strengths.append("Highest delivery volume")
            if tesla_data.get('profit_margin', 0) > 0:
                strengths.append("Profitable operations")
            if tesla_data.get('market_share', 0) > 15:
                strengths.append("Dominant market position")
            
            # Areas to Watch
            challenges = []
            for symbol, data in competitor_data.items():
                if data.get('delivery_growth_yoy', 0) > tesla_data.get('delivery_growth_yoy', 0):
                    challenges.append(f"{symbol} growing faster")
            
            return {
                "rankings": rankings,
                "market_share_captured": market_share,
                "total_tracked_market_share": total_market_share,
                "competitive_strengths": strengths,
                "areas_to_watch": challenges,
                "position_summary": self._generate_position_summary(rankings, market_share)
            }
            
        except Exception as e:
            logger.error(f"Error calculating market position: {str(e)}")
            return {}
    
    def _generate_position_summary(self, rankings: Dict, market_share: float) -> str:
        """Generate a summary of Tesla's market position"""
        if rankings.get('market_cap') == 1 and rankings.get('deliveries') == 1:
            return "Market Leader - Tesla maintains dominant position in both market cap and delivery volume"
        elif rankings.get('market_cap') == 1:
            return "Value Leader - Tesla leads in market capitalization"
        elif rankings.get('deliveries') == 1:
            return "Volume Leader - Tesla leads in delivery volume"
        else:
            return "Strong Competitor - Tesla maintains significant market presence"
    
    def generate_competitive_insights(self, tesla_data: Dict, competitor_data: Dict[str, Dict], 
                                     comparison: List[Dict]) -> List[str]:
        """
        Generate actionable insights from competitive analysis
        """
        insights = []
        
        try:
            # Market Cap Insight
            tesla_market_cap = tesla_data.get('market_cap', 0)
            total_competitor_market_cap = sum(v.get('market_cap', 0) for v in competitor_data.values())
            if tesla_market_cap > total_competitor_market_cap:
                insights.append(f"Tesla's market cap (${tesla_market_cap/1e9:.1f}B) exceeds all tracked competitors combined")
            
            # Profitability Insight
            if tesla_data.get('profit_margin', 0) > 0:
                profitable_competitors = sum(1 for v in competitor_data.values() if v.get('profit_margin', 0) > 0)
                insights.append(f"Tesla is profitable with {tesla_data['profit_margin']:.1f}% margin, while {len(competitor_data) - profitable_competitors} competitors operate at a loss")
            
            # Growth Insight
            tesla_growth = tesla_data.get('delivery_growth_yoy', 0)
            avg_competitor_growth = sum(v.get('delivery_growth_yoy', 0) for v in competitor_data.values()) / len(competitor_data)
            if tesla_growth > avg_competitor_growth:
                insights.append(f"Tesla's delivery growth ({tesla_growth:.1f}%) outpaces competitor average ({avg_competitor_growth:.1f}%)")
            else:
                insights.append(f"Competitors showing faster growth ({avg_competitor_growth:.1f}%) vs Tesla ({tesla_growth:.1f}%)")
            
            # Market Share Insight
            tesla_share = tesla_data.get('market_share', 0)
            insights.append(f"Tesla commands {tesla_share:.1f}% of the tracked EV market")
            
            return insights[:5]  # Return top 5 insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return ["Unable to generate competitive insights at this time"]
    
    def _get_generic_company_data(self, symbol: str) -> Dict[str, Any]:
        """Get generic company data for unrecognized symbols"""
        return {
            "symbol": symbol,
            "name": self.competitors.get(symbol, {}).get("name", "Unknown Company"),
            "current_price": 0.0,
            "market_cap": 0,
            "pe_ratio": None,
            "revenue_ttm": 0,
            "profit_margin": 0.0,
            "annual_deliveries": 0,
            "delivery_growth_yoy": 0.0,
            "market_share": 0.0,
            "ev_models": 0,
            "production_capacity": 0,
            "average_asp": 0,
            "r_and_d_spending": 0
        }
    
    def _get_mock_competitor_data(self) -> Dict[str, Any]:
        """Generate mock competitor data for development"""
        return {
            "tesla_data": {
                "symbol": "TSLA",
                "name": "Tesla, Inc.",
                "market_cap": 780000000000,
                "annual_deliveries": 1808581,
                "market_share": 19.5
            },
            "competitor_data": {
                "RIVN": {
                    "symbol": "RIVN",
                    "name": "Rivian",
                    "market_cap": 12000000000,
                    "annual_deliveries": 57232,
                    "market_share": 0.6
                },
                "LCID": {
                    "symbol": "LCID",
                    "name": "Lucid",
                    "market_cap": 7500000000,
                    "annual_deliveries": 6001,
                    "market_share": 0.06
                }
            },
            "insights": [
                "Tesla leads in market capitalization and delivery volume",
                "Competitors showing rapid growth from smaller base",
                "Tesla maintains profitability while most competitors are unprofitable"
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    async def analyze_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a competitor-related query and provide relevant information
        """
        try:
            query_lower = query.lower()
            
            # Determine which competitors to analyze
            competitors = []
            if "rivian" in query_lower or "rivn" in query_lower:
                competitors.append("RIVN")
            if "lucid" in query_lower or "lcid" in query_lower:
                competitors.append("LCID")
            if "nio" in query_lower:
                competitors.append("NIO")
            
            # Default to main competitors if none specified
            if not competitors:
                competitors = ["RIVN", "LCID", "NIO"]
            
            # Get comprehensive analysis
            analysis = await self.get_comprehensive_analysis("TSLA", competitors)
            
            return {
                "type": "competitor_analysis",
                "analysis": analysis,
                "summary": "Competitive analysis of Tesla vs major EV competitors",
                "insights": analysis.get('insights', [])
            }
                
        except Exception as e:
            logger.error(f"Error analyzing competitor query: {str(e)}")
            return {
                "type": "error",
                "error": f"Failed to analyze competitor query: {str(e)}",
                "summary": "Unable to provide competitor analysis"
            }
