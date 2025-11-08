"""
Industry Trends Agent for Tesla Market Intelligence
Provides real-time EV industry trends and analysis
"""
import os
import logging
import aiohttp
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class IndustryTrendsAgent:
    """Agent for fetching and analyzing EV industry trends"""
    
    def __init__(self):
        """Initialize the Industry Trends Agent with API keys"""
        # Clean API keys (remove quotes, whitespace, comments)
        self.fred_api_key = os.getenv('FRED_API_KEY', '').strip().strip('"').strip("'").split('#')[0].strip()
        self.tavily_api_key = os.getenv('TAVILYAPI_KEY', '').strip().strip('"').strip("'").split('#')[0].strip()
        self.serper_api_key = os.getenv('SERPER_API_KEY', '').strip().strip('"').strip("'").split('#')[0].strip()
        self.world_news_api_key = os.getenv('WORLD_NEWS_API_KEY', '').strip().strip('"').strip("'").split('#')[0].strip()
        self.search_api_key = os.getenv('SEARCH_API_KEY', '').strip().strip('"').strip("'").split('#')[0].strip()
        
        logger.info(f"IndustryTrendsAgent initialized:")
        logger.info(f"  - FRED API key: {bool(self.fred_api_key)}")
        logger.info(f"  - Tavily API key: {bool(self.tavily_api_key)}")
        logger.info(f"  - Serper API key: {bool(self.serper_api_key)}")
        logger.info(f"  - World News API key: {bool(self.world_news_api_key)}")
        logger.info(f"  - Search API key: {bool(self.search_api_key)}")

    async def get_industry_trends(self, symbol: str = "TSLA") -> Dict[str, Any]:
        """
        Fetch comprehensive industry trends data
        
        Returns:
            Dict containing industry trends with data sources
        """
        logger.info(f"Fetching industry trends for {symbol}")
        
        try:
            # Fetch data from multiple sources in parallel
            results = await asyncio.gather(
                self._get_ev_market_size(),
                self._get_ev_adoption_rate(),
                self._get_battery_cost_trends(),
                self._get_charging_infrastructure(),
                self._get_autonomous_driving_trends(),
                self._get_market_competition(),
                return_exceptions=True
            )
            
            # Process results
            ev_market_size = results[0] if not isinstance(results[0], Exception) else None
            ev_adoption = results[1] if not isinstance(results[1], Exception) else None
            battery_cost = results[2] if not isinstance(results[2], Exception) else None
            charging_infra = results[3] if not isinstance(results[3], Exception) else None
            autonomous = results[4] if not isinstance(results[4], Exception) else None
            competition = results[5] if not isinstance(results[5], Exception) else None
            
            # Count successful API calls
            api_success_count = sum([
                bool(ev_market_size and ev_market_size.get('source') != 'fallback'),
                bool(ev_adoption and ev_adoption.get('source') != 'fallback'),
                bool(battery_cost and battery_cost.get('source') != 'fallback'),
                bool(charging_infra and charging_infra.get('source') != 'fallback'),
                bool(autonomous and autonomous.get('source') != 'fallback'),
                bool(competition and competition.get('source') != 'fallback')
            ])
            
            is_fallback = api_success_count == 0
            
            # Build data sources list
            data_sources = []
            for result in [ev_market_size, ev_adoption, battery_cost, charging_infra, autonomous, competition]:
                if result and result.get('api_source'):
                    data_sources.append(result['api_source'])
            
            # Deduplicate sources
            unique_sources = list(set(data_sources))
            
            return {
                'ev_market_size': ev_market_size.get('value', '1.2 trillion by 2030') if ev_market_size else '1.2 trillion by 2030',
                'ev_adoption_rate': ev_adoption.get('value', '15% of new car sales in 2023') if ev_adoption else '15% of new car sales in 2023',
                'battery_cost_reduction': battery_cost.get('value', '89% decrease since 2010') if battery_cost else '89% decrease since 2010',
                'charging_infrastructure': charging_infra.get('value', '2.7 million charging stations globally') if charging_infra else '2.7 million charging stations globally',
                'autonomous_driving': autonomous.get('value', 'Level 4 autonomy expected by 2025') if autonomous else 'Level 4 autonomy expected by 2025',
                'market_competition': competition.get('value', '200+ EV models expected by 2025') if competition else '200+ EV models expected by 2025',
                'is_fallback': is_fallback,
                'data_source': ', '.join(unique_sources) if unique_sources else 'Fallback Mode (Hardcoded)',
                'api_count': api_success_count,
                'total_metrics': 6,
                'detailed_sources': {
                    'ev_market_size': ev_market_size.get('api_source', 'Fallback') if ev_market_size else 'Fallback',
                    'ev_adoption_rate': ev_adoption.get('api_source', 'Fallback') if ev_adoption else 'Fallback',
                    'battery_cost_reduction': battery_cost.get('api_source', 'Fallback') if battery_cost else 'Fallback',
                    'charging_infrastructure': charging_infra.get('api_source', 'Fallback') if charging_infra else 'Fallback',
                    'autonomous_driving': autonomous.get('api_source', 'Fallback') if autonomous else 'Fallback',
                    'market_competition': competition.get('api_source', 'Fallback') if competition else 'Fallback'
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching industry trends: {str(e)}")
            return self._get_fallback_data()

    async def _get_ev_market_size(self) -> Dict[str, Any]:
        """Get EV market size projection using Tavily AI"""
        if not self.tavily_api_key:
            logger.warning("No Tavily API key, using fallback for EV market size")
            return {'value': '1.2 trillion by 2030', 'source': 'fallback', 'api_source': 'Fallback'}
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.tavily.com/search"
                payload = {
                    "api_key": self.tavily_api_key,
                    "query": "global electric vehicle market size projection 2025 2030 USD trillion",
                    "search_depth": "basic",
                    "max_results": 3
                }
                
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])
                        
                        if results:
                            # Extract market size from first result
                            answer = data.get('answer', '')
                            if 'trillion' in answer.lower() or 'billion' in answer.lower():
                                logger.info(f"Tavily: Got EV market size data")
                                return {
                                    'value': answer[:100] if answer else '1.5 trillion by 2030',
                                    'source': 'api',
                                    'api_source': 'Tavily AI Research'
                                }
                        
                        logger.warning("Tavily: No market size data in response")
                        return {'value': '1.2 trillion by 2030', 'source': 'fallback', 'api_source': 'Fallback'}
                    else:
                        logger.error(f"Tavily API error: {response.status}")
                        return {'value': '1.2 trillion by 2030', 'source': 'fallback', 'api_source': 'Fallback'}
                        
        except Exception as e:
            logger.error(f"Error fetching EV market size: {str(e)}")
            return {'value': '1.2 trillion by 2030', 'source': 'fallback', 'api_source': 'Fallback'}

    async def _get_ev_adoption_rate(self) -> Dict[str, Any]:
        """Get EV adoption rate using Tavily AI"""
        if not self.tavily_api_key:
            logger.warning("No Tavily API key, using fallback for EV adoption rate")
            return {'value': '15% of new car sales in 2023', 'source': 'fallback', 'api_source': 'Fallback'}
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.tavily.com/search"
                payload = {
                    "api_key": self.tavily_api_key,
                    "query": "electric vehicle adoption rate 2024 2025 percentage new car sales worldwide",
                    "search_depth": "basic",
                    "max_results": 3
                }
                
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        answer = data.get('answer', '')
                        
                        if answer and '%' in answer:
                            logger.info(f"Tavily: Got EV adoption rate")
                            return {
                                'value': answer[:100] if answer else '17% of new car sales in 2024',
                                'source': 'api',
                                'api_source': 'Tavily AI Research'
                            }
                        
                        logger.warning("Tavily: No adoption rate in response")
                        return {'value': '15% of new car sales in 2023', 'source': 'fallback', 'api_source': 'Fallback'}
                    else:
                        logger.error(f"Tavily API error: {response.status}")
                        return {'value': '15% of new car sales in 2023', 'source': 'fallback', 'api_source': 'Fallback'}
                        
        except Exception as e:
            logger.error(f"Error fetching EV adoption rate: {str(e)}")
            return {'value': '15% of new car sales in 2023', 'source': 'fallback', 'api_source': 'Fallback'}

    async def _get_battery_cost_trends(self) -> Dict[str, Any]:
        """Get battery cost reduction trends using Tavily AI"""
        if not self.tavily_api_key:
            logger.warning("No Tavily API key, using fallback for battery costs")
            return {'value': '89% decrease since 2010', 'source': 'fallback', 'api_source': 'Fallback'}
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.tavily.com/search"
                payload = {
                    "api_key": self.tavily_api_key,
                    "query": "lithium ion battery pack cost decrease 2010 2024 percentage decline",
                    "search_depth": "basic",
                    "max_results": 3
                }
                
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        answer = data.get('answer', '')
                        
                        if answer and ('%' in answer or 'decrease' in answer.lower() or 'decline' in answer.lower()):
                            logger.info(f"Tavily: Got battery cost trends")
                            return {
                                'value': answer[:100] if answer else '90% decrease since 2010',
                                'source': 'api',
                                'api_source': 'Tavily AI Research'
                            }
                        
                        logger.warning("Tavily: No battery cost data in response")
                        return {'value': '89% decrease since 2010', 'source': 'fallback', 'api_source': 'Fallback'}
                    else:
                        logger.error(f"Tavily API error: {response.status}")
                        return {'value': '89% decrease since 2010', 'source': 'fallback', 'api_source': 'Fallback'}
                        
        except Exception as e:
            logger.error(f"Error fetching battery costs: {str(e)}")
            return {'value': '89% decrease since 2010', 'source': 'fallback', 'api_source': 'Fallback'}

    async def _get_charging_infrastructure(self) -> Dict[str, Any]:
        """Get charging infrastructure data using Serper API"""
        if not self.serper_api_key:
            logger.warning("No Serper API key, using fallback for charging infrastructure")
            return {'value': '2.7 million charging stations globally', 'source': 'fallback', 'api_source': 'Fallback'}
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://google.serper.dev/search"
                headers = {
                    'X-API-KEY': self.serper_api_key,
                    'Content-Type': 'application/json'
                }
                payload = {
                    "q": "global electric vehicle charging stations 2024 total number worldwide",
                    "num": 5
                }
                
                async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract from knowledge graph or answer box
                        answer_box = data.get('answerBox', {})
                        knowledge_graph = data.get('knowledgeGraph', {})
                        organic = data.get('organic', [])
                        
                        # Try to extract number from answer box
                        if answer_box:
                            snippet = answer_box.get('snippet', '') or answer_box.get('answer', '')
                            if 'million' in snippet.lower() or 'charging' in snippet.lower():
                                logger.info(f"Serper: Got charging infrastructure data")
                                return {
                                    'value': snippet[:100] if snippet else '3.2 million charging stations globally',
                                    'source': 'api',
                                    'api_source': 'Serper Search API'
                                }
                        
                        # Try organic results
                        for result in organic[:2]:
                            snippet = result.get('snippet', '')
                            if 'million' in snippet.lower() and 'charging' in snippet.lower():
                                logger.info(f"Serper: Got charging infrastructure from organic results")
                                return {
                                    'value': snippet[:100] if snippet else '3.2 million charging stations globally',
                                    'source': 'api',
                                    'api_source': 'Serper Search API'
                                }
                        
                        logger.warning("Serper: No charging infrastructure data found")
                        return {'value': '2.7 million charging stations globally', 'source': 'fallback', 'api_source': 'Fallback'}
                    else:
                        logger.error(f"Serper API error: {response.status}")
                        return {'value': '2.7 million charging stations globally', 'source': 'fallback', 'api_source': 'Fallback'}
                        
        except Exception as e:
            logger.error(f"Error fetching charging infrastructure: {str(e)}")
            return {'value': '2.7 million charging stations globally', 'source': 'fallback', 'api_source': 'Fallback'}

    async def _get_autonomous_driving_trends(self) -> Dict[str, Any]:
        """Get autonomous driving trends using Tavily AI"""
        if not self.tavily_api_key:
            logger.warning("No Tavily API key, using fallback for autonomous driving")
            return {'value': 'Level 4 autonomy expected by 2025', 'source': 'fallback', 'api_source': 'Fallback'}
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.tavily.com/search"
                payload = {
                    "api_key": self.tavily_api_key,
                    "query": "autonomous vehicle Level 4 Level 5 timeline 2025 2030 commercialization",
                    "search_depth": "basic",
                    "max_results": 3
                }
                
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        answer = data.get('answer', '')
                        
                        if answer and ('level' in answer.lower() or 'autonomous' in answer.lower()):
                            logger.info(f"Tavily: Got autonomous driving trends")
                            return {
                                'value': answer[:100] if answer else 'Level 4 autonomy expected by 2026',
                                'source': 'api',
                                'api_source': 'Tavily AI Research'
                            }
                        
                        logger.warning("Tavily: No autonomous driving data in response")
                        return {'value': 'Level 4 autonomy expected by 2025', 'source': 'fallback', 'api_source': 'Fallback'}
                    else:
                        logger.error(f"Tavily API error: {response.status}")
                        return {'value': 'Level 4 autonomy expected by 2025', 'source': 'fallback', 'api_source': 'Fallback'}
                        
        except Exception as e:
            logger.error(f"Error fetching autonomous driving trends: {str(e)}")
            return {'value': 'Level 4 autonomy expected by 2025', 'source': 'fallback', 'api_source': 'Fallback'}

    async def _get_market_competition(self) -> Dict[str, Any]:
        """Get market competition data using Tavily AI"""
        if not self.tavily_api_key:
            logger.warning("No Tavily API key, using fallback for market competition")
            return {'value': '200+ EV models expected by 2025', 'source': 'fallback', 'api_source': 'Fallback'}
        
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.tavily.com/search"
                payload = {
                    "api_key": self.tavily_api_key,
                    "query": "number of electric vehicle EV models 2024 2025 market total available",
                    "search_depth": "basic",
                    "max_results": 3
                }
                
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as response:
                    if response.status == 200:
                        data = await response.json()
                        answer = data.get('answer', '')
                        
                        if answer and ('model' in answer.lower() or 'ev' in answer.lower()):
                            logger.info(f"Tavily: Got market competition data")
                            return {
                                'value': answer[:100] if answer else '250+ EV models expected by 2025',
                                'source': 'api',
                                'api_source': 'Tavily AI Research'
                            }
                        
                        logger.warning("Tavily: No competition data in response")
                        return {'value': '200+ EV models expected by 2025', 'source': 'fallback', 'api_source': 'Fallback'}
                    else:
                        logger.error(f"Tavily API error: {response.status}")
                        return {'value': '200+ EV models expected by 2025', 'source': 'fallback', 'api_source': 'Fallback'}
                        
        except Exception as e:
            logger.error(f"Error fetching market competition: {str(e)}")
            return {'value': '200+ EV models expected by 2025', 'source': 'fallback', 'api_source': 'Fallback'}

    def _get_fallback_data(self) -> Dict[str, Any]:
        """Return fallback/mock data when APIs are unavailable"""
        return {
            'ev_market_size': '1.2 trillion by 2030',
            'ev_adoption_rate': '15% of new car sales in 2023',
            'battery_cost_reduction': '89% decrease since 2010',
            'charging_infrastructure': '2.7 million charging stations globally',
            'autonomous_driving': 'Level 4 autonomy expected by 2025',
            'market_competition': '200+ EV models expected by 2025',
            'is_fallback': True,
            'data_source': 'Fallback Mode (Hardcoded)',
            'api_count': 0,
            'total_metrics': 6,
            'detailed_sources': {
                'ev_market_size': 'Fallback',
                'ev_adoption_rate': 'Fallback',
                'battery_cost_reduction': 'Fallback',
                'charging_infrastructure': 'Fallback',
                'autonomous_driving': 'Fallback',
                'market_competition': 'Fallback'
            }
        }

