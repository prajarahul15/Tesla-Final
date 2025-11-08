"""
Market Sentiment Agent - Analyzes market sentiment from news, social media, and analyst ratings
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    overall_sentiment: str  # positive, negative, neutral
    sentiment_score: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    news_sentiment: float
    social_sentiment: float
    analyst_sentiment: float
    trending_topics: List[str]
    timestamp: datetime

@dataclass
class NewsArticle:
    title: str
    source: str
    published_date: datetime
    sentiment: str
    sentiment_score: float
    summary: str
    url: str

@dataclass
class AnalystRating:
    firm: str
    rating: str  # buy, hold, sell, strong_buy, strong_sell
    price_target: float
    date: datetime
    analyst_name: str

class MarketSentimentAgent:
    """
    Market Sentiment Agent that analyzes sentiment from multiple sources:
    - News articles (News API)
    - Social media (Twitter sentiment)
    - Analyst ratings
    """
    
    def __init__(self):
        # Clean API keys
        news_key = os.getenv('NEWS_API_KEY')
        if news_key:
            news_key = news_key.strip().strip('"').strip("'")
        self.news_api_key = news_key
        
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache for sentiment data
        
        logger.info(f"MarketSentimentAgent initialized - News API key: {bool(self.news_api_key and self.news_api_key != 'your_news_api_key_here')}")
        
        # Sentiment keywords for text analysis
        self.positive_keywords = [
            'bullish', 'growth', 'strong', 'outperform', 'beat', 'exceed', 
            'positive', 'gain', 'surge', 'rally', 'upgrade', 'buy',
            'innovative', 'breakthrough', 'leader', 'success', 'profit',
            'record', 'high', 'increase', 'soar', 'boom'
        ]
        
        self.negative_keywords = [
            'bearish', 'decline', 'weak', 'underperform', 'miss', 'below',
            'negative', 'loss', 'fall', 'crash', 'downgrade', 'sell',
            'concern', 'risk', 'challenge', 'problem', 'lawsuit', 'recall',
            'drop', 'cut', 'lower', 'warn', 'crisis'
        ]
    
    async def get_comprehensive_sentiment(self, symbol: str = "TSLA") -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis from all sources
        """
        try:
            # Check cache first
            cache_key = f"sentiment_{symbol}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                    return cached_data
            
            # Get sentiment from different sources
            news_sentiment = await self.get_news_sentiment(symbol)
            social_sentiment = await self.get_social_sentiment(symbol)
            analyst_sentiment = await self.get_analyst_sentiment(symbol)
            
            # Calculate overall sentiment
            overall_score = (
                news_sentiment['score'] * 0.4 +
                social_sentiment['score'] * 0.3 +
                analyst_sentiment['score'] * 0.3
            )
            
            # Determine overall sentiment label
            if overall_score > 0.2:
                overall_label = "positive"
            elif overall_score < -0.2:
                overall_label = "negative"
            else:
                overall_label = "neutral"
            
            # Calculate confidence
            confidence = min(
                news_sentiment['confidence'],
                social_sentiment['confidence'],
                analyst_sentiment['confidence']
            )
            
            sentiment_data = SentimentData(
                overall_sentiment=overall_label,
                sentiment_score=overall_score,
                confidence=confidence,
                news_sentiment=news_sentiment['score'],
                social_sentiment=social_sentiment['score'],
                analyst_sentiment=analyst_sentiment['score'],
                trending_topics=news_sentiment['trending_topics'],
                timestamp=datetime.now()
            )
            
            result = {
                "sentiment_data": sentiment_data.__dict__,
                "news_analysis": news_sentiment,
                "social_analysis": social_sentiment,
                "analyst_analysis": analyst_sentiment,
                "recent_news": news_sentiment.get('articles', []),
                "analyst_ratings": analyst_sentiment.get('ratings', []),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            self.cache[cache_key] = (result, datetime.now())
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting comprehensive sentiment for {symbol}: {str(e)}")
            return self._get_mock_sentiment_data(symbol)
    
    async def get_news_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze sentiment from news articles
        Fetches real news from NewsAPI.org
        """
        try:
            # Try to fetch real news from API
            articles = []
            if self.news_api_key and self.news_api_key != 'your_news_api_key_here':
                articles = await self._fetch_real_news(symbol)
            
            # If no articles from API, use mock data as fallback
            if not articles:
                logger.warning(f"No news from API, using mock data for {symbol}")
                articles = self._get_mock_news_articles(symbol)
            
            # Analyze sentiment from articles
            total_score = 0
            article_count = len(articles)
            trending_topics = []
            
            for article in articles:
                total_score += article['sentiment_score']
                trending_topics.extend(article.get('topics', []))
            
            avg_score = total_score / article_count if article_count > 0 else 0
            
            # Count trending topics
            topic_counts = {}
            for topic in trending_topics:
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
            
            top_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "score": avg_score,
                "confidence": 0.75,
                "article_count": article_count,
                "positive_count": sum(1 for a in articles if a['sentiment_score'] > 0.1),
                "negative_count": sum(1 for a in articles if a['sentiment_score'] < -0.1),
                "neutral_count": sum(1 for a in articles if -0.1 <= a['sentiment_score'] <= 0.1),
                "trending_topics": [topic for topic, _ in top_topics],
                "articles": articles[:10],  # Return top 10 articles
                "recent_news": articles[:5]  # Return top 5 for display
            }
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "score": 0.0,
                "confidence": 0.0,
                "article_count": 0,
                "trending_topics": [],
                "articles": [],
                "recent_news": []
            }
    
    async def _fetch_real_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch real news from NewsAPI.org"""
        try:
            import aiohttp
            
            # Map symbol to company name for better search results
            company_name = "Tesla" if symbol == "TSLA" else symbol
            
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': company_name,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 10,
                'apiKey': self.news_api_key
            }
            
            logger.info(f"Fetching news from NewsAPI for {company_name}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('status') == 'ok':
                            articles_data = data.get('articles', [])
                            logger.info(f"Received {len(articles_data)} articles from NewsAPI")
                            
                            # Convert to our format and analyze sentiment
                            processed_articles = []
                            for article in articles_data:
                                title = article.get('title', '')
                                description = article.get('description', '')
                                content = f"{title} {description}".lower()
                                
                                # Simple sentiment analysis
                                sentiment_score = self._analyze_text_sentiment(content)
                                
                                if sentiment_score > 0.2:
                                    sentiment = "positive"
                                elif sentiment_score < -0.2:
                                    sentiment = "negative"
                                else:
                                    sentiment = "neutral"
                                
                                # Extract topics
                                topics = self._extract_topics(content)
                                
                                processed_articles.append({
                                    "title": title,
                                    "source": article.get('source', {}).get('name', 'Unknown'),
                                    "published_date": article.get('publishedAt', datetime.now().isoformat()),
                                    "sentiment": sentiment,
                                    "sentiment_score": sentiment_score,
                                    "summary": description or title,
                                    "url": article.get('url', ''),
                                    "topics": topics
                                })
                            
                            return processed_articles
                        else:
                            logger.warning(f"NewsAPI error: {data.get('message', 'Unknown')}")
                    elif response.status == 429:
                        logger.warning("NewsAPI rate limit exceeded")
                    elif response.status == 401:
                        logger.error("NewsAPI authentication failed - check API key")
                    else:
                        logger.warning(f"NewsAPI returned status {response.status}")
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching real news: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Simple sentiment analysis based on keywords"""
        text = text.lower()
        
        positive_count = sum(1 for word in self.positive_keywords if word in text)
        negative_count = sum(1 for word in self.negative_keywords if word in text)
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        # Return score between -1.0 and 1.0
        score = (positive_count - negative_count) / max(total, 1)
        return round(score, 2)
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract relevant topics from text"""
        topics = []
        topic_keywords = {
            'deliveries': ['deliver', 'delivery', 'production', 'output'],
            'earnings': ['earnings', 'profit', 'revenue', 'income', 'quarterly'],
            'competition': ['competition', 'competitor', 'rival', 'market share'],
            'innovation': ['innovation', 'technology', 'breakthrough', 'new model'],
            'regulation': ['regulation', 'regulatory', 'government', 'policy'],
            'stock': ['stock', 'share', 'price', 'trading', 'market'],
            'expansion': ['expansion', 'factory', 'gigafactory', 'growth'],
            'lawsuit': ['lawsuit', 'legal', 'court', 'investigation']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ['general']
    
    async def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze sentiment from social media
        """
        try:
            # In production, would use Twitter API, Reddit API, etc.
            # For now, return mock data based on general market sentiment
            
            # Mock social sentiment (slightly more positive bias)
            base_score = 0.15
            
            return {
                "score": base_score,
                "confidence": 0.65,
                "mention_count": 15420,
                "positive_mentions": 8734,
                "negative_mentions": 4123,
                "neutral_mentions": 2563,
                "trending_hashtags": ["#TSLA", "#Tesla", "#ElectricVehicles", "#Musk", "#EV"],
                "sentiment_trend": "increasing"  # increasing, decreasing, stable
            }
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {str(e)}")
            return {
                "score": 0.0,
                "confidence": 0.0,
                "mention_count": 0
            }
    
    async def get_analyst_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze analyst ratings and price targets
        """
        try:
            # In production, would use financial data APIs
            # For now, return mock analyst data
            
            ratings = self._get_mock_analyst_ratings(symbol)
            
            # Calculate sentiment score from ratings
            rating_scores = {
                "strong_buy": 1.0,
                "buy": 0.5,
                "hold": 0.0,
                "sell": -0.5,
                "strong_sell": -1.0
            }
            
            total_score = sum(rating_scores.get(r['rating'], 0) for r in ratings)
            avg_score = total_score / len(ratings) if ratings else 0
            
            # Count ratings by type
            rating_counts = {}
            for rating in ratings:
                r = rating['rating']
                rating_counts[r] = rating_counts.get(r, 0) + 1
            
            # Calculate average price target
            avg_price_target = sum(r['price_target'] for r in ratings) / len(ratings) if ratings else 0
            
            return {
                "score": avg_score,
                "confidence": 0.85,
                "total_analysts": len(ratings),
                "rating_distribution": rating_counts,
                "average_price_target": avg_price_target,
                "ratings": ratings[:10]  # Return top 10 ratings
            }
            
        except Exception as e:
            logger.error(f"Error analyzing analyst sentiment: {str(e)}")
            return {
                "score": 0.0,
                "confidence": 0.0,
                "total_analysts": 0
            }
    
    def _get_mock_news_articles(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate mock news articles for development"""
        articles = [
            {
                "title": "Tesla Reports Strong Q4 Delivery Numbers, Beating Estimates",
                "source": "Reuters",
                "published_date": (datetime.now() - timedelta(days=1)).isoformat(),
                "sentiment": "positive",
                "sentiment_score": 0.75,
                "summary": "Tesla delivered more vehicles than analysts expected in Q4, showing continued growth momentum.",
                "url": "https://reuters.com/tesla-q4",
                "topics": ["deliveries", "growth", "performance"]
            },
            {
                "title": "Tesla Faces Increased Competition in EV Market",
                "source": "Bloomberg",
                "published_date": (datetime.now() - timedelta(days=2)).isoformat(),
                "sentiment": "neutral",
                "sentiment_score": -0.1,
                "summary": "Traditional automakers are ramping up EV production, intensifying competition.",
                "url": "https://bloomberg.com/tesla-competition",
                "topics": ["competition", "market_share", "ev_market"]
            },
            {
                "title": "Tesla's FSD Technology Shows Promising Results in Latest Update",
                "source": "TechCrunch",
                "published_date": (datetime.now() - timedelta(days=3)).isoformat(),
                "sentiment": "positive",
                "sentiment_score": 0.65,
                "summary": "Full Self-Driving beta receives positive feedback from users after latest software update.",
                "url": "https://techcrunch.com/tesla-fsd",
                "topics": ["technology", "autonomous_driving", "innovation"]
            },
            {
                "title": "Tesla Stock Rises on Strong Production Outlook",
                "source": "CNBC",
                "published_date": (datetime.now() - timedelta(days=4)).isoformat(),
                "sentiment": "positive",
                "sentiment_score": 0.55,
                "summary": "Investors respond positively to Tesla's production guidance for next quarter.",
                "url": "https://cnbc.com/tesla-stock",
                "topics": ["stock_price", "production", "investor_sentiment"]
            },
            {
                "title": "Analysts Debate Tesla's Valuation Amid Market Volatility",
                "source": "Wall Street Journal",
                "published_date": (datetime.now() - timedelta(days=5)).isoformat(),
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "summary": "Wall Street remains divided on whether Tesla's current valuation is justified.",
                "url": "https://wsj.com/tesla-valuation",
                "topics": ["valuation", "market_volatility", "analysis"]
            }
        ]
        return articles
    
    def _get_mock_analyst_ratings(self, symbol: str) -> List[Dict[str, Any]]:
        """Generate mock analyst ratings for development"""
        ratings = [
            {
                "firm": "Morgan Stanley",
                "rating": "buy",
                "price_target": 310.0,
                "date": (datetime.now() - timedelta(days=7)).isoformat(),
                "analyst_name": "Adam Jonas"
            },
            {
                "firm": "Goldman Sachs",
                "rating": "buy",
                "price_target": 295.0,
                "date": (datetime.now() - timedelta(days=14)).isoformat(),
                "analyst_name": "Mark Delaney"
            },
            {
                "firm": "JPMorgan",
                "rating": "hold",
                "price_target": 250.0,
                "date": (datetime.now() - timedelta(days=21)).isoformat(),
                "analyst_name": "Ryan Brinkman"
            },
            {
                "firm": "Wedbush",
                "rating": "strong_buy",
                "price_target": 350.0,
                "date": (datetime.now() - timedelta(days=10)).isoformat(),
                "analyst_name": "Dan Ives"
            },
            {
                "firm": "Bank of America",
                "rating": "buy",
                "price_target": 280.0,
                "date": (datetime.now() - timedelta(days=15)).isoformat(),
                "analyst_name": "John Murphy"
            },
            {
                "firm": "Credit Suisse",
                "rating": "hold",
                "price_target": 260.0,
                "date": (datetime.now() - timedelta(days=20)).isoformat(),
                "analyst_name": "Dan Levy"
            }
        ]
        return ratings
    
    def _get_mock_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Generate mock sentiment data for development"""
        return {
            "sentiment_data": {
                "overall_sentiment": "positive",
                "sentiment_score": 0.35,
                "confidence": 0.75,
                "news_sentiment": 0.40,
                "social_sentiment": 0.15,
                "analyst_sentiment": 0.50,
                "trending_topics": ["deliveries", "growth", "competition", "technology"],
                "timestamp": datetime.now().isoformat()
            },
            "news_analysis": {
                "score": 0.40,
                "confidence": 0.75,
                "article_count": 5,
                "positive_count": 3,
                "negative_count": 0,
                "neutral_count": 2
            },
            "social_analysis": {
                "score": 0.15,
                "confidence": 0.65,
                "mention_count": 15420
            },
            "analyst_analysis": {
                "score": 0.50,
                "confidence": 0.85,
                "total_analysts": 6,
                "average_price_target": 290.0
            },
            "timestamp": datetime.now().isoformat()
        }
    
    async def analyze_query(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a sentiment-related query and provide relevant information
        """
        try:
            query_lower = query.lower()
            symbol = "TSLA"  # Default to Tesla
            
            # Get comprehensive sentiment
            sentiment = await self.get_comprehensive_sentiment(symbol)
            
            # Customize response based on query type
            if "news" in query_lower:
                return {
                    "type": "news_sentiment",
                    "symbol": symbol,
                    "sentiment": sentiment,
                    "summary": f"News sentiment analysis for {symbol}",
                    "insights": [
                        f"Overall news sentiment: {sentiment['news_analysis']['score']:.2f}",
                        f"Positive articles: {sentiment['news_analysis']['positive_count']}",
                        f"Negative articles: {sentiment['news_analysis']['negative_count']}"
                    ]
                }
            elif "analyst" in query_lower or "rating" in query_lower:
                return {
                    "type": "analyst_sentiment",
                    "symbol": symbol,
                    "sentiment": sentiment,
                    "summary": f"Analyst sentiment for {symbol}",
                    "insights": [
                        f"Analyst sentiment score: {sentiment['analyst_analysis']['score']:.2f}",
                        f"Total analysts covering: {sentiment['analyst_analysis']['total_analysts']}",
                        f"Average price target: ${sentiment['analyst_analysis']['average_price_target']:.2f}"
                    ]
                }
            else:
                return {
                    "type": "overall_sentiment",
                    "symbol": symbol,
                    "sentiment": sentiment,
                    "summary": f"Market sentiment analysis for {symbol}",
                    "insights": [
                        f"Overall sentiment: {sentiment['sentiment_data']['overall_sentiment']} ({sentiment['sentiment_data']['sentiment_score']:.2f})",
                        f"News sentiment: {sentiment['sentiment_data']['news_sentiment']:.2f}",
                        f"Social sentiment: {sentiment['sentiment_data']['social_sentiment']:.2f}",
                        f"Analyst sentiment: {sentiment['sentiment_data']['analyst_sentiment']:.2f}"
                    ]
                }
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment query: {str(e)}")
            return {
                "type": "error",
                "error": f"Failed to analyze sentiment query: {str(e)}",
                "summary": "Unable to provide sentiment analysis"
            }
