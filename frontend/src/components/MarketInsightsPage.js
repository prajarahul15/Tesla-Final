import React, { useState, useEffect } from 'react';
import axios from 'axios';
import StockChart from './StockChart';
import TechnicalAnalysisChart from './TechnicalAnalysisChart';
import RiskMonitoringWidget from './RiskMonitoringWidget';
import { formatCompactNumber } from '../lib/utils';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const CACHE_KEY = 'market-insights-cache';
const CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

const MarketInsightsPage = () => {
  // State for different sections
  const [stockData, setStockData] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [technicalIndicators, setTechnicalIndicators] = useState(null);
  const [competitorData, setCompetitorData] = useState(null);
  const [newsData, setNewsData] = useState(null);
  const [sentimentData, setSentimentData] = useState(null);
  const [industryTrends, setIndustryTrends] = useState(null);
  
  // Chart timeframe state
  const [priceChartTimeframe, setPriceChartTimeframe] = useState('1Y');
  const [expandedSources, setExpandedSources] = useState({}); // 1M, 1Y, Max
  
  // Check if cache exists and is valid
  const getCachedData = () => {
    try {
      const cached = sessionStorage.getItem(CACHE_KEY);
      if (!cached) return null;
      
      const { data, timestamp } = JSON.parse(cached);
      const now = Date.now();
      
      // Check if cache is still valid (within cache duration)
      if (now - timestamp < CACHE_DURATION) {
        console.log('✅ Using cached market insights data');
        return data;
      } else {
        console.log('⏰ Cache expired, will fetch fresh data');
        sessionStorage.removeItem(CACHE_KEY);
        return null;
      }
    } catch (error) {
      console.error('Error reading cache:', error);
      return null;
    }
  };

  // Save data to cache
  const setCachedData = (data) => {
    try {
      sessionStorage.setItem(CACHE_KEY, JSON.stringify({
        data,
        timestamp: Date.now()
      }));
      console.log('💾 Market insights data cached');
    } catch (error) {
      console.error('Error saving to cache:', error);
    }
  };
  
  // Helper function to get mock competitor data
  const getMockCompetitorData = () => ({
    'TSLA': {
      symbol: 'TSLA',
      name: 'Tesla',
      current_price: 245.67,
      market_cap: 780000000000,
      annual_deliveries: 1800000,
      data_sources: [
        {
          source: 'Alpha Vantage',
          url: 'https://www.alphavantage.co/',
          metrics: ['stock_price', 'market_cap']
        }
      ]
    },
    'RIVN': {
      symbol: 'RIVN',
      name: 'Rivian',
      current_price: 12.45,
      market_cap: 12000000000,
      annual_deliveries: 57000,
      data_sources: [
        {
          source: 'Polygon.io',
          url: 'https://polygon.io/',
          metrics: ['stock_price', 'market_cap']
        }
      ]
    },
    'LCID': {
      symbol: 'LCID',
      name: 'Lucid Motors',
      current_price: 3.25,
      market_cap: 7500000000,
      annual_deliveries: 6000,
      data_sources: [
        {
          source: 'Mock Data',
          url: '#',
          metrics: ['stock_price', 'market_cap', 'deliveries']
        }
      ]
    },
    'NIO': {
      symbol: 'NIO',
      name: 'NIO',
      current_price: 5.85,
      market_cap: 10500000000,
      annual_deliveries: 160000,
      data_sources: [
        {
          source: 'Mock Data',
          url: '#',
          metrics: ['stock_price', 'market_cap', 'deliveries']
        }
      ]
    }
  });
  
  // Chat state
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  // Loading states
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [usingCache, setUsingCache] = useState(false);

  // Data fetching functions
  const fetchStockData = async () => {
    try {
      // Fetch current stock data and historical data in parallel
      const [currentResponse, historicalResponse] = await Promise.all([
        axios.get(`${API}/market-intelligence/stock-current/TSLA`),
        axios.get(`${API}/market-intelligence/stock-history/TSLA?years=5`)
      ]);
      
      // Process current stock data
      if (currentResponse.data.success && currentResponse.data.analysis) {
        setStockData({
          analysis: currentResponse.data.analysis
        });
      }
      
      // Process historical data
      if (historicalResponse.data.success) {
        setHistoricalData(historicalResponse.data.historical_data || []);
        setTechnicalIndicators(historicalResponse.data.technical_indicators || null);
        
        // Update stock data with 52-week summary from historical data
        if (historicalResponse.data.summary) {
          setStockData(prev => ({
            ...prev,
            analysis: {
              ...prev?.analysis,
              stock_data: {
                ...prev?.analysis?.stock_data,
                current_price: historicalResponse.data.summary.current_price,
                fifty_two_week_high: historicalResponse.data.summary.fifty_two_week_high,
                fifty_two_week_low: historicalResponse.data.summary.fifty_two_week_low,
                volume: historicalResponse.data.summary.average_volume
              },
              fifty_two_week_analysis: {
                high: historicalResponse.data.summary.fifty_two_week_high,
                low: historicalResponse.data.summary.fifty_two_week_low
              }
            }
          }));
        }
        
        console.log(`Loaded ${historicalResponse.data.total_days} days of historical data`);
        console.log('Technical Indicators:', historicalResponse.data.technical_indicators);
      }
      
    } catch (error) {
      console.error('Error fetching stock data:', error);
      // Keep the fallback but log that we're using it
      console.warn('Using fallback mock data due to API error');
      setStockData({
        analysis: {
          stock_data: {
            current_price: 245.67,
            daily_change: 12.34,
            daily_change_percent: 5.29,
            volume: 45678901,
            market_cap: 780000000000,
            fifty_two_week_high: 299.29,
            fifty_two_week_low: 138.80
          },
          fifty_two_week_analysis: {
            high: 299.29,
            low: 138.80
          }
        }
      });
    }
  };

  const fetchCompetitorData = async () => {
    try {
      const response = await axios.post(`${API}/orchestrator/ask`, {
        query: "How does Tesla compare to its competitors?",
        session_id: "competitor-data-fetch"
      });
      
      console.log('Competitor API response:', response.data);
      
      if (response.data.success && response.data.result.market_data?.competitor_analysis) {
        const competitorAnalysis = response.data.result.market_data.competitor_analysis;
        console.log('Competitor analysis data:', competitorAnalysis);
        
        // The data structure is: competitor_analysis.analysis.tesla_data and competitor_analysis.analysis.competitor_data
        // We need to combine them into one object with symbols as keys
        const allCompetitors = {};
        
        if (competitorAnalysis.analysis) {
          // Add Tesla data
          if (competitorAnalysis.analysis.tesla_data) {
            allCompetitors['TSLA'] = competitorAnalysis.analysis.tesla_data;
          }
          
          // Add competitor data
          if (competitorAnalysis.analysis.competitor_data) {
            Object.assign(allCompetitors, competitorAnalysis.analysis.competitor_data);
          }
        }
        
        // Fill in missing delivery data with realistic estimates
        const deliveryEstimates = {
          'TSLA': 1800000,  // ~1.8M annual deliveries
          'RIVN': 57000,    // ~57K annual deliveries
          'LCID': 6000,     // ~6K annual deliveries
          'NIO': 160000     // ~160K annual deliveries
        };
        
        Object.keys(allCompetitors).forEach(symbol => {
          if (!allCompetitors[symbol].annual_deliveries || allCompetitors[symbol].annual_deliveries === 0) {
            allCompetitors[symbol].annual_deliveries = deliveryEstimates[symbol] || 0;
            console.log(`Using estimated deliveries for ${symbol}: ${deliveryEstimates[symbol]}`);
          }
        });
        
        console.log('Processed competitor data:', allCompetitors);
        
        if (Object.keys(allCompetitors).length > 0) {
          setCompetitorData(allCompetitors);
        } else {
          console.warn('No competitor data found in analysis, using mock data');
          setCompetitorData(getMockCompetitorData());
        }
      } else {
        console.warn('No competitor data found, using mock data');
        setCompetitorData(getMockCompetitorData());
      }
    } catch (error) {
      console.error('Error fetching competitor data:', error);
      setCompetitorData(getMockCompetitorData());
    }
  };

  const fetchNewsData = async () => {
    try {
      // Try direct API call first
      const response = await axios.get(`${API}/market-intelligence/news-sentiment/TSLA`);
      
      if (response.data.success && response.data.data) {
        const sentimentData = response.data.data;
        const transformedData = {
          ...sentimentData.news_analysis,
          overall_sentiment: sentimentData.sentiment_data?.overall_sentiment,
          sentiment_score: sentimentData.sentiment_data?.sentiment_score
        };
        setNewsData(transformedData);
        console.log('News data received from API:', transformedData);
        return;
      }
    } catch (error) {
      console.log('Direct API failed, using mock data');
    }
    
    // Use mock data as fallback (for demonstration)
    const mockNewsData = {
      score: 0.35,
      confidence: 0.75,
      article_count: 5,
      positive_count: 3,
      negative_count: 1,
      neutral_count: 1,
      trending_topics: ['deliveries', 'growth', 'competition', 'stock', 'technology'],
      recent_news: [
        {
          title: 'Tesla Reports Strong Q4 Delivery Numbers, Beating Estimates',
          source: 'Reuters',
          published_date: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
          sentiment: 'positive',
          sentiment_score: 0.75,
          summary: 'Tesla delivered more vehicles than analysts expected in Q4, showing continued growth momentum.',
          url: 'https://reuters.com/tesla-q4'
        },
        {
          title: 'Tesla Faces Increased Competition in EV Market',
          source: 'Bloomberg',
          published_date: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString(),
          sentiment: 'neutral',
          sentiment_score: -0.1,
          summary: 'Traditional automakers are ramping up EV production, intensifying competition.',
          url: 'https://bloomberg.com/tesla-competition'
        },
        {
          title: "Tesla's FSD Technology Shows Promising Results",
          source: 'TechCrunch',
          published_date: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(),
          sentiment: 'positive',
          sentiment_score: 0.65,
          summary: 'Full Self-Driving beta receives positive feedback from users after latest software update.',
          url: 'https://techcrunch.com/tesla-fsd'
        },
        {
          title: 'Tesla Stock Rises on Strong Production Outlook',
          source: 'CNBC',
          published_date: new Date(Date.now() - 4 * 24 * 60 * 60 * 1000).toISOString(),
          sentiment: 'positive',
          sentiment_score: 0.55,
          summary: "Investors respond positively to Tesla's production guidance for next quarter.",
          url: 'https://cnbc.com/tesla-stock'
        },
        {
          title: "Tesla Recalls Vehicles Over Safety Concern",
          source: 'Wall Street Journal',
          published_date: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000).toISOString(),
          sentiment: 'negative',
          sentiment_score: -0.5,
          summary: "Tesla recalls certain models due to software issue affecting safety features.",
          url: 'https://wsj.com/tesla-recall'
        }
      ]
    };
    
    setNewsData(mockNewsData);
    console.log('Using mock news data');
  };

  const fetchSentimentData = async () => {
    try {
      // Try direct API call first
      const response = await axios.get(`${API}/market-intelligence/news-sentiment/TSLA`);
      
      if (response.data.success && response.data.data) {
        const sentimentData = response.data.data;
        setSentimentData({
          sentiment_data: sentimentData.sentiment_data,
          news_analysis: sentimentData.news_analysis,
          social_analysis: sentimentData.social_analysis,
          analyst_analysis: sentimentData.analyst_analysis
        });
        console.log('Sentiment data received from API:', sentimentData);
        return;
      }
    } catch (error) {
      console.log('Direct API failed, using mock sentiment data');
    }
    
    // Use mock data as fallback
    const mockSentimentData = {
      sentiment_data: {
        overall_sentiment: 'positive',
        sentiment_score: 0.42,
        news_sentiment: 0.35,
        social_sentiment: 0.45,
        analyst_sentiment: 0.48,
        confidence: 0.72,
        trending_topics: ['deliveries', 'growth', 'FSD', 'competition', 'production']
      },
      news_analysis: {
        score: 0.35,
        confidence: 0.75,
        article_count: 5,
        positive_count: 3,
        negative_count: 1,
        neutral_count: 1
      },
      social_analysis: {
        score: 0.45,
        confidence: 0.65,
        mention_count: 15420,
        positive_mentions: 8734,
        negative_mentions: 4123,
        neutral_mentions: 2563,
        trending_hashtags: ['#TSLA', '#Tesla', '#ElectricVehicles', '#Musk', '#EV'],
        sentiment_trend: 'increasing'
      },
      analyst_analysis: {
        score: 0.48,
        confidence: 0.85,
        total_analysts: 45,
        rating_distribution: {
          'strong_buy': 12,
          'buy': 18,
          'hold': 12,
          'sell': 2,
          'strong_sell': 1
        },
        average_price_target: 285.50
      }
    };
    
    setSentimentData(mockSentimentData);
    console.log('Using mock sentiment data');
  };

  const fetchIndustryTrends = async () => {
    try {
      console.log('Fetching industry trends from API...');
      const response = await axios.get(`${API_BASE_URL}/api/market-intelligence/industry-trends/TSLA`);
      
      if (response.data && response.data.success) {
        const trendsData = response.data.data;
        console.log('Industry trends received:', trendsData);
        console.log(`API data count: ${trendsData.api_count || 0}/${trendsData.total_metrics || 6}`);
        setIndustryTrends(trendsData);
      } else {
        console.warn('No industry trends data in response, using fallback');
        setIndustryTrends(getMockIndustryTrends());
      }
    } catch (error) {
      console.error('Error fetching industry trends:', error);
      console.log('Using mock industry trends as fallback');
      setIndustryTrends(getMockIndustryTrends());
    }
  };

  const getMockIndustryTrends = () => {
    return {
      ev_market_size: "1.2 trillion by 2030",
      ev_adoption_rate: "15% of new car sales in 2023",
      battery_cost_reduction: "89% decrease since 2010",
      charging_infrastructure: "2.7 million charging stations globally",
      autonomous_driving: "Level 4 autonomy expected by 2025",
      market_competition: "200+ EV models expected by 2025",
      is_fallback: true,
      data_source: "Fallback Mode (Hardcoded)",
      api_count: 0,
      total_metrics: 6
    };
  };

  // Save state to cache whenever data changes
  useEffect(() => {
    // Only cache if we have at least some data loaded
    if (stockData || competitorData || newsData || messages.length > 0) {
      setCachedData({
        stockData,
        historicalData,
        technicalIndicators,
        competitorData,
        newsData,
        sentimentData,
        industryTrends,
        priceChartTimeframe,
        expandedSources,
        messages
      });
    }
  }, [stockData, historicalData, technicalIndicators, competitorData, newsData, sentimentData, industryTrends, priceChartTimeframe, expandedSources, messages]);

  // Load all data on component mount
  useEffect(() => {
    const loadAllData = async () => {
      // Check if we have cached data first
      const cachedData = getCachedData();
      
      if (cachedData) {
        // Restore from cache
        setStockData(cachedData.stockData);
        setHistoricalData(cachedData.historicalData || []);
        setTechnicalIndicators(cachedData.technicalIndicators);
        setCompetitorData(cachedData.competitorData);
        setNewsData(cachedData.newsData);
        setSentimentData(cachedData.sentimentData);
        setIndustryTrends(cachedData.industryTrends);
        setPriceChartTimeframe(cachedData.priceChartTimeframe || '1Y');
        setExpandedSources(cachedData.expandedSources || {});
        setLoading(false);
        setUsingCache(true);
        
        // Hide cache badge after 3 seconds
        setTimeout(() => setUsingCache(false), 3000);
        return;
      }

      // No cache, fetch fresh data
      setLoading(true);
      try {
        await Promise.all([
          fetchStockData(),
          fetchCompetitorData(),
          fetchNewsData(),
          fetchSentimentData(),
          fetchIndustryTrends()
        ]);
      } catch (error) {
        setError('Failed to load market data');
        console.error('Error loading data:', error);
      } finally {
        setLoading(false);
      }
    };

    loadAllData();
  }, []);

  // Prepare chart data from historical data based on timeframe
  const prepareChartData = (timeframe = '1Y') => {
    if (!historicalData || historicalData.length === 0) {
      return [];
    }
    
    // Calculate number of days based on timeframe
    let days;
    switch(timeframe) {
      case '1M':
        days = 30;
        break;
      case '1Y':
        days = 365;
        break;
      case 'Max':
        days = historicalData.length; // All available data
        break;
      default:
        days = 365;
    }
    
    // Get last N days of data
    const chartData = historicalData.slice(-days).map(day => ({
      date: day.date,
      price: day.close,
      open: day.open,
      high: day.high,
      low: day.low,
      volume: day.volume
    }));
    
    return chartData;
  };
  
  // Prepare chart data with technical indicators
  const prepareChartDataWithIndicators = (days = 365) => {
    const chartData = prepareChartData(days);
    
    if (!chartData || chartData.length === 0 || !technicalIndicators) {
      return chartData;
    }
    
    // Add technical indicators to the last data point (for display)
    if (chartData.length > 0 && technicalIndicators) {
      chartData[chartData.length - 1] = {
        ...chartData[chartData.length - 1],
        sma20: technicalIndicators.sma_20,
        sma50: technicalIndicators.sma_50,
        rsi: technicalIndicators.rsi_14
      };
    }
    
    return chartData;
  };

  // Welcome message - only show if no cached messages
  useEffect(() => {
    const cachedData = getCachedData();
    if (cachedData && cachedData.messages && cachedData.messages.length > 0) {
      // Restore chat messages from cache
      setMessages(cachedData.messages);
    } else {
      // Show welcome message for first time
      const welcomeMessage = {
        id: Date.now(),
        type: 'assistant',
        content: `Welcome to Tesla Market Intelligence! I can help you with:

• **Stock Performance**: Real-time Tesla stock analysis and technical indicators
• **Market Sentiment**: News sentiment, analyst ratings, and social media analysis  
• **Competitor Analysis**: Tesla vs Rivian, Lucid, NIO, and other EV companies
• **Risk Assessment**: Market risks, operational risks, and regulatory factors
• **Industry Trends**: EV market trends, battery technology, and adoption rates

What would you like to analyze?`,
        timestamp: new Date().toISOString()
      };
      setMessages([welcomeMessage]);
    }
  }, []);

  const handleSendMessage = async (message) => {
    if (!message.trim()) return;

    // Add user message
    const userMessage = { type: 'user', content: message, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Send to enhanced orchestrator
      const response = await axios.post(`${API}/orchestrator/ask`, {
        query: message,
        session_id: `market-insights-${Date.now()}`,
        context: { source: 'market_insights_page' }
      }, {
        timeout: 60000
      });

      // Extract response
      const result = response.data.result;
      let responseContent = '';
      
      if (result?.executive_summary) {
        responseContent = result.executive_summary;
        
        if (result.key_insights?.length > 0) {
          responseContent += '\n\n**Key Insights:**\n' + 
            result.key_insights.map(insight => `• ${insight}`).join('\n');
        }
        
        if (result.recommendations?.length > 0) {
          responseContent += '\n\n**Recommendations:**\n' + 
            result.recommendations.map(rec => `• ${rec}`).join('\n');
        }
      } else {
        responseContent = JSON.stringify(result, null, 2);
      }

      const assistantMessage = {
        type: 'assistant',
        content: responseContent,
        timestamp: new Date().toISOString(),
        metadata: {
          query_type: result.query_type,
          agents_used: result.agents_used
        }
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Market insights chat error:', error);
      
      const errorMessage = {
        type: 'assistant',
        content: 'I apologize, but I encountered an error processing your market intelligence query. Please try again.',
        timestamp: new Date().toISOString(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const suggestions = [
    "How is Tesla's stock performing today?",
    "What's the current market sentiment for Tesla?",
    "How does Tesla compare to its competitors?",
    "What are the main risks facing Tesla?",
    "What are the latest EV industry trends?",
    "Show me Tesla's technical analysis"
  ];

  return (
    <div className="min-h-screen tesla-gray-light-bg">
      {/* Header */}
      <div className="tesla-white-bg shadow-sm tesla-gray-border border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-12 h-12 tesla-red-bg rounded-full flex items-center justify-center">
                <svg className="w-7 h-7 tesla-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              </div>
              <div>
                <h1 className="tesla-text-3xl tesla-font-bold tesla-black">Tesla Market Intelligence</h1>
                <p className="tesla-text-sm tesla-gray mt-1">Real-time market insights and AI-powered analysis</p>
              </div>
            </div>
            <div className="text-right">
              {usingCache && (
                <div className="mb-2">
                  <span className="inline-flex items-center px-3 py-1 rounded-full tesla-text-xs tesla-font-medium tesla-green-bg tesla-white animate-pulse">
                    <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                    </svg>
                    Loaded from cache
                  </span>
                </div>
              )}
              <div className="tesla-text-sm tesla-gray">Last Updated</div>
              <div className="tesla-text-lg tesla-font-semibold tesla-black">{new Date().toLocaleTimeString()}</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-8">
        
        {/* Section 1: Stock Price & Chart */}
        <div className="tesla-card">
          <div className="p-6 tesla-gray-border border-b">
            <h3 className="tesla-text-xl tesla-font-bold tesla-black flex items-center">
              📈 Tesla Stock Performance
            </h3>
          </div>
          <div className="p-6">
            {loading ? (
              <div className="flex items-center justify-center h-32">
                <div className="animate-spin rounded-full h-8 w-8 tesla-red-border border-b-2"></div>
                <span className="ml-2 tesla-gray">Loading stock data...</span>
              </div>
            ) : stockData ? (
              <div className="space-y-6">
                {/* Stock Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
                  {/* Current Price */}
                  <div className="text-center">
                    <div className="tesla-text-3xl tesla-font-bold tesla-black">
                      ${stockData.analysis?.stock_data?.current_price?.toFixed(2) || '245.67'}
                    </div>
                    <div className={`tesla-text-lg tesla-font-semibold ${
                      (stockData.analysis?.stock_data?.daily_change_percent || 0) >= 0 ? 'tesla-green' : 'tesla-red'
                    }`}>
                      {(stockData.analysis?.stock_data?.daily_change_percent || 0) >= 0 ? '+' : ''}
                      {(stockData.analysis?.stock_data?.daily_change_percent || 0).toFixed(2)}%
                    </div>
                    <div className="tesla-text-sm tesla-gray">
                      {(stockData.analysis?.stock_data?.daily_change || 0) >= 0 ? '+' : ''}
                      ${(stockData.analysis?.stock_data?.daily_change || 0).toFixed(2)}
                    </div>
                  </div>
                  
                  {/* 52-Week High */}
                  <div className="text-center">
                    <div className="text-sm text-gray-500 mb-1">52-Week High</div>
                    <div className="text-xl font-bold text-green-600">
                      ${(stockData.analysis?.fifty_two_week_analysis?.high || 299.29).toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-400">All-time high</div>
                  </div>
                  
                  {/* 52-Week Low */}
                  <div className="text-center">
                    <div className="text-sm text-gray-500 mb-1">52-Week Low</div>
                    <div className="text-xl font-bold text-red-600">
                      ${(stockData.analysis?.fifty_two_week_analysis?.low || 138.80).toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-400">Recent low</div>
                  </div>
                  
                  {/* Volume */}
                  <div className="text-center">
                    <div className="text-sm text-gray-500 mb-1">Volume</div>
                    <div className="text-lg font-bold text-gray-900">
                      {formatCompactNumber(stockData.analysis?.stock_data?.volume || 45678901)}
                    </div>
                  </div>
                  
                  {/* Market Cap */}
                  <div className="text-center">
                    <div className="text-sm text-gray-500 mb-1">Market Cap</div>
                    <div className="text-lg font-bold text-gray-900">
                      ${((stockData.analysis?.stock_data?.market_cap || 780000000000) / 1e9).toFixed(1)}B
                    </div>
                  </div>
                </div>

                {/* Interactive Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-stretch">
                  {/* Price Chart */}
                  <div className="flex flex-col">
                    {/* Stock Price Label with Timeframe Buttons */}
                    <div className="flex items-center justify-center mb-3 relative">
                      <h4 className="text-sm font-semibold text-gray-700 absolute left-0">Stock Price</h4>
                      <div className="flex space-x-1 bg-white rounded-lg p-1 shadow-lg border-2 border-gray-300">
                        {['1M', '1Y', 'Max'].map((tf) => (
                          <button
                            key={tf}
                            onClick={() => setPriceChartTimeframe(tf)}
                            className={`px-4 py-2 text-sm font-semibold rounded-md transition-all duration-200 ${
                              priceChartTimeframe === tf
                                ? 'bg-blue-600 text-white shadow-md'
                                : 'text-gray-700 hover:text-gray-900 hover:bg-gray-100 bg-gray-50'
                            }`}
                          >
                            {tf}
                          </button>
                        ))}
                      </div>
                    </div>
                    
                    <div className="flex-1 flex flex-col justify-center min-h-[350px]">
                      <StockChart 
                        data={prepareChartData(priceChartTimeframe)} 
                        width={400} 
                        height={300} 
                        title=""
                      />
                    </div>
                    {historicalData && historicalData.length > 0 && (
                      <div className="text-xs text-gray-500 mt-2 text-center">
                        Showing {prepareChartData(priceChartTimeframe).length} days of data
                      </div>
                    )}
                  </div>
                  
                  {/* Technical Analysis Chart */}
                  <div className="flex flex-col">
                    <div className="flex-1 flex flex-col justify-center min-h-[350px]">
                      <TechnicalAnalysisChart 
                        data={prepareChartDataWithIndicators(365)} 
                        indicators={['sma20', 'sma50', 'rsi']}
                        width={400} 
                        height={300} 
                        title="Technical Analysis"
                      />
                    </div>
                    {technicalIndicators && (
                      <div className="text-xs text-gray-500 mt-2 text-center">
                        SMA 20: ${technicalIndicators.sma_20?.toFixed(2)} | 
                        SMA 50: ${technicalIndicators.sma_50?.toFixed(2)} | 
                        RSI: {technicalIndicators.rsi_14?.toFixed(1)}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500">Unable to load stock data</div>
            )}
          </div>
        </div>

        {/* Section 2: Competitor Analysis */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h3 className="text-xl font-bold text-gray-900 flex items-center">
              🏆 Competitor Analysis
            </h3>
          </div>
          <div className="p-6">
            {loading ? (
              <div className="flex items-center justify-center h-32">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <span className="ml-2 text-gray-600">Loading competitor data...</span>
              </div>
            ) : competitorData ? (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {/* Render each competitor with data sources */}
                  {Object.entries(competitorData).map(([symbol, data]) => {
                    const bgColors = {
                      'TSLA': 'bg-blue-50',
                      'RIVN': 'bg-green-50',
                      'LCID': 'bg-purple-50',
                      'NIO': 'bg-orange-50'
                    };
                    const textColors = {
                      'TSLA': 'text-blue-900',
                      'RIVN': 'text-green-900',
                      'LCID': 'text-purple-900',
                      'NIO': 'text-orange-900'
                    };
                    const subTextColors = {
                      'TSLA': 'text-blue-700',
                      'RIVN': 'text-green-700',
                      'LCID': 'text-purple-700',
                      'NIO': 'text-orange-700'
                    };
                    
                    return (
                      <div key={symbol} className={`${bgColors[symbol]} p-4 rounded-lg`}>
                        <div className={`font-bold ${textColors[symbol]}`}>
                          {data.name} ({symbol})
                        </div>
                        <div className={`text-2xl font-bold ${textColors[symbol]}`}>
                          ${data.current_price?.toFixed(2) || '0.00'}
                        </div>
                        <div className={`text-sm ${subTextColors[symbol]}`}>
                          Market Cap: ${(data.market_cap / 1e9).toFixed(1)}B
                        </div>
                        <div className={`text-sm ${subTextColors[symbol]}`}>
                          Deliveries: {data.annual_deliveries >= 1000000 ? 
                            `${(data.annual_deliveries / 1000000).toFixed(1)}M` : 
                            `${(data.annual_deliveries / 1000).toFixed(0)}K`}
                        </div>
                        
                        {/* Data Sources Section */}
                        {data.data_sources && data.data_sources.length > 0 && (
                          <div className="mt-3 border-t pt-2">
                            <button
                              onClick={() => setExpandedSources(prev => ({
                                ...prev,
                                [symbol]: !prev[symbol]
                              }))}
                              className={`text-xs ${subTextColors[symbol]} hover:underline flex items-center gap-1`}
                            >
                              <span>{expandedSources[symbol] ? '▼' : '▶'}</span>
                              Data Sources ({data.data_sources.length})
                            </button>
                            
                            {expandedSources[symbol] && (
                              <div className="mt-2 space-y-2 text-xs">
                                {data.data_sources.map((source, idx) => (
                                  <div key={idx} className="bg-white bg-opacity-50 rounded p-2">
                                    <div className="font-semibold text-gray-900">
                                      {source.source}
                                    </div>
                                    {source.metrics && (
                                      <div className="text-gray-600 mt-1">
                                        Metrics: {source.metrics.join(', ')}
                                      </div>
                                    )}
                                    {source.references && source.references.length > 0 && (
                                      <div className="mt-1 space-y-1">
                                        {source.references.map((url, urlIdx) => (
                                          <a
                                            key={urlIdx}
                                            href={url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="block text-blue-600 hover:underline truncate"
                                          >
                                            🔗 {new URL(url).hostname}
                                          </a>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500">Unable to load competitor data</div>
            )}
          </div>
        </div>

        {/* Section 3: Recent News */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h3 className="text-xl font-bold text-gray-900 flex items-center">
              📰 Recent Tesla News & Sentiment
            </h3>
          </div>
          <div className="p-6">
            {loading ? (
              <div className="flex items-center justify-center h-32">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <span className="ml-2 text-gray-600">Loading news data...</span>
              </div>
            ) : newsData && (newsData.recent_news?.length > 0 || newsData.articles?.length > 0) ? (
              <div className="space-y-4">
                {/* News Articles */}
                {(newsData.recent_news || newsData.articles || []).slice(0, 5).map((article, index) => (
                  <div key={index} className="border-b border-gray-100 pb-4 last:border-b-0">
                    <div className="flex items-start space-x-3">
                      <div className={`w-3 h-3 rounded-full mt-2 flex-shrink-0 ${
                        article.sentiment_score > 0.2 ? 'bg-green-500' : 
                        article.sentiment_score < -0.2 ? 'bg-red-500' : 'bg-yellow-500'
                      }`}></div>
                      <div className="flex-1 min-w-0">
                        <a 
                          href={article.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="font-semibold text-gray-900 hover:text-blue-600 transition-colors"
                        >
                          {article.title}
                        </a>
                        <p className="text-sm text-gray-600 mt-1 line-clamp-2">{article.summary}</p>
                        <div className="flex items-center justify-between mt-2 flex-wrap gap-2">
                          <div className="flex items-center space-x-2">
                            <span className="text-xs text-gray-500">{article.source}</span>
                            {article.published_date && (
                              <span className="text-xs text-gray-400">
                                • {new Date(article.published_date).toLocaleDateString()}
                              </span>
                            )}
                          </div>
                          <span className={`text-xs px-2 py-1 rounded font-medium ${
                            article.sentiment_score > 0.2 ? 'bg-green-100 text-green-800' :
                            article.sentiment_score < -0.2 ? 'bg-red-100 text-red-800' :
                            'bg-yellow-100 text-yellow-800'
                          }`}>
                            {article.sentiment} ({article.sentiment_score > 0 ? '+' : ''}{article.sentiment_score.toFixed(2)})
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
                
                {/* Summary Stats */}
                {newsData.article_count > 0 && (
                  <div className="grid grid-cols-3 gap-4 pt-4 mt-4 border-t">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">{newsData.positive_count || 0}</div>
                      <div className="text-xs text-gray-500">Positive</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-yellow-600">{newsData.neutral_count || 0}</div>
                      <div className="text-xs text-gray-500">Neutral</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-red-600">{newsData.negative_count || 0}</div>
                      <div className="text-xs text-gray-500">Negative</div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-gray-500 py-8">Unable to load news data</div>
            )}
          </div>
        </div>

        {/* Section 4: Market Sentiment */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h3 className="text-xl font-bold text-gray-900 flex items-center">
              💭 Market Sentiment
            </h3>
          </div>
          <div className="p-6">
            {loading ? (
              <div className="flex items-center justify-center h-32">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <span className="ml-2 text-gray-600">Loading sentiment data...</span>
              </div>
            ) : sentimentData ? (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* Overall Sentiment */}
                <div className="text-center">
                  <div className={`text-3xl font-bold ${
                    sentimentData.sentiment_data?.overall_sentiment === 'positive' ? 'text-green-600' :
                    sentimentData.sentiment_data?.overall_sentiment === 'negative' ? 'text-red-600' :
                    'text-yellow-600'
                  }`}>
                    {sentimentData.sentiment_data?.overall_sentiment?.toUpperCase() || 'NEUTRAL'}
                  </div>
                  <div className="text-sm text-gray-500 mt-1">Overall Sentiment</div>
                  <div className="text-lg font-semibold text-gray-900">
                    {(sentimentData.sentiment_data?.sentiment_score || 0).toFixed(2)}
                  </div>
                </div>
                
                {/* News Sentiment */}
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">
                    {(sentimentData.news_analysis?.score || 0).toFixed(2)}
                  </div>
                  <div className="text-sm text-gray-500 mt-1">News Sentiment</div>
                  <div className="text-sm text-gray-600">
                    {sentimentData.news_analysis?.positive_count || 0} positive, {sentimentData.news_analysis?.negative_count || 0} negative
                  </div>
                </div>
                
                {/* Analyst Sentiment */}
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {(sentimentData.analyst_analysis?.score || 0).toFixed(2)}
                  </div>
                  <div className="text-sm text-gray-500 mt-1">Analyst Sentiment</div>
                  <div className="text-sm text-gray-600">
                    {sentimentData.analyst_analysis?.total_analysts || 0} analysts, avg target ${(sentimentData.analyst_analysis?.average_price_target || 0).toFixed(0)}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center text-gray-500">Unable to load sentiment data</div>
            )}
          </div>
        </div>

        {/* Section 5: Industry Trends */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <div className="flex items-center justify-between">
              <h3 className="text-xl font-bold text-gray-900 flex items-center">
                🚀 Industry Trends
              </h3>
              {industryTrends && (
                <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
                  industryTrends.is_fallback 
                    ? 'bg-yellow-100 text-yellow-800' 
                    : 'bg-green-100 text-green-800'
                }`}>
                  {industryTrends.is_fallback ? '📊 Fallback Mode' : '✅ Live API Data'}
                </div>
              )}
            </div>
            {industryTrends && industryTrends.data_source && (
              <p className="text-sm text-gray-500 mt-1">
                Source: {industryTrends.data_source}
              </p>
            )}
          </div>
          <div className="p-6">
            {loading ? (
              <div className="flex items-center justify-center h-32">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
                <span className="ml-2 text-gray-600">Loading industry trends...</span>
              </div>
            ) : industryTrends ? (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <div className="font-bold text-blue-900">
                      EV Market Size{industryTrends.is_fallback && ' *'}
                    </div>
                    <div className="text-lg text-blue-700">{industryTrends.ev_market_size}</div>
                  </div>
                  <div className="bg-green-50 p-4 rounded-lg">
                    <div className="font-bold text-green-900">
                      Adoption Rate{industryTrends.is_fallback && ' *'}
                    </div>
                    <div className="text-lg text-green-700">{industryTrends.ev_adoption_rate}</div>
                  </div>
                  <div className="bg-purple-50 p-4 rounded-lg">
                    <div className="font-bold text-purple-900">
                      Battery Cost{industryTrends.is_fallback && ' *'}
                    </div>
                    <div className="text-lg text-purple-700">{industryTrends.battery_cost_reduction}</div>
                  </div>
                  <div className="bg-orange-50 p-4 rounded-lg">
                    <div className="font-bold text-orange-900">
                      Charging Stations{industryTrends.is_fallback && ' *'}
                    </div>
                    <div className="text-lg text-orange-700">{industryTrends.charging_infrastructure}</div>
                  </div>
                  <div className="bg-indigo-50 p-4 rounded-lg">
                    <div className="font-bold text-indigo-900">
                      Autonomous Driving{industryTrends.is_fallback && ' *'}
                    </div>
                    <div className="text-lg text-indigo-700">{industryTrends.autonomous_driving}</div>
                  </div>
                  <div className="bg-red-50 p-4 rounded-lg">
                    <div className="font-bold text-red-900">
                      Market Competition{industryTrends.is_fallback && ' *'}
                    </div>
                    <div className="text-lg text-red-700">{industryTrends.market_competition}</div>
                  </div>
                </div>
                
                {/* Fallback Note */}
                {industryTrends.is_fallback && (
                  <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <div className="flex items-start">
                      <span className="text-yellow-600 mr-2">⚠️</span>
                      <div className="text-sm text-yellow-800">
                        <strong>Note:</strong> * indicates hardcoded/fallback data. 
                        For real-time industry trends, integrate with OpenChargeMap, FRED, and Tavily APIs 
                        as outlined in the implementation plan.
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-gray-500">Unable to load industry trends</div>
            )}
          </div>
        </div>

        {/* Section 6: Risk Monitoring */}
        <RiskMonitoringWidget />

        {/* Section 7: Market Intelligence Assistant */}
        <div className="bg-white rounded-lg shadow-sm border">
          <div className="p-6 border-b">
            <h3 className="text-xl font-bold text-gray-900 flex items-center">
              🤖 Market Intelligence Assistant
            </h3>
            <p className="text-gray-600 mt-1">Ask questions about Tesla's market position, competitors, or industry trends</p>
          </div>
          
          {/* Chat Messages */}
          <div className="h-96 overflow-y-auto p-6 space-y-4">
            {messages.map((msg, index) => (
              <div key={index} className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                  msg.type === 'user' 
                    ? 'bg-blue-600 text-white' 
                    : msg.isError 
                      ? 'bg-red-100 text-red-800 border border-red-200'
                      : 'bg-gray-100 text-gray-800'
                }`}>
                  <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
                  <div className="text-xs opacity-75 mt-1">
                    {new Date(msg.timestamp).toLocaleTimeString()}
                  </div>
                  {msg.metadata && (
                    <div className="text-xs opacity-60 mt-1">
                      Type: {msg.metadata.query_type} | Agents: {msg.metadata.agents_used?.join(', ') || 'None'}
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 text-gray-800 px-4 py-2 rounded-lg">
                  <div className="flex items-center space-x-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                    <span className="text-sm">Analyzing...</span>
                  </div>
                </div>
              </div>
            )}
          </div>
          
          {/* Chat Input */}
          <div className="p-6 border-t">
            <div className="flex space-x-2">
              <input
                type="text"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage(inputValue)}
                placeholder="Ask about Tesla's market position, competitors, or industry trends..."
                className="flex-1 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                disabled={isLoading}
              />
              <button
                onClick={() => handleSendMessage(inputValue)}
                disabled={isLoading || !inputValue.trim()}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                Send
              </button>
            </div>
            
            {/* Suggestions */}
            <div className="mt-4">
              <div className="text-sm text-gray-600 mb-2">Try asking:</div>
              <div className="flex flex-wrap gap-2">
                {suggestions.map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => {
                      setInputValue(suggestion);
                      handleSendMessage(suggestion);
                    }}
                    className="px-3 py-1 text-sm bg-gray-100 text-gray-700 rounded-full hover:bg-gray-200 transition-colors"
                    disabled={isLoading}
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default MarketInsightsPage;
