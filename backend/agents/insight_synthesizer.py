"""
Insight Synthesizer for Multi-Agent Coordination
Intelligently combines insights from multiple agents rather than simple concatenation
"""

from typing import Dict, List, Any, Optional, Tuple
import re
from collections import defaultdict

class InsightSynthesizer:
    """
    Synthesizes insights from multiple agents into cohesive, integrated analysis
    """
    
    def __init__(self):
        self.financial_keywords = {
            'revenue', 'profit', 'margin', 'eps', 'cash flow', 'valuation',
            'dcf', 'cagr', 'growth', 'cost', 'expenses'
        }
        self.market_keywords = {
            'stock', 'price', 'market', 'sentiment', 'volatility', 'pe ratio',
            'competition', 'trend', 'bullish', 'bearish'
        }
    
    def synthesize(self, 
                   market_insights: List[str],
                   financial_insights: List[str],
                   query_context: str = "") -> List[str]:
        """
        Synthesize insights from market and financial agents
        Creates integrated insights rather than simple concatenation
        """
        
        # Extract key themes
        market_themes = self._extract_themes(market_insights, self.market_keywords)
        financial_themes = self._extract_themes(financial_insights, self.financial_keywords)
        
        # Find synergies and connections
        synergies = self._find_synergies(market_themes, financial_themes)
        
        # Create integrated insights
        integrated_insights = []
        
        # Add synergy insights first (most valuable)
        for synergy in synergies[:2]:  # Top 2 synergies
            integrated_insights.append(synergy)
        
        # Add best unique insights from each domain
        integrated_insights.extend(self._select_best_insights(market_insights, 2, 'market'))
        integrated_insights.extend(self._select_best_insights(financial_insights, 2, 'financial'))
        
        return integrated_insights[:5]  # Return top 5 integrated insights
    
    def _extract_themes(self, insights: List[str], keywords: set) -> Dict[str, List[str]]:
        """Extract thematic elements from insights"""
        themes = defaultdict(list)
        
        for insight in insights:
            insight_lower = insight.lower()
            for keyword in keywords:
                if keyword in insight_lower:
                    themes[keyword].append(insight)
        
        return dict(themes)
    
    def _find_synergies(self, market_themes: Dict, financial_themes: Dict) -> List[str]:
        """Find connections between market and financial insights"""
        synergies = []
        
        # Example synergy patterns
        synergy_patterns = [
            ('stock', 'valuation', 'Market valuation aligns with financial fundamentals'),
            ('volatility', 'margin', 'Market volatility reflects margin uncertainty'),
            ('sentiment', 'growth', 'Market sentiment driven by growth trajectory'),
            ('price', 'eps', 'Stock price movement correlates with earnings performance'),
            ('trend', 'revenue', 'Market trend reflects revenue momentum')
        ]
        
        for market_key, financial_key, template in synergy_patterns:
            if market_key in market_themes and financial_key in financial_themes:
                # Create synergy insight
                market_data = self._extract_numbers(market_themes[market_key])
                financial_data = self._extract_numbers(financial_themes[financial_key])
                
                if market_data and financial_data:
                    synergy = self._create_synergy_insight(
                        market_key, financial_key, 
                        market_data, financial_data, template
                    )
                    if synergy:
                        synergies.append(synergy)
        
        return synergies
    
    def _extract_numbers(self, insights: List[str]) -> List[Tuple[float, str]]:
        """Extract numerical values and their context from insights"""
        numbers = []
        
        for insight in insights:
            # Find numbers with context
            matches = re.findall(r'([\d.]+)([%$BM]|\s*billion|\s*million)?', insight)
            for value, unit in matches:
                try:
                    num = float(value)
                    numbers.append((num, unit.strip() if unit else ''))
                except ValueError:
                    continue
        
        return numbers
    
    def _create_synergy_insight(self, market_key: str, financial_key: str,
                                market_data: List, financial_data: List,
                                template: str) -> Optional[str]:
        """Create an integrated insight showing the connection"""
        
        # This is a simplified version - in production, would be more sophisticated
        if market_data and financial_data:
            return f"🔗 {template}: Market {market_key} dynamics complement {financial_key} fundamentals"
        
        return None
    
    def _select_best_insights(self, insights: List[str], count: int, source: str) -> List[str]:
        """Select the most valuable insights from a list"""
        if not insights:
            return []
        
        # Score insights by quality indicators
        scored_insights = []
        for insight in insights:
            score = self._score_insight(insight)
            scored_insights.append((score, insight))
        
        # Sort by score and take top N
        scored_insights.sort(reverse=True, key=lambda x: x[0])
        
        # Add source prefix
        return [f"{source.title()}: {insight}" for _, insight in scored_insights[:count]]
    
    def _score_insight(self, insight: str) -> float:
        """Score insight quality based on heuristics"""
        score = 0.0
        
        # Longer insights tend to be more detailed
        score += min(len(insight) / 200, 1.0) * 0.3
        
        # Insights with numbers are more concrete
        if re.search(r'\d+', insight):
            score += 0.3
        
        # Insights with percentages are valuable
        if '%' in insight:
            score += 0.2
        
        # Insights with dollar amounts are concrete
        if '$' in insight:
            score += 0.2
        
        return score
    
    def resolve_conflicts(self, conflicting_insights: List[Dict]) -> str:
        """
        Resolve conflicting insights from different agents
        Returns a balanced perspective
        """
        if not conflicting_insights:
            return ""
        
        # Group by sentiment
        bullish = []
        bearish = []
        neutral = []
        
        for insight in conflicting_insights:
            text = insight.get('text', '').lower()
            if any(word in text for word in ['increase', 'grow', 'improve', 'strong', 'positive']):
                bullish.append(insight)
            elif any(word in text for word in ['decrease', 'decline', 'weak', 'negative', 'risk']):
                bearish.append(insight)
            else:
                neutral.append(insight)
        
        # Create balanced perspective
        if bullish and bearish:
            return (f"⚖️ Mixed signals detected: {len(bullish)} positive indicators vs "
                   f"{len(bearish)} risk factors suggest monitoring both opportunities and risks")
        elif bullish:
            return "📈 Consensus positive outlook with multiple supporting factors"
        elif bearish:
            return "📉 Consensus caution with multiple risk indicators"
        else:
            return "➡️ Neutral outlook, awaiting clearer signals"
    
    def create_executive_summary(self, 
                                 market_summary: str,
                                 financial_summary: str,
                                 synergies: List[str]) -> str:
        """
        Create an integrated executive summary
        Rather than concatenating, synthesizes key points
        """
        
        # Extract key points from each
        market_key = self._extract_key_point(market_summary)
        financial_key = self._extract_key_point(financial_summary)
        
        # Create integrated summary
        if synergies:
            summary = (f"Integrated Analysis: {market_key} The financial fundamentals show "
                      f"{financial_key.lower()} These factors create {synergies[0].lower()}")
        else:
            summary = (f"Combined Perspective: Market analysis indicates {market_key.lower()} "
                      f"while financial analysis reveals {financial_key.lower()}")
        
        return summary
    
    def _extract_key_point(self, text: str) -> str:
        """Extract the most important point from a summary"""
        if not text:
            return "analysis pending"
        
        # Take first sentence or first 150 characters
        sentences = text.split('.')
        if sentences:
            return sentences[0].strip() + '.'
        
        return text[:150].strip() + '...'
    
    def generate_integrated_recommendations(self,
                                           market_recs: List[str],
                                           financial_recs: List[str]) -> List[str]:
        """
        Generate integrated recommendations that consider both market and financial factors
        """
        integrated_recs = []
        
        # Categorize recommendations
        short_term = []
        long_term = []
        risk_mgmt = []
        
        all_recs = [(rec, 'market') for rec in market_recs] + [(rec, 'financial') for rec in financial_recs]
        
        for rec, source in all_recs:
            rec_lower = rec.lower()
            if any(word in rec_lower for word in ['monitor', 'watch', 'track', 'immediate']):
                short_term.append((rec, source))
            elif any(word in rec_lower for word in ['long-term', 'strategic', 'invest', 'develop']):
                long_term.append((rec, source))
            elif any(word in rec_lower for word in ['risk', 'hedge', 'protect', 'mitigate']):
                risk_mgmt.append((rec, source))
        
        # Create integrated recommendations
        if short_term:
            integrated_recs.append(f"Near-term: {short_term[0][0]}")
        if long_term:
            integrated_recs.append(f"Strategic: {long_term[0][0]}")
        if risk_mgmt:
            integrated_recs.append(f"Risk Management: {risk_mgmt[0][0]}")
        
        # Add best remaining
        remaining = [rec for rec, _ in all_recs 
                    if not any(rec in ir for ir in integrated_recs)]
        integrated_recs.extend(remaining[:2])
        
        return integrated_recs[:4]  # Top 4 integrated recommendations

