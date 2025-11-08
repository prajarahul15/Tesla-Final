"""
Tesla Financial Assistant Agent
Provides AI-powered responses for financial modeling questions
"""

import os
from typing import Dict, List, Optional
from openai import OpenAI
import json

class TeslaFAAgent:
    """
    AI-powered Tesla Financial Assistant for answering questions about financial modeling
    """
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o-mini"
        
        # Financial context and capabilities
        self.capabilities = [
            "Revenue forecasting and analysis",
            "Cost structure and margin analysis", 
            "Financial statement interpretation",
            "Scenario analysis (Base, Best, Worst cases)",
            "DCF valuation methodology",
            "Vehicle model performance analysis",
            "Price elasticity and demand modeling",
            "Macroeconomic impact assessment",
            "Risk analysis and sensitivity testing",
            "Strategic financial recommendations"
        ]
        
        self.data_sources = [
            "Tesla monthly financial data (Revenue, COGS, R&D, SG&A)",
            "Vehicle delivery and production data by model",
            "Historical financial statements (Income, Balance Sheet, Cash Flow)",
            "DCF valuation models and assumptions",
            "Scenario analysis results",
            "Macroeconomic indicators and forecasts",
            "Price elasticity models",
            "Market analysis and competitive positioning"
        ]
    
    def generate_response(self, user_message: str, context: str = "financial_modeling") -> str:
        """
        Generate AI response for Tesla financial modeling questions
        
        Args:
            user_message: User's question or message
            context: Context type (financial_modeling, etc.)
            
        Returns:
            AI-generated response
        """
        try:
            # Create comprehensive system prompt
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with context
            user_prompt = self._create_user_prompt(user_message, context)
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1000,
                temperature=0.7,
                timeout=30
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating Tesla FA response: {e}")
            return self._get_fallback_response(user_message)
    
    def _create_system_prompt(self) -> str:
        """Create comprehensive system prompt for Tesla FA"""
        return f"""You are Tesla Financial Assistant, an AI-powered financial modeling expert specializing in Tesla's business and financial analysis.

## Your Role
You help users understand Tesla's financial modeling, analyze data, interpret forecasts, and provide strategic insights. You have access to comprehensive Tesla financial data and advanced modeling capabilities.

## Your Capabilities
{self._format_capabilities()}

## Available Data Sources
{self._format_data_sources()}

## Response Guidelines
1. **Be Accurate**: Base responses on actual financial data and sound modeling principles
2. **Be Clear**: Explain complex financial concepts in accessible language
3. **Be Specific**: Provide concrete numbers, percentages, and timeframes when relevant
4. **Be Helpful**: Offer actionable insights and recommendations
5. **Be Professional**: Maintain a knowledgeable but approachable tone

## Key Financial Metrics You Can Discuss
- Revenue: Automotive, Services, Energy, Total
- Costs: COGS, R&D, SG&A, Operating Expenses
- Profitability: Gross Margin, Operating Margin, Net Margin
- Cash Flow: Operating, Investing, Financing, Free Cash Flow
- Valuation: DCF, P/E ratios, Enterprise Value
- Growth: Revenue growth, delivery growth, market expansion

## Scenario Analysis
You can explain and analyze:
- Base Case: Conservative assumptions
- Best Case: Optimistic growth scenarios  
- Worst Case: Challenging market conditions
- Sensitivity analysis and key drivers

## When You Don't Know
- Acknowledge limitations
- Suggest alternative approaches
- Offer to help with related questions
- Direct users to specific data sources

## Special Instructions for Driver Impact Questions
If the user asks about the impact of a change in a financial driver (e.g., 'What would be the impact on the Income statement, if the Automotive Revenue is changed by 29% from the base year 2024 Automotive Revenue?'), you must:
1. Retrieve the base year value from the model/data.
2. Calculate the new value after the specified change.
3. Propagate the change through the Income Statement (COGS, Gross Profit, Net Profit), assuming margins and OpEx are constant unless otherwise specified.
4. Present a before/after table for all key metrics.
5. Clearly state all assumptions.
6. Reference the data source.
7. Provide a brief summary and actionable recommendations.

Always return your answer as a JSON object with these fields:
- before_after_table: (array of objects with metric, base_value, new_value)
- calculation_steps: (step-by-step explanation)
- assumptions: (list)
- data_source: (string)
- summary: (string)
- recommendations: (list)

Always provide context, explain your reasoning, and offer to dive deeper into specific areas of interest."""

    def _format_capabilities(self) -> str:
        """Format capabilities list"""
        return "\n".join([f"• {cap}" for cap in self.capabilities])
    
    def _format_data_sources(self) -> str:
        """Format data sources list"""
        return "\n".join([f"• {source}" for source in self.data_sources])
    
    def _create_user_prompt(self, user_message: str, context: str) -> str:
        """Create user prompt with context and enforce structured output for driver impact questions"""
        return f"""Context: {context}

User Question: {user_message}

If the question is about the impact of a change in a financial driver (such as a percentage change in revenue, cost, or margin), you must:
- Retrieve the relevant base year value from the data/model
- Calculate the new value after the specified change
- Propagate the change through the Income Statement (COGS, Gross Profit, Net Profit), assuming margins and OpEx are constant unless otherwise specified
- Present a before/after table for all key metrics
- Clearly state all assumptions
- Reference the data source
- Provide a brief summary and actionable recommendations

Always return your answer as a JSON object with these fields:
- before_after_table: (array of objects with metric, base_value, new_value)
- calculation_steps: (step-by-step explanation)
- assumptions: (list)
- data_source: (string)
- summary: (string)
- recommendations: (list)

If the question is general, provide insights and suggest specific areas to explore further."""

    def _get_fallback_response(self, user_message: str) -> str:
        """Fallback response when AI is unavailable"""
        return f"""I apologize, but I'm currently experiencing technical difficulties and cannot process your question: "{user_message}"

However, I can help you with:
• Revenue forecasting and analysis
• Cost structure and margin analysis
• Financial statement interpretation
• Scenario analysis and DCF valuation
• Vehicle model performance insights

Please try rephrasing your question or ask about a specific aspect of Tesla's financial modeling. I'm here to help you understand the data and make informed decisions."""

    def get_capabilities_summary(self) -> Dict:
        """Get summary of Tesla FA capabilities"""
        return {
            "name": "Tesla Financial Assistant",
            "description": "AI-powered financial modeling expert for Tesla analysis",
            "capabilities": self.capabilities,
            "data_sources": self.data_sources,
            "model": self.model
        }

# Global instance
tesla_fa_agent = TeslaFAAgent()
