import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ChatWindow from './ChatWindow';
import MessageBubble from './MessageBubble';
import ChatInput from './ChatInput';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const TeslaFAChat = () => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const messagesEndRef = useRef(null);

  // Welcome message
  useEffect(() => {
    const welcomeMessage = {
      id: Date.now(),
      type: 'assistant',
      content: `Hello! 👋 I'm your enhanced **Tesla Financial & Market Intelligence Assistant**.

I can help you with:

**📊 Financial Analysis**
• Revenue, costs, and profitability analysis
• Income statement, balance sheet, and cash flow insights
• Scenario modeling and forecasting simulations
• Growth impact calculations and what-if analysis
**📈 Market Intelligence**
• Tesla stock performance and technical analysis
• Market sentiment and analyst ratings
• Competitor analysis and industry trends
• Risk assessment and market alerts
**💡 Pro Tips:**
• Ask specific questions for detailed analysis
• Use the quick questions below to get started
• I can simulate scenarios and show cross-statement impacts

**Ready to dive in?** What would you like to know about Tesla's financials?`,
      timestamp: new Date().toISOString()
    };
    setMessages([welcomeMessage]);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (messageText) => {
    if (!messageText.trim()) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: messageText,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      // Use orchestrator for intelligent multi-agent coordination
      const response = await axios.post(`${API}/orchestrator/ask`, {
        query: messageText,
        session_id: `tesla-fa-${Date.now()}`,
        context: {
          scenario: 'base',
          source: 'tesla_fa_chat'
        }
      }, {
        timeout: 60000  // Increased timeout for multi-agent workflows
      });

      // Extract response from orchestrated result
      let responseContent = '';
      
      const result = response.data.result;
      
      // Handle different response formats
      if (result?.executive_summary) {
        responseContent = result.executive_summary;
        
        // Add key insights if available
        if (Array.isArray(result.key_insights)) {
          responseContent += '\n\n**Key Insights:**\n' + 
            result.key_insights.map(insight => 
              typeof insight === 'string' ? `• ${insight}` : `• ${JSON.stringify(insight)}`
            ).join('\n');
        }
        
        // Add recommendations if available
        if (Array.isArray(result.recommendations)) {
          responseContent += '\n\n**Recommendations:**\n' + 
            result.recommendations.map(rec => 
              typeof rec === 'string' ? `• ${rec}` : `• ${JSON.stringify(rec)}`
            ).join('\n');
        }
        
        // Add next steps if available
        if (Array.isArray(result.next_steps)) {
          responseContent += '\n\n**Next Steps:**\n' + 
            result.next_steps.map((step, i) => 
              typeof step === 'string' ? `${i+1}. ${step}` : `${i+1}. ${JSON.stringify(step)}`
            ).join('\n');
        }
      } else if (result?.response && typeof result.response === 'string') {
        responseContent = result.response;
      } else if (result?.summary && typeof result.summary === 'string') {
        responseContent = result.summary;
      } else if (result?.results) {
        // Handle results object - convert to readable format
        responseContent = 'Analysis completed. Here are the results:\n\n';
        Object.entries(result.results).forEach(([key, value]) => {
          responseContent += `**${key}:**\n`;
          if (typeof value === 'object') {
            responseContent += JSON.stringify(value, null, 2) + '\n\n';
          } else {
            responseContent += `${value}\n\n`;
          }
        });
      } else {
        // Fallback: convert entire result to readable JSON
        responseContent = 'Here are the results:\n\n```json\n' + 
          JSON.stringify(result, null, 2) + '\n```';
      }

      const assistantMessage = {
        id: Date.now() + 1,
        type: 'assistant',
        content: responseContent,
        timestamp: new Date().toISOString(),
        metadata: {
          agents_used: response.data.agents_used,
          tasks_executed: response.data.tasks_executed
        }
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);
      
      // Fallback to single-agent Tesla FA if orchestrator fails
      try {
        const fallbackResponse = await axios.post(`${API}/tesla-fa/chat`, {
          message: messageText,
          context: 'financial_modeling'
        }, {
          timeout: 30000
        });

        const assistantMessage = {
          id: Date.now() + 1,
          type: 'assistant',
          content: fallbackResponse.data.response,
          timestamp: new Date().toISOString()
        };

        setMessages(prev => [...prev, assistantMessage]);
      } catch (fallbackError) {
        const errorMessage = {
          id: Date.now() + 1,
          type: 'assistant',
          content: 'I apologize, but I encountered an error processing your request. Please try again or rephrase your question.',
          timestamp: new Date().toISOString(),
          isError: true
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleMinimize = () => {
    setIsMinimized(!isMinimized);
  };

  const handleMaximize = () => {
    setIsMaximized(!isMaximized);
  };

  const handleClose = () => {
    // Navigate back to previous page or home
    window.history.back();
  };

  return (
    <div className="min-h-screen tesla-gray-light-bg">
      {/* Header */}
      <div className="tesla-white-bg shadow-sm tesla-gray-border border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 tesla-red-bg rounded-full flex items-center justify-center">
                <svg className="w-6 h-6 tesla-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
              <div>
                <h1 className="tesla-text-2xl tesla-font-bold tesla-black">Tesla Financial Assistant</h1>
                <p className="tesla-text-sm tesla-gray">Your AI-powered financial modeling companion</p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <button
                onClick={handleMinimize}
                className="p-2 tesla-gray hover:tesla-black hover:tesla-gray-light-bg rounded-lg transition-colors"
                title="Minimize"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
                </svg>
              </button>
              <button
                onClick={handleMaximize}
                className="p-2 tesla-gray hover:tesla-black hover:tesla-gray-light-bg rounded-lg transition-colors"
                title={isMaximized ? "Restore" : "Maximize"}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={isMaximized ? "M9 9V4.5M9 9H4.5M9 9L3.5 3.5M15 9h4.5M15 9V4.5M15 9l5.5-5.5" : "M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"} />
                </svg>
              </button>
              <button
                onClick={handleClose}
                className="p-2 tesla-gray hover:tesla-red hover:tesla-gray-light-bg rounded-lg transition-colors"
                title="Close"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Chat Window */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <ChatWindow
          isMinimized={isMinimized}
          isMaximized={isMaximized}
          onMinimize={handleMinimize}
          onMaximize={handleMaximize}
          onClose={handleClose}
        >
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
              />
            ))}
            {/* Show loading indicator separately after user message */}
            {isLoading && (
              <MessageBubble
                key="loading-indicator"
                message={{ id: 'loading', type: 'assistant', content: '', timestamp: new Date().toISOString() }}
                isLoading={true}
              />
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <ChatInput
            onSendMessage={handleSendMessage}
            disabled={isLoading}
            placeholder="Ask about Tesla's financial modeling or market intelligence..."
          />
        </ChatWindow>
      </div>
    </div>
  );
};

export default TeslaFAChat;
