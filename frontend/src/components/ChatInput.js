import React, { useState, useRef, useEffect } from 'react';

const ChatInput = ({ onSendMessage, disabled = false, placeholder = "Type your message..." }) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (message.trim() && !disabled) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  return (
    <div className="border-t bg-white p-5">
      <form onSubmit={handleSubmit} className="flex items-end space-x-4">
        <div className="flex-1">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={placeholder}
            disabled={disabled}
            rows={1}
            className={`
              w-full px-5 py-4 border-2 border-gray-300 rounded-xl resize-none
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500
              disabled:opacity-50 disabled:cursor-not-allowed
              transition-all duration-200
              text-base
              ${disabled ? 'bg-gray-50' : 'bg-white'}
            `}
            style={{ 
              minHeight: '56px', 
              maxHeight: '140px',
              fontSize: '15px',
              lineHeight: '1.6'
            }}
          />
        </div>
        <button
          type="submit"
          disabled={!message.trim() || disabled}
          className={`
            px-7 py-4 rounded-xl font-semibold transition-all duration-200
            flex items-center space-x-2 shadow-sm
            text-base
            ${disabled || !message.trim()
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800 hover:shadow-md'
            }
          `}
        >
          {disabled ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-gray-500"></div>
              <span>Sending...</span>
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
              <span>Send</span>
            </>
          )}
        </button>
      </form>
      
      {/* Quick Actions */}
      <div className="mt-4 flex flex-wrap gap-2">
        <span className="text-sm font-medium text-gray-600 mr-2">Quick questions:</span>
        <button
          type="button"
          onClick={() => setMessage("What is Tesla's revenue forecast for 2025?")}
          disabled={disabled}
          className="text-sm px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 hover:shadow-sm transition-all disabled:opacity-50 font-medium"
        >
          📊 Revenue Forecast
        </button>
        <button
          type="button"
          onClick={() => setMessage("Explain the cost structure and margins")}
          disabled={disabled}
          className="text-sm px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 hover:shadow-sm transition-all disabled:opacity-50 font-medium"
        >
          💰 Cost Analysis
        </button>
        <button
          type="button"
          onClick={() => setMessage("Simulate 25% revenue growth and show impact")}
          disabled={disabled}
          className="text-sm px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 hover:shadow-sm transition-all disabled:opacity-50 font-medium"
        >
          📈 Growth Simulation
        </button>
        <button
          type="button"
          onClick={() => setMessage("What are Tesla's key financial metrics for 2025?")}
          disabled={disabled}
          className="text-sm px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 hover:shadow-sm transition-all disabled:opacity-50 font-medium"
        >
          🎯 Key Metrics
        </button>
      </div>
    </div>
  );
};

export default ChatInput;
