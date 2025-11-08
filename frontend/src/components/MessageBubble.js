import React from 'react';

const MessageBubble = ({ message, isLoading = false }) => {
  const isUser = message.type === 'user';
  const isError = message.isError;

  const formatTime = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const formatContent = (content) => {
    // Enhanced formatting for better readability
    let formatted = content
      // Bold text
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-gray-900">$1</strong>')
      // Italic text
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      // Headers (### )
      .replace(/###\s+(.*?)(<br \/>|$)/g, '<h3 class="text-lg font-bold text-gray-900 mt-4 mb-2">$1</h3>')
      // Bullet points
      .replace(/^•\s+(.+)$/gm, '<div class="flex items-start"><span class="text-blue-600 mr-2">•</span><span>$1</span></div>')
      // Line breaks
      .replace(/\n/g, '<br />');
    
    return formatted;
  };

  if (isLoading) {
    return (
      <div className="flex justify-start">
        <div className="flex items-start space-x-4 max-w-2xl lg:max-w-4xl">
          <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>
          <div className="bg-white border border-gray-200 rounded-xl px-5 py-4 shadow-sm">
            <div className="flex items-center space-x-2">
              <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce"></div>
              <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
              <div className="w-2.5 h-2.5 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
              <span className="ml-2 text-sm text-gray-600 font-medium">Analyzing...</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex items-start space-x-4 max-w-2xl lg:max-w-4xl ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
        {/* Avatar */}
        {!isUser && (
          <div className="w-10 h-10 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>
        )}

        {/* Message Content */}
        <div className={`rounded-xl px-5 py-4 shadow-sm ${
          isUser 
            ? 'bg-blue-600 text-white' 
            : isError 
              ? 'bg-red-50 text-red-800 border border-red-200' 
              : 'bg-white text-gray-900 border border-gray-200'
        }`}>
          <div 
            className="text-base"
            style={{ 
              fontSize: '15px',
              lineHeight: '1.2'
            }}
            dangerouslySetInnerHTML={{ 
              __html: formatContent(message.content) 
            }}
          />
          {message.metadata && (
            <div className="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">
              <span className="font-medium">Agents:</span> {message.metadata.agents_used?.join(', ') || 'N/A'}
            </div>
          )}
          <div className={`text-xs mt-2 ${
            isUser ? 'text-blue-100' : isError ? 'text-red-600' : 'text-gray-500'
          }`}>
            {formatTime(message.timestamp)}
          </div>
        </div>

        {/* User Avatar */}
        {isUser && (
          <div className="w-10 h-10 bg-gray-300 rounded-full flex items-center justify-center flex-shrink-0">
            <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
            </svg>
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;
