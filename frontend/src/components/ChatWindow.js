import React from 'react';

const ChatWindow = ({ 
  children, 
  isMinimized, 
  isMaximized, 
  onMinimize, 
  onMaximize, 
  onClose 
}) => {
  const getWindowClasses = () => {
    let baseClasses = "bg-white rounded-lg shadow-lg border transition-all duration-300 ease-in-out";
    
    if (isMinimized) {
      return `${baseClasses} h-16 overflow-hidden`;
    }
    
    if (isMaximized) {
      return `${baseClasses} fixed inset-4 z-50 h-[calc(100vh-2rem)]`;
    }
    
    // Use more vertical space automatically (85% of viewport or minimum 700px)
    return `${baseClasses} h-[calc(85vh)] min-h-[700px]`;
  };

  const getContentClasses = () => {
    if (isMinimized) {
      return "hidden";
    }
    
    return "flex flex-col h-full";
  };

  return (
    <div className={getWindowClasses()}>
      {/* Window Header */}
      <div className="flex items-center justify-between p-4 border-b bg-gray-50 rounded-t-lg">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
          <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={onMinimize}
            className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded transition-colors"
            title={isMinimized ? "Restore" : "Minimize"}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
            </svg>
          </button>
          <button
            onClick={onMaximize}
            className="p-1 text-gray-400 hover:text-gray-600 hover:bg-gray-200 rounded transition-colors"
            title={isMaximized ? "Restore" : "Maximize"}
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={isMaximized ? "M9 9V4.5M9 9H4.5M9 9L3.5 3.5M15 9h4.5M15 9V4.5M15 9l5.5-5.5" : "M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"} />
            </svg>
          </button>
          <button
            onClick={onClose}
            className="p-1 text-gray-400 hover:text-red-600 hover:bg-red-100 rounded transition-colors"
            title="Close"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* Window Content */}
      <div className={getContentClasses()}>
        {children}
      </div>

      {/* Minimized State Content */}
      {isMinimized && (
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full flex items-center justify-center">
              <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
              </svg>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900">Tesla Financial Assistant</p>
              <p className="text-xs text-gray-500">Click to restore chat</p>
            </div>
          </div>
          <button
            onClick={onMinimize}
            className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
        </div>
      )}
    </div>
  );
};

export default ChatWindow;
