import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const FloatingChatButton = () => {
  const [isHovered, setIsHovered] = useState(false);
  const navigate = useNavigate();

  const handleClick = () => {
    navigate('/tesla-fa-chat');
  };

  return (
    <div className="fixed bottom-6 right-6 z-50">
      <button
        onClick={handleClick}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        className={`
          group relative flex items-center justify-center w-16 h-16 rounded-full shadow-lg
          transition-all duration-300 ease-in-out transform hover:scale-110
          bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700
          ${isHovered ? 'shadow-2xl' : 'shadow-lg'}
        `}
      >
        {/* Chat Icon */}
        <svg
          className="w-8 h-8 text-white transition-transform duration-300 group-hover:scale-110"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
          />
        </svg>

        {/* Tooltip */}
        <div className={`
          absolute right-full mr-4 px-3 py-2 bg-gray-900 text-white text-sm rounded-lg
          transition-opacity duration-300 whitespace-nowrap
          ${isHovered ? 'opacity-100' : 'opacity-0 pointer-events-none'}
        `}>
          Ask Tesla FA
          <div className="absolute left-full top-1/2 transform -translate-y-1/2 w-0 h-0 border-l-4 border-l-gray-900 border-t-4 border-t-transparent border-b-4 border-b-transparent"></div>
        </div>

        {/* Pulse Animation */}
        <div className="absolute inset-0 rounded-full bg-blue-400 animate-ping opacity-20"></div>
      </button>
    </div>
  );
};

export default FloatingChatButton;
