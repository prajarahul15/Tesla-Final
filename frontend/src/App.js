import React, { useState, useEffect } from 'react';
import './App.css';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import axios from 'axios';
import TeslaDashboard from './components/TeslaDashboard';
import TeslaFAChat from './components/TeslaFAChat';
import MarketInsightsPage from './components/MarketInsightsPage';
import TeslaFontDemo from './components/TeslaFontDemo';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  return (
    <div className="App font-tesla">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<TeslaDashboard />} />
          <Route path="/tesla-fa-chat" element={<TeslaFAChat />} />
          <Route path="/market-insights" element={<MarketInsightsPage />} />
          <Route path="/font-demo" element={<TeslaFontDemo />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

export default App;