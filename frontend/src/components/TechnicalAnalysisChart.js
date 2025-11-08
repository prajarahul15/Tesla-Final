import React, { useRef, useEffect } from 'react';

const TechnicalAnalysisChart = ({ 
  data, 
  indicators = ['sma20', 'sma50', 'rsi'], 
  width = 400, 
  height = 300,
  title = "Technical Analysis"
}) => {
  const svgRef = useRef(null);

  useEffect(() => {
    if (!data || data.length === 0) return;

    const svg = svgRef.current;
    const margin = { top: 20, right: 30, bottom: 60, left: 50 };
    const chartWidth = width - margin.left - margin.right;
    const chartHeight = height - margin.top - margin.bottom;

    // Clear previous content
    svg.innerHTML = '';

    // Create main group
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('transform', `translate(${margin.left},${margin.top})`);
    svg.appendChild(g);

    // Prepare data
    const dates = data.map(d => new Date(d.date));
    const prices = data.map(d => d.price);
    const volumes = data.map(d => d.volume || 0);
    
    const minDate = Math.min(...dates);
    const maxDate = Math.max(...dates);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);
    const maxVolume = Math.max(...volumes);

    // Calculate technical indicators
    const sma20 = calculateSMA(prices, 20);
    const sma50 = calculateSMA(prices, 50);
    const rsi = calculateRSI(prices, 14);

    // Scales
    const xScale = (date) => ((date - minDate) / (maxDate - minDate)) * chartWidth;
    const yScale = (price) => chartHeight - ((price - minPrice) / (maxPrice - minPrice)) * chartHeight;
    const volumeScale = (volume) => (volume / maxVolume) * (chartHeight * 0.3);

    // Create grid
    const gridGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    gridGroup.setAttribute('class', 'grid');
    gridGroup.setAttribute('opacity', '0.2');
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = (chartHeight / 5) * i;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', '0');
      line.setAttribute('y1', y);
      line.setAttribute('x2', chartWidth);
      line.setAttribute('y2', y);
      line.setAttribute('stroke', '#6B7280');
      line.setAttribute('stroke-width', '1');
      gridGroup.appendChild(line);
    }
    
    g.appendChild(gridGroup);

    // Plot volume bars
    data.forEach((point, index) => {
      if (point.volume > 0) {
        const bar = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        bar.setAttribute('x', xScale(dates[index]) - 2);
        bar.setAttribute('y', chartHeight - volumeScale(volumes[index]));
        bar.setAttribute('width', '4');
        bar.setAttribute('height', volumeScale(volumes[index]));
        bar.setAttribute('fill', prices[index] >= (index > 0 ? prices[index - 1] : prices[index]) ? '#10B981' : '#EF4444');
        bar.setAttribute('opacity', '0.3');
        g.appendChild(bar);
      }
    });

    // Plot price line
    let pricePath = `M ${xScale(dates[0])} ${yScale(prices[0])}`;
    for (let i = 1; i < dates.length; i++) {
      pricePath += ` L ${xScale(dates[i])} ${yScale(prices[i])}`;
    }

    const priceLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    priceLine.setAttribute('d', pricePath);
    priceLine.setAttribute('fill', 'none');
    priceLine.setAttribute('stroke', '#3B82F6');
    priceLine.setAttribute('stroke-width', '2');
    g.appendChild(priceLine);

    // Plot SMA lines
    if (indicators.includes('sma20')) {
      let sma20Path = '';
      for (let i = 19; i < dates.length; i++) {
        if (sma20[i]) {
          const path = `M ${xScale(dates[i])} ${yScale(sma20[i])}`;
          if (i === 19) sma20Path = path;
          else sma20Path += ` L ${xScale(dates[i])} ${yScale(sma20[i])}`;
        }
      }
      
      const sma20Line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      sma20Line.setAttribute('d', sma20Path);
      sma20Line.setAttribute('fill', 'none');
      sma20Line.setAttribute('stroke', '#F59E0B');
      sma20Line.setAttribute('stroke-width', '1.5');
      sma20Line.setAttribute('stroke-dasharray', '5,5');
      g.appendChild(sma20Line);
    }

    if (indicators.includes('sma50')) {
      let sma50Path = '';
      for (let i = 49; i < dates.length; i++) {
        if (sma50[i]) {
          const path = `M ${xScale(dates[i])} ${yScale(sma50[i])}`;
          if (i === 49) sma50Path = path;
          else sma50Path += ` L ${xScale(dates[i])} ${yScale(sma50[i])}`;
        }
      }
      
      const sma50Line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      sma50Line.setAttribute('d', sma50Path);
      sma50Line.setAttribute('fill', 'none');
      sma50Line.setAttribute('stroke', '#EF4444');
      sma50Line.setAttribute('stroke-width', '1.5');
      sma50Line.setAttribute('stroke-dasharray', '5,5');
      g.appendChild(sma50Line);
    }

    // Add legend
    const legendGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    legendGroup.setAttribute('class', 'legend');
    
    const legendItems = [
      { label: 'Price', color: '#3B82F6', type: 'line' },
      { label: 'SMA 20', color: '#F59E0B', type: 'dashed' },
      { label: 'SMA 50', color: '#EF4444', type: 'dashed' }
    ];

    legendItems.forEach((item, index) => {
      const legendItem = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      legendItem.setAttribute('transform', `translate(${chartWidth - 150}, ${20 + index * 20})`);
      
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', '0');
      line.setAttribute('y1', '0');
      line.setAttribute('x2', '15');
      line.setAttribute('y2', '0');
      line.setAttribute('stroke', item.color);
      line.setAttribute('stroke-width', '2');
      if (item.type === 'dashed') {
        line.setAttribute('stroke-dasharray', '5,5');
      }
      legendItem.appendChild(line);
      
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', '20');
      text.setAttribute('y', '4');
      text.setAttribute('font-size', '12');
      text.setAttribute('fill', '#374151');
      text.textContent = item.label;
      legendItem.appendChild(text);
      
      legendGroup.appendChild(legendItem);
    });
    
    g.appendChild(legendGroup);

    // Add RSI subplot if requested
    if (indicators.includes('rsi')) {
      const rsiGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      rsiGroup.setAttribute('transform', `translate(0, ${chartHeight + 40})`);
      
      // RSI background
      const rsiBg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rsiBg.setAttribute('x', '0');
      rsiBg.setAttribute('y', '0');
      rsiBg.setAttribute('width', chartWidth);
      rsiBg.setAttribute('height', '60');
      rsiBg.setAttribute('fill', '#F9FAFB');
      rsiBg.setAttribute('stroke', '#E5E7EB');
      rsiGroup.appendChild(rsiBg);
      
      // RSI line
      let rsiPath = '';
      for (let i = 14; i < dates.length; i++) {
        if (rsi[i]) {
          const x = xScale(dates[i]);
          const y = 60 - (rsi[i] / 100) * 60;
          const path = `M ${x} ${y}`;
          if (i === 14) rsiPath = path;
          else rsiPath += ` L ${x} ${y}`;
        }
      }
      
      const rsiLine = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      rsiLine.setAttribute('d', rsiPath);
      rsiLine.setAttribute('fill', 'none');
      rsiLine.setAttribute('stroke', '#8B5CF6');
      rsiLine.setAttribute('stroke-width', '1.5');
      rsiGroup.appendChild(rsiLine);
      
      // RSI levels
      const rsi30Line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      rsi30Line.setAttribute('x1', '0');
      rsi30Line.setAttribute('y1', '42'); // 70% of 60
      rsi30Line.setAttribute('x2', chartWidth);
      rsi30Line.setAttribute('y2', '42');
      rsi30Line.setAttribute('stroke', '#10B981');
      rsi30Line.setAttribute('stroke-width', '1');
      rsi30Line.setAttribute('stroke-dasharray', '2,2');
      rsiGroup.appendChild(rsi30Line);
      
      const rsi70Line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      rsi70Line.setAttribute('x1', '0');
      rsi70Line.setAttribute('y1', '18'); // 30% of 60
      rsi70Line.setAttribute('x2', chartWidth);
      rsi70Line.setAttribute('y2', '18');
      rsi70Line.setAttribute('stroke', '#EF4444');
      rsi70Line.setAttribute('stroke-width', '1');
      rsi70Line.setAttribute('stroke-dasharray', '2,2');
      rsiGroup.appendChild(rsi70Line);
      
      // RSI label
      const rsiLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      rsiLabel.setAttribute('x', '5');
      rsiLabel.setAttribute('y', '15');
      rsiLabel.setAttribute('font-size', '10');
      rsiLabel.setAttribute('fill', '#6B7280');
      rsiLabel.textContent = 'RSI (14)';
      rsiGroup.appendChild(rsiLabel);
      
      g.appendChild(rsiGroup);
    }

    // Add axes
    const axisGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    axisGroup.setAttribute('class', 'axes');
    
    // X-axis
    const xAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    xAxis.setAttribute('x1', '0');
    xAxis.setAttribute('y1', chartHeight);
    xAxis.setAttribute('x2', chartWidth);
    xAxis.setAttribute('y2', chartHeight);
    xAxis.setAttribute('stroke', '#374151');
    xAxis.setAttribute('stroke-width', '1');
    axisGroup.appendChild(xAxis);
    
    // Y-axis
    const yAxis = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    yAxis.setAttribute('x1', '0');
    yAxis.setAttribute('y1', '0');
    yAxis.setAttribute('x2', '0');
    yAxis.setAttribute('y2', chartHeight);
    yAxis.setAttribute('stroke', '#374151');
    yAxis.setAttribute('stroke-width', '1');
    axisGroup.appendChild(yAxis);
    
    g.appendChild(axisGroup);

  }, [data, indicators, width, height]);

  // Helper functions for technical indicators
  function calculateSMA(prices, period) {
    const sma = [];
    for (let i = 0; i < prices.length; i++) {
      if (i < period - 1) {
        sma.push(null);
      } else {
        const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
        sma.push(sum / period);
      }
    }
    return sma;
  }

  function calculateRSI(prices, period) {
    const rsi = [];
    const gains = [];
    const losses = [];
    
    // Calculate price changes
    for (let i = 1; i < prices.length; i++) {
      const change = prices[i] - prices[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }
    
    // Calculate RSI
    for (let i = 0; i < prices.length; i++) {
      if (i < period) {
        rsi.push(null);
      } else {
        const avgGain = gains.slice(i - period, i).reduce((a, b) => a + b, 0) / period;
        const avgLoss = losses.slice(i - period, i).reduce((a, b) => a + b, 0) / period;
        
        if (avgLoss === 0) {
          rsi.push(100);
        } else {
          const rs = avgGain / avgLoss;
          rsi.push(100 - (100 / (1 + rs)));
        }
      }
    }
    
    return rsi;
  }

  return (
    <div className="technical-analysis-chart">
      <h4 className="text-sm font-medium text-gray-700 mb-2">{title}</h4>
      <svg ref={svgRef} width={width} height={indicators.includes('rsi') ? height + 80 : height} className="border border-gray-200 rounded">
        {/* Chart will be rendered here */}
      </svg>
    </div>
  );
};

export default TechnicalAnalysisChart;
