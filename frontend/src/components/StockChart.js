import React, { useRef, useEffect, useState } from 'react';

const StockChart = ({ data, width = 400, height = 200, title = "Stock Chart" }) => {
  const svgRef = useRef(null);
  const [tooltip, setTooltip] = useState({ show: false, x: 0, y: 0, date: '', price: 0 });

  useEffect(() => {
    if (!data || data.length === 0) return;

    const svg = svgRef.current;
    const margin = { top: 20, right: 30, bottom: 40, left: 40 };
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
    
    const minDate = Math.min(...dates);
    const maxDate = Math.max(...dates);
    const minPrice = Math.min(...prices);
    const maxPrice = Math.max(...prices);

    // Scales
    const xScale = (date) => ((date - minDate) / (maxDate - minDate)) * chartWidth;
    const yScale = (price) => chartHeight - ((price - minPrice) / (maxPrice - minPrice)) * chartHeight;

    // Create gradient for line
    const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    const gradient = document.createElementNS('http://www.w3.org/2000/svg', 'linearGradient');
    gradient.setAttribute('id', 'stockGradient');
    gradient.setAttribute('x1', '0%');
    gradient.setAttribute('y1', '0%');
    gradient.setAttribute('x2', '0%');
    gradient.setAttribute('y2', '100%');
    
    const stop1 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop1.setAttribute('offset', '0%');
    stop1.setAttribute('stop-color', '#3B82F6');
    stop1.setAttribute('stop-opacity', '0.3');
    
    const stop2 = document.createElementNS('http://www.w3.org/2000/svg', 'stop');
    stop2.setAttribute('offset', '100%');
    stop2.setAttribute('stop-color', '#3B82F6');
    stop2.setAttribute('stop-opacity', '0');
    
    gradient.appendChild(stop1);
    gradient.appendChild(stop2);
    defs.appendChild(gradient);
    svg.appendChild(defs);

    // Create area path
    let areaPath = `M ${xScale(dates[0])} ${chartHeight}`;
    for (let i = 0; i < dates.length; i++) {
      areaPath += ` L ${xScale(dates[i])} ${yScale(prices[i])}`;
    }
    areaPath += ` L ${xScale(dates[dates.length - 1])} ${chartHeight} Z`;

    const area = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    area.setAttribute('d', areaPath);
    area.setAttribute('fill', 'url(#stockGradient)');
    g.appendChild(area);

    // Create line path
    let linePath = `M ${xScale(dates[0])} ${yScale(prices[0])}`;
    for (let i = 1; i < dates.length; i++) {
      linePath += ` L ${xScale(dates[i])} ${yScale(prices[i])}`;
    }

    const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    line.setAttribute('d', linePath);
    line.setAttribute('fill', 'none');
    line.setAttribute('stroke', '#3B82F6');
    line.setAttribute('stroke-width', '2');
    line.setAttribute('stroke-linecap', 'round');
    line.setAttribute('stroke-linejoin', 'round');
    g.appendChild(line);

    // Add invisible hover areas for better interaction
    data.forEach((point, index) => {
      const hoverArea = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      const cx = xScale(dates[index]);
      const cy = yScale(prices[index]);
      
      hoverArea.setAttribute('x', cx - 10);
      hoverArea.setAttribute('y', 0);
      hoverArea.setAttribute('width', '20');
      hoverArea.setAttribute('height', chartHeight);
      hoverArea.setAttribute('fill', 'transparent');
      hoverArea.setAttribute('cursor', 'pointer');
      
      // Add hover effect with tooltip
      hoverArea.addEventListener('mouseenter', (e) => {
        const svgRect = svg.getBoundingClientRect();
        setTooltip({
          show: true,
          x: e.clientX - svgRect.left,
          y: e.clientY - svgRect.top,
          date: dates[index].toLocaleDateString('en-US', { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric' 
          }),
          price: prices[index].toFixed(2)
        });
      });
      
      hoverArea.addEventListener('mousemove', (e) => {
        const svgRect = svg.getBoundingClientRect();
        setTooltip(prev => ({
          ...prev,
          x: e.clientX - svgRect.left,
          y: e.clientY - svgRect.top
        }));
      });
      
      hoverArea.addEventListener('mouseleave', () => {
        setTooltip({ show: false, x: 0, y: 0, date: '', price: 0 });
      });
      
      g.appendChild(hoverArea);
      
      // Add visible circle on top
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      circle.setAttribute('cx', cx);
      circle.setAttribute('cy', cy);
      circle.setAttribute('r', '2');
      circle.setAttribute('fill', '#3B82F6');
      circle.setAttribute('stroke', '#fff');
      circle.setAttribute('stroke-width', '1');
      circle.setAttribute('pointer-events', 'none');
      
      g.appendChild(circle);
    });

    // Add grid lines
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
    
    // Vertical grid lines
    for (let i = 0; i <= 5; i++) {
      const x = (chartWidth / 5) * i;
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', x);
      line.setAttribute('y1', '0');
      line.setAttribute('x2', x);
      line.setAttribute('y2', chartHeight);
      line.setAttribute('stroke', '#6B7280');
      line.setAttribute('stroke-width', '1');
      gridGroup.appendChild(line);
    }
    
    g.insertBefore(gridGroup, g.firstChild);

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

    // Add axis labels
    const labelsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    labelsGroup.setAttribute('class', 'labels');
    
    // X-axis labels
    for (let i = 0; i <= 4; i++) {
      const x = (chartWidth / 4) * i;
      const dateIndex = Math.floor((dates.length - 1) * (i / 4));
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', x);
      text.setAttribute('y', chartHeight + 20);
      text.setAttribute('text-anchor', 'middle');
      text.setAttribute('font-size', '10');
      text.setAttribute('fill', '#6B7280');
      text.textContent = dates[dateIndex].toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
      labelsGroup.appendChild(text);
    }
    
    // Y-axis labels
    for (let i = 0; i <= 4; i++) {
      const y = (chartHeight / 4) * i;
      const price = minPrice + (maxPrice - minPrice) * (1 - i / 4);
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      text.setAttribute('x', -10);
      text.setAttribute('y', y + 4);
      text.setAttribute('text-anchor', 'end');
      text.setAttribute('font-size', '10');
      text.setAttribute('fill', '#6B7280');
      text.textContent = `$${price.toFixed(0)}`;
      labelsGroup.appendChild(text);
    }
    
    g.appendChild(labelsGroup);

  }, [data, width, height]);

  return (
    <div className="stock-chart relative">
      {title && <h4 className="text-sm font-medium text-gray-700 mb-2">{title}</h4>}
      <svg ref={svgRef} width={width} height={height} className="border border-gray-200 rounded">
        {/* Chart will be rendered here */}
      </svg>
      
      {/* Tooltip */}
      {tooltip.show && (
        <div
          className="absolute bg-gray-900 text-white px-3 py-2 rounded shadow-lg text-xs pointer-events-none z-10"
          style={{
            left: `${tooltip.x + 10}px`,
            top: `${tooltip.y - 40}px`,
            transform: 'translateX(-50%)'
          }}
        >
          <div className="font-semibold">${tooltip.price}</div>
          <div className="text-gray-300">{tooltip.date}</div>
        </div>
      )}
    </div>
  );
};

export default StockChart;
