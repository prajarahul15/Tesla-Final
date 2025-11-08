import React, { useEffect, useState, useMemo, useRef } from 'react';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function LineChart({ series, xLabels = [], height=220, padding=32 }) {
  const containerRef = useRef(null);
  const [w, setW] = useState(600);
  const [hoverX, setHoverX] = useState(null);
  const [tooltip, setTooltip] = useState(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => setW(el.clientWidth || 600);
    update();
    if (typeof ResizeObserver !== 'undefined') {
      const ro = new ResizeObserver(update);
      ro.observe(el);
      window.addEventListener('resize', update);
      return () => { try { ro.disconnect(); } catch {} window.removeEventListener('resize', update); };
    } else {
      window.addEventListener('resize', update);
      return () => window.removeEventListener('resize', update);
    }
  }, []);

  const width = Math.max(300, w);
  const allPoints = series.flatMap(s => s.points);
  if (allPoints.length === 0) return <div ref={containerRef} className="w-full" />;
  const xs = allPoints.map(p => p.x);
  const ys = allPoints.map(p => p.y);
  const uniqueXs = Array.from(new Set(xs)).sort((a,b)=>a-b);
  const minX = Math.min(...xs, 0), maxX = Math.max(...xs, 1);
  const minY = Math.min(0, ...ys), maxY = Math.max(...ys, 1);
  const sx = (x) => padding + (width - 2*padding) * ((x - minX) / (maxX - minX || 1));
  const sy = (y) => height - padding - (height - 2*padding) * ((y - minY) / (maxY - minY || 1));
  const path = (pts) => pts.map((p,i) => `${i?'L':'M'}${sx(p.x)},${sy(p.y)}`).join(' ');

  const onMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const xPx = e.clientX - rect.left;
    const xVal = minX + ((xPx - padding) / (width - 2*padding)) * (maxX - minX);
    let nearest = uniqueXs[0];
    let best = Infinity;
    for (const x of uniqueXs) {
      const d = Math.abs(x - xVal);
      if (d < best) { best = d; nearest = x; }
    }
    setHoverX(nearest);
    const items = [];
    series.forEach(s => {
      const p = s.points.find(pt => pt.x === nearest);
      if (p) items.push({ label: s.label, color: s.color, value: p.y });
    });
    const dateLabel = xLabels[nearest] || '';
    setTooltip({
      x: sx(nearest),
      y: padding,
      items,
      dateLabel
    });
  };

  const onLeave = () => { setHoverX(null); setTooltip(null); };

  return (
    <div ref={containerRef} className="w-full relative">
      <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet"
        onMouseMove={onMove} onMouseLeave={onLeave}>
        <rect x={0} y={0} width={width} height={height} fill="#ffffff" stroke="#e5e7eb" />
        {[0.25,0.5,0.75].map((g, i) => (
          <line key={i} x1={padding} x2={width-padding} y1={padding + (height-2*padding)*g} y2={padding + (height-2*padding)*g} stroke="#f3f4f6" />
        ))}
        {series.map((s, idx) => (
          <path key={idx} d={path(s.points)} fill="none" stroke={s.color} strokeWidth={2} />
        ))}
        {/* Hover elements */}
        {hoverX !== null && (
          <g>
            <line x1={sx(hoverX)} x2={sx(hoverX)} y1={padding} y2={height-padding} stroke="#9ca3af" strokeDasharray="4 4" />
            {series.map((s, i) => {
              const p = s.points.find(pt => pt.x === hoverX);
              if (!p) return null;
              return <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3} fill={s.color} stroke="#ffffff" strokeWidth={1}/>;
            })}
          </g>
        )}
        {/* Legend */}
        <g transform={`translate(${padding},${padding/1.5})`}>
          {series.map((s, i) => (
            <g key={i} transform={`translate(${i*160},0)`}>
              <rect width={14} height={3} y={5} fill={s.color} />
              <text x={20} y={10} fontSize={12} fill="#374151">{s.label}</text>
            </g>
          ))}
        </g>
      </svg>
      {tooltip && (
        <div style={{ position:'absolute', left: Math.min(Math.max(tooltip.x + 12, 8), width - 220), top: 12 }} className="bg-white shadow-lg border rounded px-3 py-2 text-xs">
          <div className="font-semibold text-gray-800 mb-1">{tooltip.dateLabel}</div>
          {tooltip.items.map((it, i) => (
            <div key={i} className="flex items-center gap-2">
              <span style={{background:it.color, width:8, height:3, display:'inline-block'}}></span>
              <span className="text-gray-600">{it.label}:</span>
              <span className="font-semibold text-gray-900">{Math.round(it.value).toLocaleString()}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

const VehicleForecast = () => {
  const [models, setModels] = useState([]);
  const [modelKey, setModelKey] = useState('');
  const [forecastType, setForecastType] = useState('univariate');
  const [monthsAhead, setMonthsAhead] = useState(12);
  const [includeRevenue, setIncludeRevenue] = useState(false);
  const [aspPerUnit, setAspPerUnit] = useState('');
  const [priceChange, setPriceChange] = useState('');
  const [elasticity, setElasticity] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const CHART_TABLE_HEIGHT = 275;

  useEffect(() => {
    const loadModels = async () => {
      try {
        const res = await axios.get(`${API}/vehicles/models`);
        const list = res.data?.models || [];
        setModels(list);
        if (list.length > 0) setModelKey(list[0].model_key);
      } catch (e) {
        setError('Failed to load vehicle models');
      }
    };
    loadModels();
  }, []);

  const runForecast = async () => {
    if (!modelKey) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const payload = {
        model_key: modelKey,
        forecast_type: forecastType,
        months_ahead: monthsAhead,
        include_revenue: includeRevenue,
        test_window: 6
      };
      if (includeRevenue && aspPerUnit) payload.asp_per_unit = parseFloat(aspPerUnit);
      if (priceChange !== '' && elasticity !== '') {
        payload.price_change = parseFloat(priceChange);
        payload.elasticity = parseFloat(elasticity);
      }
      const res = await axios.post(`${API}/vehicles/forecast`, payload);
      setResult(res.data?.forecast || null);
    } catch (e) {
      setError(e?.response?.data?.detail || 'Forecast failed');
    } finally {
      setLoading(false);
    }
  };

  const chartSeries = useMemo(() => {
    if (!result) return [];
    const pointsHistory = (result.history?.dates || []).map((d, i) => ({ x: i, y: result.history.actuals[i] || 0 }));
    const hLen = pointsHistory.length;
    const testLen = result.test_evaluation?.dates?.length || 0;
    // Do NOT include separate Test Actual series; show History (Actual) + Test (Pred) only
    const pointsTestPred = (result.test_evaluation?.predictions || []).map((v, i) => ({ x: hLen - testLen + i, y: v || 0 }));
    const pointsForecast = (result.forecasts || []).map((f, i) => ({ x: hLen + i, y: f.forecast || 0 }));
    return [
      { label: 'History (Actual)', color: '#2563eb', points: pointsHistory },
      { label: 'Test (Pred)', color: '#f59e0b', points: pointsTestPred },
      { label: 'Forecast', color: '#ef4444', points: pointsForecast },
    ];
  }, [result]);

  const xLabels = useMemo(() => {
    if (!result) return [];
    const labels = [];
    const hDates = result.history?.dates || [];
    hDates.forEach((d, i) => labels[i] = d);
    const testDates = result.test_evaluation?.dates || [];
    const hLen = hDates.length;
    const tLen = testDates.length;
    testDates.forEach((d, i) => labels[hLen - tLen + i] = d);
    const fcs = result.forecasts || [];
    fcs.forEach((f, i) => labels[hLen + i] = f.date);
    return labels;
  }, [result]);

  return (
    <div className="space-y-6">
      <div className="bg-gradient-to-r from-indigo-600 to-blue-600 text-white p-6 rounded-lg shadow">
        <h2 className="text-2xl font-bold">Vehicle Forecast</h2>
        <p className="text-indigo-100">Monthly deliveries forecast per model with optional revenue and elasticity</p>
      </div>

      <div className="bg-white rounded-lg shadow-sm border p-6 space-y-4">
        {/* Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
            <select value={modelKey} onChange={e => setModelKey(e.target.value)} className="w-full p-2 border rounded">
              {models.map(m => (
                <option key={m.model_key} value={m.model_key}>{m.display_name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Method</label>
            <select value={forecastType} onChange={e => setForecastType(e.target.value)} className="w-full p-2 border rounded">
              <option value="univariate">Univariate</option>
              <option value="multivariate">Multivariate</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Horizon</label>
            <select value={monthsAhead} onChange={e => setMonthsAhead(parseInt(e.target.value))} className="w-full p-2 border rounded">
              <option value={6}>6 months</option>
              <option value={12}>12 months</option>
              <option value={18}>18 months</option>
              <option value={24}>24 months</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 pt-2">
          <div className="flex items-center gap-2">
            <input id="rev" type="checkbox" checked={includeRevenue} onChange={e => setIncludeRevenue(e.target.checked)} />
            <label htmlFor="rev" className="text-sm text-gray-700">Include Revenue</label>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">ASP (optional)</label>
            <input type="number" value={aspPerUnit} onChange={e => setAspPerUnit(e.target.value)} className="w-full p-2 border rounded" placeholder="e.g. 52000" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Price change (e.g. -0.05)</label>
            <input type="number" step="0.01" value={priceChange} onChange={e => setPriceChange(e.target.value)} className="w-full p-2 border rounded" placeholder="-0.05" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Elasticity (e.g. -1.0)</label>
            <input type="number" step="0.1" value={elasticity} onChange={e => setElasticity(e.target.value)} className="w-full p-2 border rounded" placeholder="-1.0" />
          </div>
        </div>

        <div className="pt-2">
          <button onClick={runForecast} disabled={loading || !modelKey} className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded">
            {loading ? 'Running…' : 'Run Forecast'}
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded">{error}</div>
      )}

      {result && (
        <div className="bg-white rounded-lg shadow-sm border p-6 space-y-6">
          <h3 className="text-lg font-semibold text-gray-900">Model Performance</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-blue-50 p-3 rounded">
              <div className="text-sm text-blue-800">MAE (test)</div>
              <div className="text-2xl font-bold text-blue-700">{(result.model_metrics?.mae || 0).toFixed(0)}</div>
            </div>
            <div className="bg-green-50 p-3 rounded">
              <div className="text-sm text-green-800">MAPE (test)</div>
              <div className="text-2xl font-bold text-green-700">{((result.model_metrics?.mape || 0)*100).toFixed(1)}%</div>
            </div>
            <div className="bg-purple-50 p-3 rounded">
              <div className="text-sm text-purple-800">Total Forecast</div>
              <div className="text-2xl font-bold text-purple-700">{Math.round(result.forecasts.reduce((s,f)=>s+f.forecast,0)).toLocaleString()}</div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-10 gap-6">
            <div className="md:col-span-7">
              <LineChart series={chartSeries} xLabels={xLabels} height={CHART_TABLE_HEIGHT} />
            </div>
            <div className="md:col-span-3">
              <div className="overflow-x-auto">
                <div className={`border rounded`} style={{height: CHART_TABLE_HEIGHT, overflowY: 'hidden', overflowX: 'auto'}}>
                  <table className="min-w-[720px] divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider whitespace-nowrap">Date</th>
                        <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Actual</th>
                        <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Test Prediction</th>
                        <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Accuracy (MAPE)</th>
                      </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                      {(result.test_evaluation?.dates || []).map((d, i) => {
                        const actual = result.test_evaluation?.actuals?.[i] ?? null;
                        const pred = result.test_evaluation?.predictions?.[i] ?? null;
                        const acc = (actual && pred) ? (Math.abs((actual - pred) / (actual || 1)) * 100) : null;
                        return (
                          <tr key={i}>
                            <td className="px-4 py-2 text-sm text-gray-900 whitespace-nowrap">{d}</td>
                            <td className="px-4 py-2 text-sm text-right text-gray-700">{actual !== null ? Math.round(actual).toLocaleString() : '-'}</td>
                            <td className="px-4 py-2 text-sm text-right text-gray-700">{pred !== null ? Math.round(pred).toLocaleString() : '-'}</td>
                            <td className={`px-4 py-2 text-sm text-right font-semibold ${acc !== null ? (acc<=10?'text-green-600':acc<=20?'text-yellow-600':'text-red-600') : 'text-gray-500'}`}>{acc !== null ? `${acc.toFixed(1)}%` : '-'}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* Results table */}
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Month</th>
                  <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Deliveries</th>
                  {result.forecasts[0]?.revenue !== undefined && (
                    <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">Revenue</th>
                  )}
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {result.forecasts.map((f, idx) => (
                  <tr key={idx} className={idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{new Date(f.date).toLocaleDateString('en-GB', {year:'numeric', month:'short'})}</td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-blue-700 font-semibold">{Math.round(f.forecast).toLocaleString()}</td>
                    {f.revenue !== undefined && (
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-right text-purple-700 font-semibold">${Math.round(f.revenue).toLocaleString()}</td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default VehicleForecast;
