import React from 'react';

const TeslaFontDemo = () => {
  return (
    <div className="p-8 tesla-white-bg">
      <h1 className="tesla-text-5xl tesla-font-bold tesla-red mb-8">
        Tesla Financial Model
      </h1>
      
      <div className="mb-8">
        <h2 className="tesla-text-3xl tesla-font-semibold tesla-black mb-4">
          Tesla Color Palette
        </h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div className="tesla-card text-center">
            <div className="w-16 h-16 tesla-red-bg rounded-lg mx-auto mb-2"></div>
            <div className="tesla-text-sm tesla-font-semibold tesla-black">Tesla Red</div>
            <div className="tesla-text-xs tesla-gray">#e31e24</div>
          </div>
          <div className="tesla-card text-center">
            <div className="w-16 h-16 tesla-black-bg rounded-lg mx-auto mb-2"></div>
            <div className="tesla-text-sm tesla-font-semibold tesla-black">Tesla Black</div>
            <div className="tesla-text-xs tesla-gray">#000000</div>
          </div>
          <div className="tesla-card text-center">
            <div className="w-16 h-16 tesla-white-bg tesla-gray-border border rounded-lg mx-auto mb-2"></div>
            <div className="tesla-text-sm tesla-font-semibold tesla-black">Tesla White</div>
            <div className="tesla-text-xs tesla-gray">#ffffff</div>
          </div>
          <div className="tesla-card text-center">
            <div className="w-16 h-16 tesla-gray-bg rounded-lg mx-auto mb-2"></div>
            <div className="tesla-text-sm tesla-font-semibold tesla-black">Tesla Gray</div>
            <div className="tesla-text-xs tesla-gray">#5c5e62</div>
          </div>
          <div className="tesla-card text-center">
            <div className="w-16 h-16 tesla-blue-bg rounded-lg mx-auto mb-2"></div>
            <div className="tesla-text-sm tesla-font-semibold tesla-white">Tesla Blue</div>
            <div className="tesla-text-xs tesla-white">#007acc</div>
          </div>
          <div className="tesla-card text-center">
            <div className="w-16 h-16 tesla-green-bg rounded-lg mx-auto mb-2"></div>
            <div className="tesla-text-sm tesla-font-semibold tesla-white">Tesla Green</div>
            <div className="tesla-text-xs tesla-white">#00d4aa</div>
          </div>
          <div className="tesla-card text-center">
            <div className="w-16 h-16 tesla-orange-bg rounded-lg mx-auto mb-2"></div>
            <div className="tesla-text-sm tesla-font-semibold tesla-white">Tesla Orange</div>
            <div className="tesla-text-xs tesla-white">#ff6b35</div>
          </div>
          <div className="tesla-card text-center">
            <div className="w-16 h-16 tesla-silver-bg rounded-lg mx-auto mb-2"></div>
            <div className="tesla-text-sm tesla-font-semibold tesla-black">Tesla Silver</div>
            <div className="tesla-text-xs tesla-gray">#c0c0c0</div>
          </div>
        </div>
      </div>
      
      <div className="space-y-6">
        <div>
          <h2 className="tesla-text-3xl tesla-font-semibold mb-4">
            Typography Scale
          </h2>
          <div className="space-y-2">
            <p className="tesla-text-xs">Extra Small Text (12px)</p>
            <p className="tesla-text-sm">Small Text (14px)</p>
            <p className="tesla-text-base">Base Text (16px)</p>
            <p className="tesla-text-lg">Large Text (18px)</p>
            <p className="tesla-text-xl">Extra Large Text (20px)</p>
            <p className="tesla-text-2xl">2X Large Text (24px)</p>
            <p className="tesla-text-3xl">3X Large Text (30px)</p>
            <p className="tesla-text-4xl">4X Large Text (36px)</p>
          </div>
        </div>

        <div>
          <h2 className="tesla-text-2xl tesla-font-semibold mb-4">
            Font Weights
          </h2>
          <div className="space-y-2">
            <p className="tesla-font-light">Light Weight (300)</p>
            <p className="tesla-font-normal">Normal Weight (400)</p>
            <p className="tesla-font-medium">Medium Weight (500)</p>
            <p className="tesla-font-semibold">Semibold Weight (600)</p>
            <p className="tesla-font-bold">Bold Weight (700)</p>
            <p className="tesla-font-extrabold">Extrabold Weight (800)</p>
          </div>
        </div>

        <div>
          <h2 className="tesla-text-2xl tesla-font-semibold tesla-black mb-4">
            Tesla Component Styles
          </h2>
          <div className="space-y-4">
            <div className="tesla-card-title tesla-black">Card Title</div>
            <div className="tesla-card-subtitle tesla-gray">Card Subtitle</div>
            <button className="tesla-button-primary">
              Primary Button
            </button>
            <button className="tesla-button-secondary">
              Secondary Button
            </button>
            <div className="tesla-metric-value tesla-red">$2.96</div>
            <div className="tesla-metric-label tesla-gray">EPS</div>
            <div className="tesla-alert-success">Success: Data loaded successfully</div>
            <div className="tesla-alert-warning">Warning: Check your inputs</div>
            <div className="tesla-alert-error">Error: Something went wrong</div>
            <div className="tesla-alert-info">Info: Additional information</div>
          </div>
        </div>

        <div>
          <h2 className="tesla-text-2xl tesla-font-semibold tesla-black mb-4">
            Financial Table Example
          </h2>
          <table className="tesla-table">
            <thead>
              <tr>
                <th className="tesla-table-header">Metric</th>
                <th className="tesla-table-header">Value</th>
                <th className="tesla-table-header">Change</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="tesla-table-cell">Revenue</td>
                <td className="tesla-table-cell tesla-font-semibold tesla-black">$118.28B</td>
                <td className="tesla-table-cell tesla-green">+15.2%</td>
              </tr>
              <tr>
                <td className="tesla-table-cell">Gross Margin</td>
                <td className="tesla-table-cell tesla-font-semibold tesla-black">19.5%</td>
                <td className="tesla-table-cell tesla-green">+1.2%</td>
              </tr>
              <tr>
                <td className="tesla-table-cell">EPS</td>
                <td className="tesla-table-cell tesla-font-semibold tesla-black">$2.96</td>
                <td className="tesla-table-cell tesla-green">+22.1%</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div>
          <h2 className="tesla-text-2xl tesla-font-semibold mb-4">
            Monospace Font (for code/data)
          </h2>
          <div className="tesla-mono bg-gray-100 p-4 rounded">
            <div>Revenue: $118,280,000,000</div>
            <div>EPS: $2.96</div>
            <div>Shares: 3,172,000,000</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TeslaFontDemo;
