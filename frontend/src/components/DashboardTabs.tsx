import React, { useState } from 'react';
import PredictionDashboard from './PredictionDashboard';
import HistoricalDashboard from './HistoricalDashboard';

const DashboardTabs: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'predictions' | 'historical'>('predictions');

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">Global Terrorism Analysis Dashboard</h1>
        <p className="text-gray-600">Analytics and predictions for global security incidents</p>
      </header>

      <div className="mb-6">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex">
            <button
              onClick={() => setActiveTab('predictions')}
              className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                activeTab === 'predictions'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Predictions
            </button>
            <button
              onClick={() => setActiveTab('historical')}
              className={`py-4 px-6 text-center border-b-2 font-medium text-sm ${
                activeTab === 'historical'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Historical Data
            </button>
          </nav>
        </div>
      </div>

      {activeTab === 'predictions' ? (
        <PredictionDashboard />
      ) : (
        <HistoricalDashboard />
      )}
    </div>
  );
};

export default DashboardTabs; 