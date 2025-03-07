import React, { useState, useEffect } from 'react';
import { Bar, Pie, Line } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';
import HistoricalMap from './HistoricalMap';
import TimelineSlider from './TimelineSlider';

Chart.register(...registerables);

interface HistoricalAttack {
  id: number;
  year: number;
  month: number;
  day: number;
  region: string;
  country: string;
  city: string;
  latitude: number;
  longitude: number;
  attack_type: string;
  weapon_type: string;
  target_type: string;
  num_killed: number;
  num_wounded: number;
}

const HistoricalDashboard: React.FC = () => {
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedYear, setSelectedYear] = useState<number>(2020);
  const [historicalData, setHistoricalData] = useState<HistoricalAttack[]>([]);
  const [activeTab, setActiveTab] = useState<'overview' | 'regions' | 'trends'>('overview');

  useEffect(() => {
    const fetchHistoricalData = async () => {
      try {
        setLoading(true);
        // Replace with your endpoint
        const response = await fetch(`http://localhost:8000/historical-data?year=${selectedYear}`);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        setHistoricalData(data);
      } catch (error) {
        console.error('Error fetching historical data:', error);
        setError(error instanceof Error ? error.message : 'Failed to load historical data');
      } finally {
        setLoading(false);
      }
    };

    fetchHistoricalData();
  }, [selectedYear]);

  if (loading) {
    return <div className="flex items-center justify-center h-96">Loading historical data...</div>;
  }

  if (error) {
    return <div className="text-red-600">Error: {error}</div>;
  }

  // Calculate statistics for visualizations
  const attacksByRegion = historicalData.reduce((acc, attack) => {
    acc[attack.region] = (acc[attack.region] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  
  const attacksByType = historicalData.reduce((acc, attack) => {
    acc[attack.attack_type] = (acc[attack.attack_type] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);
  
  const attacksByMonth = Array(12).fill(0);
  historicalData.forEach(attack => {
    attacksByMonth[attack.month - 1]++;
  });

  // Prepare chart data
  const regionChartData = {
    labels: Object.keys(attacksByRegion),
    datasets: [
      {
        label: 'Number of Attacks',
        data: Object.values(attacksByRegion),
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      },
    ],
  };

  const typeChartData = {
    labels: Object.keys(attacksByType),
    datasets: [
      {
        label: 'Types of Attacks',
        data: Object.values(attacksByType),
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 206, 86, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(153, 102, 255, 0.5)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const monthlyChartData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    datasets: [
      {
        label: 'Monthly Attacks',
        data: attacksByMonth,
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <h2 className="text-2xl font-bold mb-4">Historical Data Analysis</h2>
      <div className="mb-6">
        <div className="mb-4">
          <TimelineSlider
            years={[2018, 2019, 2020, 2021, 2022]}
            selectedYear={selectedYear}
            onChange={setSelectedYear}
          />
        </div>
        
        <div className="border-b border-gray-200 mb-6">
          <nav className="-mb-px flex">
            <button
              onClick={() => setActiveTab('overview')}
              className={`py-2 px-4 text-center border-b-2 font-medium text-sm ${
                activeTab === 'overview'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('regions')}
              className={`py-2 px-4 text-center border-b-2 font-medium text-sm ${
                activeTab === 'regions'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Regional Analysis
            </button>
            <button
              onClick={() => setActiveTab('trends')}
              className={`py-2 px-4 text-center border-b-2 font-medium text-sm ${
                activeTab === 'trends'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Trends
            </button>
          </nav>
        </div>
      </div>

      {activeTab === 'overview' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-xl font-semibold mb-4">Attacks by Month ({selectedYear})</h3>
            <Line data={monthlyChartData} />
          </div>
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-xl font-semibold mb-4">Attack Types ({selectedYear})</h3>
            <Pie data={typeChartData} />
          </div>
          <div className="col-span-1 lg:col-span-2 bg-white p-4 rounded-lg shadow">
            <h3 className="text-xl font-semibold mb-4">Historical Incidents Map ({selectedYear})</h3>
            <HistoricalMap incidents={historicalData} />
          </div>
        </div>
      )}

      {activeTab === 'regions' && (
        <div className="grid grid-cols-1 gap-6">
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-xl font-semibold mb-4">Attacks by Region ({selectedYear})</h3>
            <Bar data={regionChartData} options={{ indexAxis: 'y' }} />
          </div>
          {/* Add more regional visualizations here */}
        </div>
      )}

      {activeTab === 'trends' && (
        <div className="grid grid-cols-1 gap-6">
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-xl font-semibold mb-4">Yearly Trends</h3>
            {/* Add trend visualizations here */}
            <p className="text-gray-500">Trend analysis to be implemented</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default HistoricalDashboard; 