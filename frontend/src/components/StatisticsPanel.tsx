import React from 'react';
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale
} from 'chart.js';
import { Pie } from 'react-chartjs-2';
import { Prediction } from '../types/predictions';

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale
);

interface StatisticsPanelProps {
  predictions: Prediction[];
  selectedYear: number;
}

const StatisticsPanel: React.FC<StatisticsPanelProps> = ({ predictions, selectedYear }) => {
  if (!predictions || predictions.length === 0) {
    return <div>Loading statistics...</div>;
  }

  const yearPredictions = predictions.filter(p => p.year === selectedYear);
  
  const riskLevelData = {
    labels: ['High Risk', 'Medium Risk', 'Low Risk'],
    datasets: [{
      data: [
        yearPredictions.filter(p => p.risk_level === 'High').length,
        yearPredictions.filter(p => p.risk_level === 'Medium').length,
        yearPredictions.filter(p => p.risk_level === 'Low').length,
      ],
      backgroundColor: ['#ef4444', '#f59e0b', '#22c55e'],
    }]
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-bold mb-4">Global Statistics</h2>
      
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {yearPredictions.length}
          </div>
          <div className="text-sm text-gray-600">Predicted Events</div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {yearPredictions.reduce((acc, curr) => acc + curr.expected_attacks, 0)}
          </div>
          <div className="text-sm text-gray-600">Total Expected Attacks</div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {(yearPredictions.reduce((acc, curr) => acc + curr.confidence_score, 0) / yearPredictions.length * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Average Confidence</div>
        </div>
      </div>

      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-2">Risk Level Distribution</h3>
        <div className="h-64">
          <Pie data={riskLevelData} options={{ maintainAspectRatio: false }} />
        </div>
      </div>
    </div>
  );
};

export default StatisticsPanel; 