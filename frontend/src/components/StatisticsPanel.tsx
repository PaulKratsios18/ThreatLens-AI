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
  
  // Calculate total expected attacks
  const totalExpectedAttacks = Math.round(yearPredictions.reduce((sum, pred) => sum + (pred.expected_attacks || 0), 0));
  
  // Calculate average confidence score
  const averageConfidence = yearPredictions.length > 0
    ? yearPredictions.reduce((sum, pred) => sum + (typeof pred.confidence_score === 'number' ? pred.confidence_score : 0), 0) / yearPredictions.length
    : 0;

  // Calculate average GTI score
  const averageGtiScore = yearPredictions.length > 0 && yearPredictions[0].gti_score !== undefined
    ? yearPredictions.reduce((sum, pred) => sum + (pred.gti_score || 0), 0) / yearPredictions.length
    : undefined;
  
  // Calculate risk level distribution
  const riskCounts = yearPredictions.reduce((counts, pred) => {
    counts[pred.risk_level] = (counts[pred.risk_level] || 0) + 1;
    return counts;
  }, {} as Record<string, number>);
  
  const riskData = {
    labels: Object.keys(riskCounts),
    datasets: [
      {
        data: Object.values(riskCounts),
        backgroundColor: [
          '#ef4444', // red for High
          '#f59e0b', // amber for Medium
          '#22c55e', // green for Low
        ],
        borderWidth: 1,
      },
    ],
  };

  // Calculate most threatened countries (by GTI score)
  const mostThreatenedCountries = [...yearPredictions]
    .sort((a, b) => (b.gti_score || 0) - (a.gti_score || 0))
    .slice(0, 5);

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-bold mb-4">Global Statistics ({selectedYear})</h2>
      
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {yearPredictions.length}
          </div>
          <div className="text-sm text-gray-600">Predicted Events</div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {totalExpectedAttacks}
          </div>
          <div className="text-sm text-gray-600">Total Expected Attacks</div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {(averageConfidence * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Average Confidence</div>
        </div>
        {averageGtiScore !== undefined && (
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="text-2xl font-bold text-gray-900">
              {averageGtiScore.toFixed(1)}
            </div>
            <div className="text-sm text-gray-600">Average GTI Score</div>
          </div>
        )}
      </div>

      <div className="mt-6">
        <h3 className="text-lg font-semibold mb-2">Risk Level Distribution</h3>
        <div className="h-64">
          <Pie 
            data={riskData} 
            options={{
              responsive: true,
              maintainAspectRatio: false,
              plugins: {
                legend: {
                  position: 'right',
                }
              }
            }} 
          />
        </div>
      </div>

      {mostThreatenedCountries.length > 0 && (
        <>
          <h3 className="text-lg font-semibold mb-2">Most Threatened Countries</h3>
          <ul className="divide-y">
            {mostThreatenedCountries.map((pred) => (
              <li key={pred.country} className="py-2 flex justify-between items-center">
                <span>{pred.country}</span>
                <div className="flex items-center">
                  <span className="font-medium mr-2">GTI: {pred.gti_score?.toFixed(1)}</span>
                  <span className={`px-2 py-1 rounded-full text-xs text-white ${
                    pred.risk_level === 'High' ? 'bg-red-500' : 
                    pred.risk_level === 'Medium' ? 'bg-amber-500' : 
                    'bg-green-500'
                  }`}>
                    {pred.risk_level}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
};

export default StatisticsPanel; 