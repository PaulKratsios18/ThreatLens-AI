import React from 'react';
import { Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import AttackTypeVisualization from './AttackTypeVisualization';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface RegionDetailsProps {
  region: string;
  predictions: any[];
  year: number;
}

const RegionDetails: React.FC<RegionDetailsProps> = ({ region, predictions, year }) => {
  const regionData = predictions.find(p => p.region_name === region && p.year === year);

  if (!regionData) return null;

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
      <h2 className="text-xl font-bold mb-4">{region} Details</h2>
      
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {regionData.expected_attacks}
          </div>
          <div className="text-sm text-gray-600">Expected Attacks</div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {(regionData.confidence_score * 100).toFixed(1)}%
          </div>
          <div className="text-sm text-gray-600">Confidence Score</div>
        </div>
      </div>

      <AttackTypeVisualization
        attackTypes={regionData.attack_types}
        region={region}
        year={year}
      />
    </div>
  );
};

export default RegionDetails; 