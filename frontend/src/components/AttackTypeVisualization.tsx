import React from 'react';
import {
  Chart as ChartJS,
  ArcElement,
  BarElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';

ChartJS.register(
  ArcElement,
  BarElement,
  CategoryScale,
  LinearScale,
  Title,
  Tooltip,
  Legend
);

interface AttackTypeVisualizationProps {
  attackTypes: Record<string, number>;
  region: string;
  year: number;
}

const AttackTypeVisualization: React.FC<AttackTypeVisualizationProps> = ({
  attackTypes,
  region,
  year
}) => {
  const chartData = {
    labels: Object.keys(attackTypes),
    datasets: [
      {
        data: Object.values(attackTypes).map(v => v * 100),
        backgroundColor: [
          'rgba(255, 99, 132, 0.8)',
          'rgba(54, 162, 235, 0.8)',
          'rgba(255, 206, 86, 0.8)',
          'rgba(75, 192, 192, 0.8)',
          'rgba(153, 102, 255, 0.8)',
        ],
        borderColor: [
          'rgba(255, 99, 132, 1)',
          'rgba(54, 162, 235, 1)',
          'rgba(255, 206, 86, 1)',
          'rgba(75, 192, 192, 1)',
          'rgba(153, 102, 255, 1)',
        ],
        borderWidth: 1,
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'right' as const,
      },
      title: {
        display: true,
        text: `Attack Types Distribution - ${region} (${year})`,
      },
    },
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mt-6">
      <h3 className="text-lg font-semibold mb-4">Attack Type Analysis</h3>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="h-64">
          <Doughnut data={chartData} options={options} />
        </div>
        
        <div className="h-64">
          <Bar
            data={{
              ...chartData,
              datasets: [{
                ...chartData.datasets[0],
                borderRadius: 4,
              }],
            }}
            options={{
              ...options,
              indexAxis: 'y' as const,
              plugins: {
                ...options.plugins,
                legend: {
                  display: false,
                },
              },
            }}
          />
        </div>
      </div>

      <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
        {Object.entries(attackTypes).map(([type, probability]) => (
          <div
            key={type}
            className="p-4 bg-gray-50 rounded-lg"
          >
            <div className="text-sm font-medium text-gray-600">{type}</div>
            <div className="text-2xl font-bold text-gray-900">
              {(probability * 100).toFixed(1)}%
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default AttackTypeVisualization; 