export interface StatisticsPanelProps {
  predictions: {
    year: number;
    risk_level: 'High' | 'Medium' | 'Low';
    expected_attacks: number;
  }[];
  selectedYear: number;
} 