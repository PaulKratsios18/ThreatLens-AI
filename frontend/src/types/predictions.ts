export interface Prediction {
  year: number;
  region_name: string;
  expected_attacks: number;
  confidence_score: number;
  risk_level: 'High' | 'Medium' | 'Low';
  attack_types: Record<string, number>;
}

export interface PredictionMapProps {
  predictions: Prediction[];
  selectedYear: number;
  onRegionClick: (region: string) => void;
} 