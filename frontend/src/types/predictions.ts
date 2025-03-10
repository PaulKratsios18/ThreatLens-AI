export interface Prediction {
  year: number;
  country: string;
  region_name: string;
  expected_attacks: number;
  confidence_score: number;
  risk_level: 'High' | 'Medium' | 'Low';
  gti_score?: number;
  rank?: number;
  change_from_previous?: number;
  attack_types: Record<string, number>;
  primary_groups?: string[];
}

export interface PredictionMapProps {
  predictions: Prediction[];
  selectedYear: number;
  onRegionClick: (region: string) => void;
  onCountryClick?: (country: string) => void;
} 