export interface SocioeconomicFactors {
  gdp_per_capita: number;
  unemployment_rate: number;
  gini_index: number;
  population: number;
  urban_population_percent: number;
  primary_school_enrollment: number;
  life_expectancy: number;
}

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
  socioeconomic_factors?: SocioeconomicFactors;
}

export interface PredictionMapProps {
  predictions: Prediction[];
  selectedYear: number;
  onRegionSelect: (region: string) => void;
  onCountrySelect: (country: string) => void;
}

export interface RegionMetric {
  actual_attacks: number;
  predicted_attacks: number;
  absolute_error: number;
  accuracy: number;
}

export interface YearAccuracy {
  overall_accuracy: number;
  region_metrics: {
    [region: string]: RegionMetric;
  };
}

export interface AccuracyMetrics {
  [year: string]: YearAccuracy;
} 