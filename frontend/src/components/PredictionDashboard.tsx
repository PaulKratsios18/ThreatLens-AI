import React, { useState, useEffect } from 'react';
import PredictionMap from './PredictionMap';
import RegionDetails from './RegionDetails';
import CountryDetails from './CountryDetails';
import TimelineSlider from './TimelineSlider';
import StatisticsPanel from './StatisticsPanel';
import { Prediction } from '../types/predictions';

interface ApiPrediction {
  country: string;
  region: string;
  expected_attacks: number;
  confidence_score: number;
  risk_level: string;
  gti_score: number;
  rank: number;
  change_from_previous: number;
  attack_types: Record<string, number>;
  primary_groups: string[];
}

const PredictionDashboard: React.FC = () => {
  const [selectedYear, setSelectedYear] = useState(2023);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:8000/static-predictions');
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        // Transform the predictions for the selected year
        const yearPredictions = data.predictions[selectedYear.toString()] || [];
        const transformedPredictions: Prediction[] = yearPredictions.map((pred: ApiPrediction) => ({
          year: selectedYear,
          country: pred.country,
          region_name: pred.region,
          expected_attacks: pred.expected_attacks,
          confidence_score: pred.confidence_score,
          risk_level: pred.risk_level as 'High' | 'Medium' | 'Low',
          gti_score: pred.gti_score,
          rank: pred.rank,
          change_from_previous: pred.change_from_previous,
          attack_types: pred.attack_types,
          primary_groups: pred.primary_groups
        }));

        setPredictions(transformedPredictions);
      } catch (error) {
        console.error('Error fetching predictions:', error);
        setError(error instanceof Error ? error.message : 'Failed to load predictions');
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, [selectedYear]);

  // Function to handle country selection
  const handleCountryClick = (country: string) => {
    setSelectedCountry(country);
    
    // Find the region for this country
    const prediction = predictions.find(p => p.country === country);
    if (prediction) {
      setSelectedRegion(prediction.region_name);
    }
  };

  if (loading) {
    return <div className="min-h-screen flex items-center justify-center">Loading...</div>;
  }

  if (error) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center">
        <div className="text-red-600 mb-2">Failed to load predictions</div>
        <div className="text-sm text-gray-500">Error details: {error}</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Threat Prediction Dashboard</h1>
        <p className="text-gray-600">Interactive visualization of global security predictions</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <PredictionMap 
            predictions={predictions}
            selectedYear={selectedYear}
            onRegionClick={setSelectedRegion}
            onCountryClick={handleCountryClick}
          />
          <TimelineSlider
            years={[2023, 2024, 2025]}
            selectedYear={selectedYear}
            onChange={setSelectedYear}
          />
        </div>
        
        <div>
          <StatisticsPanel predictions={predictions} selectedYear={selectedYear} />
          
          {selectedCountry && (
            <CountryDetails
              country={selectedCountry}
              predictions={predictions}
              year={selectedYear}
            />
          )}
          
          {selectedRegion && !selectedCountry && (
            <RegionDetails
              region={selectedRegion}
              predictions={predictions}
              year={selectedYear}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionDashboard; 