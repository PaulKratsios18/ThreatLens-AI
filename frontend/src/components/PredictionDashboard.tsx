import React, { useState, useEffect } from 'react';
import PredictionMap from './PredictionMap';
import RegionDetails from './RegionDetails';
import CountryDetails from './CountryDetails';
import TimelineSlider from './TimelineSlider';
import StatisticsPanel from './StatisticsPanel';
import { Prediction, AccuracyMetrics, RegionMetric } from '../types/predictions';
import { Box, Grid, Typography, CircularProgress, Alert, Card, CardContent } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

interface ApiPrediction {
  country: string;
  region: string;
  region_name?: string;
  expected_attacks: number;
  confidence_score: number;
  risk_level: string;
  gti_score: number;
  rank: number;
  change_from_previous: number;
  attack_types: Record<string, number>;
  primary_groups: string[];
  actual_attacks?: number;
  accuracy?: number;
  socioeconomic_factors?: {
    gdp_per_capita: number;
    unemployment_rate: number;
    gini_index: number;
    population: number;
    urban_population_percent: number;
    primary_school_enrollment: number;
    life_expectancy: number;
  };
}

interface PredictionApiResponse {
  predictions: {
    [year: string]: ApiPrediction[];
  };
  accuracy?: AccuracyMetrics;
}

const PredictionDashboard: React.FC = () => {
  const [selectedYear, setSelectedYear] = useState(2023);
  const [selectedRegion, setSelectedRegion] = useState<string | null>(null);
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [availableYears, setAvailableYears] = useState<number[]>([2023, 2024, 2025]);
  const [accuracyMetrics, setAccuracyMetrics] = useState<AccuracyMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [allPredictions, setAllPredictions] = useState<{[year: string]: Prediction[]}>({});

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        setLoading(true);
        // Use the model-predictions endpoint with a wider range of years
        const response = await fetch('http://localhost:8000/model-predictions?start_year=2000&end_year=2025');
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data: PredictionApiResponse = await response.json();
        
        // Get all available years from the response
        const years = Object.keys(data.predictions).map(year => parseInt(year)).sort((a, b) => a - b);
        
        if (years.length > 0) {
          setAvailableYears(years);
          
          // If selectedYear isn't in the available years, set it to the most recent year
          if (!years.includes(selectedYear)) {
            setSelectedYear(years[years.length - 1]);
          }
          
          // Transform and store all years' predictions
          const transformedPredictions: {[year: string]: Prediction[]} = {};
          
          for (const yearStr in data.predictions) {
            const yearData = data.predictions[yearStr];
            transformedPredictions[yearStr] = yearData.map((pred: ApiPrediction) => ({
              year: parseInt(yearStr),
              country: pred.country,
              region_name: pred.region_name || pred.region,
              expected_attacks: pred.expected_attacks,
              confidence_score: pred.confidence_score,
              risk_level: pred.risk_level as 'High' | 'Medium' | 'Low',
              gti_score: pred.gti_score,
              rank: pred.rank,
              change_from_previous: pred.change_from_previous,
              attack_types: pred.attack_types,
              primary_groups: pred.primary_groups,
              socioeconomic_factors: pred.socioeconomic_factors,
              actual_attacks: pred.actual_attacks,
              accuracy: pred.accuracy
            }));
          }
          
          setAllPredictions(transformedPredictions);
          
          // Set current year predictions
          setPredictions(transformedPredictions[selectedYear.toString()] || []);
          
          // Set accuracy metrics if available
          if (data.accuracy) {
            setAccuracyMetrics(data.accuracy);
          }
        } else {
          throw new Error('No prediction data available');
        }
      } catch (error) {
        console.error('Error fetching predictions:', error);
        setError(error instanceof Error ? error.message : 'Failed to load predictions');
        
        // Fallback to static predictions if model predictions fail
        try {
          const staticResponse = await fetch('http://localhost:8000/static-predictions');
          if (staticResponse.ok) {
            const staticData = await staticResponse.json();
            const staticYearPredictions = staticData.predictions[selectedYear.toString()] || [];
            const staticTransformedPredictions: Prediction[] = staticYearPredictions.map((pred: ApiPrediction) => ({
              year: selectedYear,
              country: pred.country,
              region_name: pred.region_name || pred.region,
              expected_attacks: pred.expected_attacks,
              confidence_score: pred.confidence_score,
              risk_level: pred.risk_level as 'High' | 'Medium' | 'Low',
              gti_score: pred.gti_score,
              rank: pred.rank,
              change_from_previous: pred.change_from_previous,
              attack_types: pred.attack_types,
              primary_groups: pred.primary_groups,
              socioeconomic_factors: pred.socioeconomic_factors,
              actual_attacks: pred.actual_attacks,
              accuracy: pred.accuracy
            }));
            setPredictions(staticTransformedPredictions);
            setError("Using fallback data: " + (error instanceof Error ? error.message : 'Failed to load model predictions'));
          } else {
            throw new Error('Static predictions also failed');
          }
        } catch (fallbackError) {
          console.error('Error fetching fallback predictions:', fallbackError);
          
          // Hardcoded fallback data when all API calls fail
          const mockYears = [2023, 2024, 2025];
          setAvailableYears(mockYears);
          
          const mockPredictions: {[year: string]: Prediction[]} = {};
          
          // Generate mock data for each year
          mockYears.forEach(year => {
            mockPredictions[year.toString()] = [
              {
                year: year,
                country: "Iraq",
                region_name: "Middle East & North Africa",
                expected_attacks: 180,
                confidence_score: 0.85,
                risk_level: "High",
                gti_score: 8.5,
                rank: 1,
                change_from_previous: -5,
                attack_types: {
                  "Bombing/Explosion": 60,
                  "Armed Assault": 40,
                  "Hostage Taking": 10,
                  "Other": 15
                },
                primary_groups: ["ISIS", "Al-Qaeda", "Local Militants"]
              },
              {
                year: year,
                country: "Afghanistan",
                region_name: "South Asia",
                expected_attacks: 150,
                confidence_score: 0.78,
                risk_level: "High",
                gti_score: 7.8,
                rank: 2,
                change_from_previous: -8,
                attack_types: {
                  "Bombing/Explosion": 55,
                  "Armed Assault": 35,
                  "Assassination": 15,
                  "Other": 10
                },
                primary_groups: ["Taliban", "Local Extremists"]
              },
              {
                year: year,
                country: "Somalia",
                region_name: "Sub-Saharan Africa",
                expected_attacks: 120,
                confidence_score: 0.72,
                risk_level: "High",
                gti_score: 7.2,
                rank: 3,
                change_from_previous: 2,
                attack_types: {
                  "Bombing/Explosion": 40,
                  "Armed Assault": 30,
                  "Assassination": 20,
                  "Other": 30
                },
                primary_groups: ["Al-Shabaab", "Local Militants"]
              },
              {
                year: year,
                country: "Syria",
                region_name: "Middle East & North Africa",
                expected_attacks: 110,
                confidence_score: 0.68,
                risk_level: "High",
                gti_score: 6.8,
                rank: 4,
                change_from_previous: -12,
                attack_types: {
                  "Bombing/Explosion": 45,
                  "Armed Assault": 25,
                  "Hostage Taking": 15,
                  "Other": 25
                },
                primary_groups: ["ISIS", "Al-Nusra", "Local Militants"]
              },
              {
                year: year,
                country: "Nigeria",
                region_name: "Sub-Saharan Africa",
                expected_attacks: 95,
                confidence_score: 0.65,
                risk_level: "High",
                gti_score: 6.5,
                rank: 5,
                change_from_previous: -3,
                attack_types: {
                  "Bombing/Explosion": 35,
                  "Armed Assault": 30,
                  "Hostage Taking": 20,
                  "Other": 10
                },
                primary_groups: ["Boko Haram", "ISWAP", "Local Militants"]
              }
            ];
          });
          
          setAllPredictions(mockPredictions);
          setPredictions(mockPredictions[selectedYear.toString()] || mockPredictions['2023']);
          setError("Using embedded mock data: Backend API unavailable");
        }
      } finally {
        setLoading(false);
      }
    };

    fetchPredictions();
  }, []);

  const handleYearChange = (year: number) => {
    setSelectedYear(year);
    setSelectedCountry(null);
    setSelectedRegion(null);
    
    // Update predictions for the selected year
    setPredictions(allPredictions[year.toString()] || []);
  };

  const handleCountrySelect = (country: string) => {
    setSelectedCountry(country);
    setSelectedRegion(null);
  };

  const handleRegionSelect = (region: string) => {
    setSelectedRegion(region);
    setSelectedCountry(null);
  };

  const renderAccuracyMetrics = () => {
    // Only show for historical years (2000-2022)
    if (!accuracyMetrics || selectedYear > 2022) {
      return null;
    }

    const yearMetrics = accuracyMetrics[selectedYear.toString()];
    if (!yearMetrics) {
      return null;
    }

    // Prepare data for the accuracy chart
    const chartData = Object.entries(yearMetrics.region_metrics).map(([region, metrics]) => ({
      region,
      accuracy: (metrics as RegionMetric).accuracy,
      predicted: (metrics as RegionMetric).predicted_attacks,
      actual: (metrics as RegionMetric).actual_attacks,
    }));

    return (
      <Card variant="outlined" sx={{ mt: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>Prediction Accuracy for {selectedYear}</Typography>
          <Typography variant="body1">
            Overall Accuracy: {yearMetrics.overall_accuracy.toFixed(1)}%
          </Typography>
          
          <Box sx={{ mt: 2 }}>
            <Typography variant="subtitle1" gutterBottom>Region Accuracy</Typography>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="region" angle={-45} textAnchor="end" height={80} />
                <YAxis yAxisId="left" label={{ value: 'Accuracy %', angle: -90, position: 'insideLeft' }} />
                <YAxis yAxisId="right" orientation="right" label={{ value: 'Attacks', angle: 90, position: 'insideRight' }} />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="accuracy" stroke="#8884d8" name="Accuracy %" />
                <Line yAxisId="right" type="monotone" dataKey="predicted" stroke="#82ca9d" name="Predicted Attacks" />
                <Line yAxisId="right" type="monotone" dataKey="actual" stroke="#ff7300" name="Actual Attacks" />
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>
    );
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box sx={{ p: 3 }}>
        <Alert severity="error">{error}</Alert>
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Terrorism Threat Prediction Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* Timeline Slider */}
        <Grid item xs={12}>
          <TimelineSlider
            selectedYear={selectedYear}
            onYearChange={handleYearChange}
            years={availableYears}
          />
        </Grid>

        {/* Map */}
        <Grid item xs={12} md={8}>
          <PredictionMap
            predictions={predictions}
            selectedYear={selectedYear}
            onCountrySelect={handleCountrySelect}
            onRegionSelect={handleRegionSelect}
          />
        </Grid>

        {/* Statistics Panel */}
        <Grid item xs={12} md={4}>
          <StatisticsPanel
            predictions={predictions}
            selectedYear={selectedYear}
          />
        </Grid>

        {/* Historical Accuracy Metrics (only for historical years) */}
        {selectedYear <= 2022 && (
          <Grid item xs={12}>
            {renderAccuracyMetrics()}
          </Grid>
        )}

        {/* Country/Region Details */}
        <Grid item xs={12}>
          {selectedCountry ? (
            <CountryDetails
              prediction={predictions.find(p => p.country === selectedCountry)!}
            />
          ) : selectedRegion ? (
            <RegionDetails
              region={selectedRegion}
              predictions={predictions}
              year={selectedYear}
            />
          ) : (
            <Typography variant="body1" color="textSecondary">
              Select a country or region to view detailed information
            </Typography>
          )}
        </Grid>
      </Grid>
    </Box>
  );
};

export default PredictionDashboard; 