import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  CircularProgress,
  Alert
} from '@mui/material';

interface FeatureImportance {
  feature: string;
  importance: number;
  effect: 'positive' | 'negative';
}

interface PredictionFactorsChartProps {
  country: string;
  year: number;
}

const PredictionFactorsChart: React.FC<PredictionFactorsChartProps> = ({ 
  country, 
  year 
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [features, setFeatures] = useState<FeatureImportance[]>([]);

  useEffect(() => {
    const fetchFeatureImportance = async () => {
      if (!country) return;
      
      try {
        setLoading(true);
        // Mock API request for explanation - in a real app, this would call your backend
        // Example: const response = await fetch(`http://localhost:8000/explain?country=${country}&year=${year}`);
        
        // For demo purposes, we'll use mock data
        const mockFeatures: FeatureImportance[] = [
          { feature: 'Previous Attacks', importance: 0.28, effect: 'positive' },
          { feature: 'Political Instability', importance: 0.23, effect: 'positive' },
          { feature: 'GDP Per Capita', importance: 0.18, effect: 'negative' },
          { feature: 'Unemployment Rate', importance: 0.15, effect: 'positive' },
          { feature: 'Urban Population %', importance: 0.09, effect: 'negative' },
          { feature: 'Life Expectancy', importance: 0.07, effect: 'negative' },
        ];
        
        // Sort by importance
        mockFeatures.sort((a, b) => b.importance - a.importance);
        
        setFeatures(mockFeatures);
      } catch (err) {
        console.error("Error fetching feature importance:", err);
        setError("Failed to load prediction factors. Please try again later.");
      } finally {
        setLoading(false);
      }
    };

    fetchFeatureImportance();
  }, [country, year]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return <Alert severity="error">{error}</Alert>;
  }

  return (
    <Paper elevation={2} sx={{ p: 3, height: '100%' }}>
      <Typography variant="h6" gutterBottom>
        Factors Influencing Risk for {country}
      </Typography>
      <Typography variant="body2" color="text.secondary" gutterBottom>
        These factors had the most impact on the prediction. Positive factors increase risk, negative factors decrease it.
      </Typography>
      
      <Box sx={{ mt: 2 }}>
        {features.map((feature, index) => (
          <Box key={index} sx={{ mb: 2 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 0.5 }}>
              <Typography variant="body2">{feature.feature}</Typography>
              <Typography variant="body2" color={feature.effect === 'positive' ? 'error' : 'success'}>
                {(feature.importance * 100).toFixed(1)}% 
                {feature.effect === 'positive' ? ' ↑' : ' ↓'}
              </Typography>
            </Box>
            <Box sx={{ position: 'relative', height: 12, bgcolor: 'grey.200', borderRadius: 1 }}>
              <Box 
                sx={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  height: '100%',
                  width: `${feature.importance * 100}%`,
                  bgcolor: feature.effect === 'positive' ? 'error.main' : 'success.main',
                  borderRadius: 1
                }}
              />
            </Box>
          </Box>
        ))}
      </Box>
    </Paper>
  );
};

export default PredictionFactorsChart; 