import React from 'react';
import { Prediction } from '../types/predictions';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  LinearProgress,
  Tooltip,
  Divider
} from '@mui/material';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer } from 'recharts';
import PredictionFactorsChart from './PredictionFactorsChart';

interface CountryDetailsProps {
  prediction: Prediction;
}

const CountryDetails: React.FC<CountryDetailsProps> = ({ prediction }) => {
  const formatNumber = (num: number) => {
    if (num >= 1000000) {
      return `${(num / 1000000).toFixed(1)}M`;
    }
    if (num >= 1000) {
      return `${(num / 1000).toFixed(1)}K`;
    }
    return num.toFixed(1);
  };

  const formatPercentage = (num: number) => `${num.toFixed(1)}%`;

  const socioeconomicData = [
    {
      name: 'GDP per Capita',
      value: prediction.socioeconomic_factors?.gdp_per_capita,
      unit: '$',
      format: formatNumber,
      description: 'Gross Domestic Product per person'
    },
    {
      name: 'Unemployment Rate',
      value: prediction.socioeconomic_factors?.unemployment_rate,
      unit: '%',
      format: formatPercentage,
      description: 'Percentage of labor force that is unemployed'
    },
    {
      name: 'Gini Index',
      value: prediction.socioeconomic_factors?.gini_index,
      unit: '',
      format: (num: number) => num.toFixed(2),
      description: 'Measure of income inequality (0-1)'
    },
    {
      name: 'Population',
      value: prediction.socioeconomic_factors?.population,
      unit: '',
      format: formatNumber,
      description: 'Total population'
    },
    {
      name: 'Urban Population',
      value: prediction.socioeconomic_factors?.urban_population_percent,
      unit: '%',
      format: formatPercentage,
      description: 'Percentage of population living in urban areas'
    },
    {
      name: 'Primary School Enrollment',
      value: prediction.socioeconomic_factors?.primary_school_enrollment,
      unit: '%',
      format: formatPercentage,
      description: 'Percentage of children enrolled in primary school'
    },
    {
      name: 'Life Expectancy',
      value: prediction.socioeconomic_factors?.life_expectancy,
      unit: 'years',
      format: (num: number) => num.toFixed(1),
      description: 'Average life expectancy at birth'
    }
  ];

  // Make sure attack_types exists before trying to use Object.entries
  const attackTypeData = prediction.attack_types 
    ? Object.entries(prediction.attack_types).map(([type, count]) => ({
        name: type,
        value: count
      }))
    : [];
    
  // Check if we have benchmark data for this prediction
  const hasBenchmarkData = prediction.actual_attacks !== undefined;
  
  // If we have benchmark data, prepare data for comparison chart
  const benchmarkData = hasBenchmarkData ? [
    { name: 'Predicted', value: Math.round(prediction.expected_attacks) },
    { name: 'Actual', value: prediction.actual_attacks }
  ] : [];

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          {prediction.country}
        </Typography>
        
        <Grid container spacing={2}>
          {/* Risk Level */}
          <Grid item xs={12}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle1" gutterBottom>
                Risk Level: {prediction.risk_level}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={prediction.confidence_score * 100}
                color={
                  prediction.risk_level === 'High' ? 'error' :
                  prediction.risk_level === 'Medium' ? 'warning' : 'success'
                }
              />
            </Box>
          </Grid>

          {/* Expected Attacks */}
          <Grid item xs={12} sm={hasBenchmarkData ? 4 : 6}>
            <Card variant="outlined">
              <CardContent>
                <Typography variant="h6">Expected Attacks</Typography>
                <Typography variant="h4">{Math.round(prediction.expected_attacks)}</Typography>
                <Typography variant="body2" color="textSecondary">
                  Confidence: {(prediction.confidence_score * 100).toFixed(1)}%
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Actual Attacks (when available) */}
          {hasBenchmarkData && (
            <Grid item xs={12} sm={4}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6">Actual Attacks</Typography>
                  <Typography variant="h4">{prediction.actual_attacks}</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Historical Record
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Prediction Accuracy (when available) */}
          {hasBenchmarkData && (
            <Grid item xs={12} sm={4}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6">Prediction Accuracy</Typography>
                  <Typography 
                    variant="h4" 
                    color={prediction.accuracy && prediction.accuracy > 80 ? 'success.main' : 
                           prediction.accuracy && prediction.accuracy > 60 ? 'warning.main' : 'error.main'}
                  >
                    {prediction.accuracy ? prediction.accuracy.toFixed(1) : '0'}%
                  </Typography>
                  <Typography variant="body2" color="textSecondary">
                    Model Performance
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* GTI Score */}
          {prediction.gti_score && (
            <Grid item xs={12} sm={hasBenchmarkData ? 4 : 6}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6">GTI Score</Typography>
                  <Typography variant="h4">{prediction.gti_score.toFixed(1)}</Typography>
                  <Typography variant="body2" color="textSecondary">
                    Rank: #{prediction.rank}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          )}
          
          {/* Benchmark Comparison Chart */}
          {hasBenchmarkData && (
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>Prediction vs. Actual</Typography>
                  <Box sx={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={benchmarkData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <RechartsTooltip />
                        <Legend />
                        <Bar dataKey="value" name="Attacks" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}
          
          {/* Prediction Factors Visualization */}
          <Grid item xs={12}>
            <PredictionFactorsChart 
              country={prediction.country} 
              year={prediction.year} 
            />
          </Grid>

          {/* Attack Types Chart */}
          {attackTypeData.length > 0 && (
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>Attack Types Distribution</Typography>
                  <Box sx={{ height: 300 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={attackTypeData}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="name" />
                        <YAxis />
                        <RechartsTooltip />
                        <Bar dataKey="value" fill="#8884d8" />
                      </BarChart>
                    </ResponsiveContainer>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Socioeconomic Factors */}
          {prediction.socioeconomic_factors && (
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>Socioeconomic Factors</Typography>
                  <Grid container spacing={2}>
                    {socioeconomicData.map((item) => (
                      <Grid item xs={12} sm={6} md={4} key={item.name}>
                        <Tooltip title={item.description}>
                          <Box>
                            <Typography variant="subtitle2" color="textSecondary">
                              {item.name}
                            </Typography>
                            <Typography variant="h6">
                              {item.value ? `${item.format(item.value)}${item.unit}` : 'N/A'}
                            </Typography>
                          </Box>
                        </Tooltip>
                      </Grid>
                    ))}
                  </Grid>
                </CardContent>
              </Card>
            </Grid>
          )}

          {/* Primary Groups */}
          {prediction.primary_groups && prediction.primary_groups.length > 0 && (
            <Grid item xs={12}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>Primary Groups</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {prediction.primary_groups.map((group) => (
                      <Typography
                        key={group}
                        variant="body2"
                        sx={{
                          bgcolor: 'primary.light',
                          color: 'primary.contrastText',
                          px: 1,
                          py: 0.5,
                          borderRadius: 1
                        }}
                      >
                        {group}
                      </Typography>
                    ))}
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          )}
        </Grid>
      </CardContent>
    </Card>
  );
};

export default CountryDetails; 