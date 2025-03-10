import React from 'react';
import { Box, Typography, Paper, Chip, List, ListItem, ListItemText, Divider } from '@mui/material';
import { Line } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';
import { Prediction } from '../types/predictions';

Chart.register(...registerables);

interface CountryDetailsProps {
  country: string;
  predictions: Prediction[];
  year: number;
}

const CountryDetails: React.FC<CountryDetailsProps> = ({ country, predictions, year }) => {
  // Find prediction for the selected country
  const prediction = predictions.find(p => p.country === country);

  if (!prediction) {
    return (
      <Paper elevation={3} sx={{ p: 2, my: 2 }}>
        <Typography variant="h6" component="h3" gutterBottom>
          No data available for {country}
        </Typography>
      </Paper>
    );
  }

  // Calculate total attacks from attack types
  const totalAttacks = Object.values(prediction.attack_types).reduce((a, b) => a + b, 0);

  // Prepare data for attack types chart
  const attackTypeData = {
    labels: Object.keys(prediction.attack_types),
    datasets: [
      {
        label: 'Attack Types',
        data: Object.values(prediction.attack_types),
        backgroundColor: [
          'rgba(255, 99, 132, 0.5)',
          'rgba(54, 162, 235, 0.5)',
          'rgba(255, 206, 86, 0.5)',
          'rgba(75, 192, 192, 0.5)',
          'rgba(153, 102, 255, 0.5)',
        ],
        borderWidth: 1,
      },
    ],
  };

  // Color for risk level
  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      case 'low':
        return 'success';
      default:
        return 'default';
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 2, my: 2 }}>
      <Typography variant="h6" component="h3" gutterBottom>
        {country} - {year} Threat Analysis
      </Typography>
      
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="body1" sx={{ mr: 2 }}>
          Risk Level:
        </Typography>
        <Chip 
          label={prediction.risk_level} 
          color={getRiskColor(prediction.risk_level)} 
          size="small" 
        />
      </Box>

      <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mb: 3 }}>
        <Paper variant="outlined" sx={{ p: 2, flex: '1 1 180px' }}>
          <Typography variant="subtitle2" color="text.secondary">
            GTI Score
          </Typography>
          <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
            {prediction.gti_score?.toFixed(1) || 'N/A'}
          </Typography>
          <Typography variant="caption" color={prediction.change_from_previous && prediction.change_from_previous > 0 ? 'error.main' : 'success.main'}>
            {prediction.change_from_previous ? (prediction.change_from_previous > 0 ? `↑ +${prediction.change_from_previous.toFixed(1)}` : `↓ ${prediction.change_from_previous.toFixed(1)}`) : ''}
          </Typography>
        </Paper>

        <Paper variant="outlined" sx={{ p: 2, flex: '1 1 180px' }}>
          <Typography variant="subtitle2" color="text.secondary">
            Global Rank
          </Typography>
          <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
            {prediction.rank || 'N/A'}
          </Typography>
        </Paper>

        <Paper variant="outlined" sx={{ p: 2, flex: '1 1 180px' }}>
          <Typography variant="subtitle2" color="text.secondary">
            Expected Attacks
          </Typography>
          <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
            {prediction.expected_attacks}
          </Typography>
        </Paper>

        <Paper variant="outlined" sx={{ p: 2, flex: '1 1 180px' }}>
          <Typography variant="subtitle2" color="text.secondary">
            Confidence
          </Typography>
          <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
            {(prediction.confidence_score * 100).toFixed(0)}%
          </Typography>
        </Paper>
      </Box>

      <Typography variant="subtitle1" gutterBottom sx={{ mt: 3 }}>
        Attack Type Distribution
      </Typography>
      <Box sx={{ height: 200, mb: 3 }}>
        <Line
          data={attackTypeData}
          options={{
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
              legend: {
                position: 'top',
              },
            },
          }}
        />
      </Box>

      {prediction.primary_groups && prediction.primary_groups.length > 0 && (
        <>
          <Typography variant="subtitle1" gutterBottom>
            Primary Threat Groups
          </Typography>
          <List>
            {prediction.primary_groups.map((group, index) => (
              <React.Fragment key={group}>
                <ListItem>
                  <ListItemText primary={group} />
                </ListItem>
                {index < prediction.primary_groups!.length - 1 && <Divider />}
              </React.Fragment>
            ))}
          </List>
        </>
      )}
    </Paper>
  );
};

export default CountryDetails; 