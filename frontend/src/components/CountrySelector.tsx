import React from 'react';
import { 
  Autocomplete, 
  TextField, 
  Paper, 
  Typography, 
  Box,
  Chip,
  Divider,
  List,
  ListItem,
  ListItemText
} from '@mui/material';
import { HistoricalAttack, standardizeCountryName } from '../utils/countryUtils';

interface CountrySelectorProps {
  incidents: HistoricalAttack[];
  selectedCountry: string | null;
  onChange: (country: string | null) => void;
}

interface CountryOption {
  label: string;
  originalName: string;
  standardizedName: string;
  attackCount: number;
  deaths: number;
  wounded: number;
  groups: Record<string, number>;
}

const CountrySelector: React.FC<CountrySelectorProps> = ({ 
  incidents, 
  selectedCountry, 
  onChange 
}) => {
  const countryStats = React.useMemo(() => {
    const stats: Record<string, CountryOption> = {};
    
    incidents.forEach(incident => {
      const standardizedName = standardizeCountryName(incident.country);
      
      if (!stats[standardizedName]) {
        stats[standardizedName] = {
          label: standardizedName,
          originalName: incident.country,
          standardizedName,
          attackCount: 0,
          deaths: 0,
          wounded: 0,
          groups: {}
        };
      }
      
      stats[standardizedName].attackCount += 1;
      stats[standardizedName].deaths += incident.num_killed;
      stats[standardizedName].wounded += incident.num_wounded;
      
      if (incident.group_name) {
        const groupName = incident.group_name;
        stats[standardizedName].groups[groupName] = (stats[standardizedName].groups[groupName] || 0) + 1;
      }
    });
    
    return Object.values(stats)
      .sort((a, b) => b.attackCount - a.attackCount);
  }, [incidents]);
  
  const selectedOption = React.useMemo(() => {
    if (!selectedCountry) return null;
    return countryStats.find(stat => stat.standardizedName === selectedCountry) || null;
  }, [selectedCountry, countryStats]);

  const topGroups = React.useMemo(() => {
    if (!selectedOption) return [];
    
    return Object.entries(selectedOption.groups)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([name, count]) => ({ name, count }));
  }, [selectedOption]);

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Country Analysis
      </Typography>
      
      <Autocomplete
        value={selectedOption}
        onChange={(_event, newValue) => {
          onChange(newValue ? newValue.standardizedName : null);
        }}
        options={countryStats}
        getOptionLabel={(option) => option.label}
        renderInput={(params) => (
          <TextField
            {...params}
            label="Select a country"
            variant="outlined"
            size="small"
          />
        )}
        renderOption={(props, option) => (
          <li {...props}>
            <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <Typography variant="body1">
                  {option.standardizedName}
                </Typography>
                <Chip 
                  label={`${option.attackCount} attacks`} 
                  size="small"
                  color={
                    option.attackCount > 100 ? "error" :
                    option.attackCount > 50 ? "warning" :
                    option.attackCount > 10 ? "info" : "success"
                  }
                />
              </Box>
              {option.standardizedName !== option.originalName && (
                <Typography variant="caption" color="text.secondary">
                  Original name: {option.originalName}
                </Typography>
              )}
            </Box>
          </li>
        )}
      />
      
      {selectedOption && (
        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Statistics for {selectedOption.standardizedName}
          </Typography>
          
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2 }}>
            <Paper variant="outlined" sx={{ p: 1, flex: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Attacks
              </Typography>
              <Typography variant="h6">
                {selectedOption.attackCount}
              </Typography>
            </Paper>
            
            <Paper variant="outlined" sx={{ p: 1, flex: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Deaths
              </Typography>
              <Typography variant="h6" color="error.main">
                {selectedOption.deaths}
              </Typography>
            </Paper>
            
            <Paper variant="outlined" sx={{ p: 1, flex: 1 }}>
              <Typography variant="caption" color="text.secondary">
                Wounded
              </Typography>
              <Typography variant="h6" color="warning.main">
                {selectedOption.wounded}
              </Typography>
            </Paper>
          </Box>
          
          {topGroups.length > 0 && (
            <>
              <Divider sx={{ my: 2 }} />
              <Typography variant="subtitle2" gutterBottom>
                Top Perpetrator Groups
              </Typography>
              
              <List dense>
                {topGroups.map(group => (
                  <ListItem key={group.name} disableGutters>
                    <ListItemText 
                      primary={group.name}
                      secondary={`${group.count} attacks (${Math.round(group.count / selectedOption.attackCount * 100)}%)`}
                    />
                    <Chip 
                      label={group.count}
                      size="small"
                      color={
                        group.count > 50 ? "error" :
                        group.count > 20 ? "warning" :
                        group.count > 5 ? "info" : "default"
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </>
          )}
        </Box>
      )}
    </Paper>
  );
};

export default CountrySelector; 