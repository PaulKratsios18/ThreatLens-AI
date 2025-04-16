import React, { useState } from 'react';
import { FormControl, InputLabel, Select, MenuItem, SelectChangeEvent, Box, Typography } from '@mui/material';

interface TimelineSliderProps {
  selectedYear: number;
  onYearChange: (year: number) => void;
  years?: number[];
}

const TimelineSlider: React.FC<TimelineSliderProps> = ({ 
  selectedYear, 
  onYearChange, 
  years = [2023, 2024, 2025] 
}) => {
  const handleChange = (event: SelectChangeEvent<number>) => {
    onYearChange(Number(event.target.value));
  };

  // Group years into historical and future
  const historicalYears = years.filter(year => year <= 2022);
  const futureYears = years.filter(year => year >= 2023);

  return (
    <Box sx={{ 
      mt: 2, 
      px: 2, 
      py: 2, 
      bgcolor: 'background.paper', 
      borderRadius: 1,
      boxShadow: 1 
    }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6" component="h2">Timeline</Typography>
        <Typography variant="subtitle1" color="text.secondary">
          {selectedYear <= 2022 ? 'Historical Data' : 'Future Predictions'}
        </Typography>
      </Box>
      
      <FormControl fullWidth>
        <InputLabel id="year-select-label">Year</InputLabel>
        <Select
          labelId="year-select-label"
          id="year-select"
          value={selectedYear}
          label="Year"
          onChange={handleChange}
        >
          {/* Option for historical years */}
          {historicalYears.length > 0 && (
            [
              <MenuItem key="historical-group" disabled sx={{ fontWeight: 'bold', color: 'text.primary' }}>
                Historical (2000-2022)
              </MenuItem>,
              ...historicalYears
                .sort((a, b) => b - a) // Sort in descending order
                .map(year => (
                  <MenuItem key={year} value={year}>
                    {year}
                  </MenuItem>
                ))
            ]
          )}
          
          {/* Divider between historical and future years */}
          {historicalYears.length > 0 && futureYears.length > 0 && (
            <MenuItem key="divider" disabled sx={{ borderTop: '1px solid #ddd', margin: '8px 0' }} />
          )}
          
          {/* Option for future years */}
          {futureYears.length > 0 && (
            [
              <MenuItem key="future-group" disabled sx={{ fontWeight: 'bold', color: 'text.primary' }}>
                Future Predictions (2023-2025)
              </MenuItem>,
              ...futureYears
                .sort((a, b) => a - b) // Sort in ascending order
                .map(year => (
                  <MenuItem key={year} value={year}>
                    {year}
                  </MenuItem>
                ))
            ]
          )}
        </Select>
      </FormControl>
    </Box>
  );
};

export default TimelineSlider; 