import React, { useState, useEffect, useMemo } from 'react';
import { Bar, Pie, Line } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';
import HistoricalMap from './HistoricalMap';
import AttacksTable from './AttacksTable';
import { HistoricalAttack } from '../utils/countryUtils';
import CountrySelector from './CountrySelector';
import { useHistoricalData } from '../contexts/HistoricalDataContext';
import { 
  Box, 
  Typography, 
  Paper, 
  Grid, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  TextField,
  Chip,
  IconButton,
  Tooltip,
  Collapse,
  Button,
  Divider
} from '@mui/material';
import { ExpandMore, ExpandLess } from '@mui/icons-material';

Chart.register(...registerables);

const HistoricalDashboard: React.FC = () => {
  // Generate an array of all years from 1970 to 2021
  const availableYears = useMemo(() => {
    const years = [];
    for (let year = 1970; year <= 2021; year++) {
      years.push(year);
    }
    return years;
  }, []);
  
  // Basic state
  const [selectedYear, setSelectedYear] = useState<number | 'all'>(2021);
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState<boolean>(true);
  
  // Filter state
  const [filters, setFilters] = useState({
    region: '',
    country: '',
    attackType: '',
    weaponType: '',
    targetType: '',
    groupName: '',
    minCasualties: 0,
    city: '',
    searchTerm: ''
  });
  
  // Use the cached historical data context
  const { historicalData, loadingYears, errors, fetchDataForYear } = useHistoricalData();
  
  useEffect(() => {
    // Fetch data for the selected year if not already cached
    if (selectedYear === 'all') {
      // Fetch data for all years
      availableYears.forEach(year => fetchDataForYear(year));
    } else {
      fetchDataForYear(selectedYear);
    }
  }, [selectedYear, fetchDataForYear, availableYears]);
  
  // Get loading and error state
  const loading = selectedYear === 'all' 
    ? availableYears.some(year => loadingYears[year])
    : loadingYears[selectedYear as number] || false;
  
  const error = selectedYear === 'all'
    ? availableYears.map(year => errors[year]).filter(Boolean)[0] || null
    : errors[selectedYear as number] || null;
  
  // Get data based on selected year(s)
  const rawYearData = useMemo(() => {
    if (selectedYear === 'all') {
      // Combine data from all years
      return Object.values(historicalData).flat();
    }
    return historicalData[selectedYear as number] || [];
  }, [historicalData, selectedYear]);

  // Apply filters to the data
  const filteredData = useMemo(() => {
    return rawYearData.filter(incident => {
      const matchesRegion = !filters.region || incident.region === filters.region;
      const matchesCountry = !filters.country || incident.country === filters.country;
      const matchesAttackType = !filters.attackType || incident.attack_type === filters.attackType;
      const matchesWeaponType = !filters.weaponType || incident.weapon_type === filters.weaponType;
      const matchesTargetType = !filters.targetType || incident.target_type === filters.targetType;
      const matchesGroupName = !filters.groupName || incident.group_name === filters.groupName;
      const matchesCasualties = (incident.num_killed + incident.num_wounded) >= filters.minCasualties;
      const matchesCity = !filters.city || 
        incident.city.toLowerCase().includes(filters.city.toLowerCase());
      
      const matchesSearch = !filters.searchTerm || 
        incident.country.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
        incident.city.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
        incident.attack_type.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
        (incident.group_name && incident.group_name.toLowerCase().includes(filters.searchTerm.toLowerCase()));
        
      return matchesRegion && matchesCountry && matchesAttackType && 
             matchesWeaponType && matchesTargetType && matchesGroupName && 
             matchesCasualties && matchesCity && matchesSearch;
    });
  }, [rawYearData, filters]);

  // Extract unique values for filter dropdowns
  const uniqueRegions = useMemo(() => 
    [...new Set(rawYearData.map(i => i.region))].sort(), 
    [rawYearData]
  );
  
  const uniqueCountries = useMemo(() => 
    [...new Set(rawYearData.map(i => i.country))].sort(), 
    [rawYearData]
  );
  
  const uniqueAttackTypes = useMemo(() => 
    [...new Set(rawYearData.map(i => i.attack_type))].sort(), 
    [rawYearData]
  );
  
  const uniqueWeaponTypes = useMemo(() => 
    [...new Set(rawYearData.map(i => i.weapon_type))].sort(), 
    [rawYearData]
  );
  
  const uniqueTargetTypes = useMemo(() => 
    [...new Set(rawYearData.map(i => i.target_type))].sort(), 
    [rawYearData]
  );
  
  const uniqueGroups = useMemo(() => 
    [...new Set(rawYearData.map(i => i.group_name || 'Unknown'))].sort(), 
    [rawYearData]
  );

  // Handle filter changes
  const handleFilterChange = (name: keyof typeof filters, value: any) => {
    setFilters(prev => ({ ...prev, [name]: value }));
  };

  // Calculate statistics for visualizations
  const attacksByRegion = useMemo(() => 
    filteredData.reduce((acc, attack) => {
      acc[attack.region] = (acc[attack.region] || 0) + 1;
      return acc;
    }, {} as Record<string, number>),
    [filteredData]
  );
  
  const attacksByType = useMemo(() => 
    filteredData.reduce((acc, attack) => {
      acc[attack.attack_type] = (acc[attack.attack_type] || 0) + 1;
      return acc;
    }, {} as Record<string, number>),
    [filteredData]
  );
  
  const attacksByMonth = useMemo(() => {
    const months = Array(12).fill(0);
    filteredData.forEach(attack => {
      if (attack.month > 0 && attack.month <= 12) {
        months[attack.month - 1]++;
      }
    });
    return months;
  }, [filteredData]);

  // Calculate attacks by year if showing all years
  const attacksByYear = useMemo(() => {
    if (selectedYear === 'all') {
      return filteredData.reduce((acc, attack) => {
        acc[attack.year] = (acc[attack.year] || 0) + 1;
        return acc;
      }, {} as Record<number, number>);
    }
    return null;
  }, [filteredData, selectedYear]);

  // Prepare chart data
  const regionChartData = {
    labels: Object.keys(attacksByRegion),
    datasets: [
      {
        label: 'Number of Attacks',
        data: Object.values(attacksByRegion),
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
        borderColor: 'rgba(54, 162, 235, 1)',
        borderWidth: 1,
      },
    ],
  };

  const typeChartData = {
    labels: Object.keys(attacksByType),
    datasets: [
      {
        label: 'Types of Attacks',
        data: Object.values(attacksByType),
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

  const monthlyChartData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    datasets: [
      {
        label: 'Monthly Attacks',
        data: attacksByMonth,
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1,
      },
    ],
  };

  const yearlyChartData = attacksByYear ? {
    labels: Object.keys(attacksByYear).sort(),
    datasets: [
      {
        label: 'Yearly Attacks',
        data: Object.keys(attacksByYear).sort().map(year => attacksByYear[parseInt(year)]),
        backgroundColor: 'rgba(153, 102, 255, 0.5)',
        borderColor: 'rgba(153, 102, 255, 1)',
        borderWidth: 1,
      },
    ],
  } : null;

  if (loading) {
    return <div className="flex items-center justify-center h-96">Loading historical data...</div>;
  }

  if (error) {
    return <div className="text-red-600">Error: {error}</div>;
  }

  const yearDisplay = selectedYear === 'all' ? 'All Years' : selectedYear;

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <Typography variant="h4" className="mb-6">Historical Data Analysis</Typography>
      
      {/* Main Filter Bar */}
      <Paper elevation={2} className="mb-6 p-4">
        <div className="flex justify-between items-center mb-4">
          <Typography variant="h6">Data Filters</Typography>
          <div className="flex gap-2">
            <Chip 
              label={`${filteredData.length} incidents`} 
              color="primary" 
              variant="outlined"
            />
            <IconButton 
              color={showFilters ? "primary" : "default"}
              onClick={() => setShowFilters(!showFilters)}
              size="small"
            >
              {showFilters ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          </div>
        </div>

        <Collapse in={showFilters}>
          <Grid container spacing={3}>
            {/* First row of filters */}
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Year</InputLabel>
                <Select
                  value={selectedYear}
                  label="Year"
                  onChange={(e) => setSelectedYear(e.target.value as number | 'all')}
                >
                  <MenuItem value="all">All Years</MenuItem>
                  {availableYears.map((year) => (
                    <MenuItem key={year} value={year}>{year}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Region</InputLabel>
                <Select
                  value={filters.region}
                  label="Region"
                  onChange={(e) => handleFilterChange('region', e.target.value)}
                >
                  <MenuItem value="">All Regions</MenuItem>
                  {uniqueRegions.map((region) => (
                    <MenuItem key={region} value={region}>{region}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Country</InputLabel>
                <Select
                  value={filters.country}
                  label="Country"
                  onChange={(e) => {
                    handleFilterChange('country', e.target.value);
                    setSelectedCountry(e.target.value as string || null);
                  }}
                >
                  <MenuItem value="">All Countries</MenuItem>
                  {uniqueCountries.map((country) => (
                    <MenuItem key={country} value={country}>{country}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Attack Type</InputLabel>
                <Select
                  value={filters.attackType}
                  label="Attack Type"
                  onChange={(e) => handleFilterChange('attackType', e.target.value)}
                >
                  <MenuItem value="">All Attack Types</MenuItem>
                  {uniqueAttackTypes.map((type) => (
                    <MenuItem key={type} value={type}>{type}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            {/* Second row of filters */}
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Weapon Type</InputLabel>
                <Select
                  value={filters.weaponType}
                  label="Weapon Type"
                  onChange={(e) => handleFilterChange('weaponType', e.target.value)}
                >
                  <MenuItem value="">All Weapon Types</MenuItem>
                  {uniqueWeaponTypes.map((type) => (
                    <MenuItem key={type} value={type}>{type}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Target Type</InputLabel>
                <Select
                  value={filters.targetType}
                  label="Target Type"
                  onChange={(e) => handleFilterChange('targetType', e.target.value)}
                >
                  <MenuItem value="">All Target Types</MenuItem>
                  {uniqueTargetTypes.map((type) => (
                    <MenuItem key={type} value={type}>{type}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Group</InputLabel>
                <Select
                  value={filters.groupName}
                  label="Group"
                  onChange={(e) => handleFilterChange('groupName', e.target.value)}
                >
                  <MenuItem value="">All Groups</MenuItem>
                  {uniqueGroups.map((group) => (
                    <MenuItem key={group} value={group}>{group}</MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <TextField
                label="Min. Casualties"
                type="number"
                variant="outlined"
                size="small"
                fullWidth
                value={filters.minCasualties}
                onChange={(e) => handleFilterChange('minCasualties', parseInt(e.target.value) || 0)}
                InputProps={{ inputProps: { min: 0 } }}
              />
            </Grid>
            
            <Grid item xs={12} sm={6} md={4} lg={3}>
              <TextField
                label="City"
                variant="outlined"
                size="small"
                fullWidth
                value={filters.city}
                onChange={(e) => handleFilterChange('city', e.target.value)}
              />
            </Grid>
          </Grid>
          
          <Box display="flex" justifyContent="flex-end" mt={2}>
            <Button 
              variant="outlined" 
              onClick={() => {
                setFilters({
                  region: '',
                  country: '',
                  attackType: '',
                  weaponType: '',
                  targetType: '',
                  groupName: '',
                  minCasualties: 0,
                  city: '',
                  searchTerm: ''
                });
                setSelectedCountry(null);
              }}
            >
              Clear Filters
            </Button>
          </Box>
        </Collapse>
      </Paper>
      
      {/* Map Section - Always visible */}
      <Paper elevation={2} className="mb-6 p-4">
        <Box className="mb-4">
          <Typography variant="h6" gutterBottom>
            Historical Incidents Map ({yearDisplay})
            {selectedCountry && (
              <Button 
                size="small"
                color="primary"
                className="ml-2"
                onClick={() => {
                  setSelectedCountry(null);
                  setFilters(prev => ({ ...prev, country: '' }));
                }}
              >
                Clear Selection
              </Button>
            )}
          </Typography>
          <Typography variant="body2" color="textSecondary" className="mb-4">
            {filteredData.length === 0 
              ? 'No incidents match your search criteria.' 
              : `Showing ${filteredData.length} incidents on the map.`}
          </Typography>
        </Box>
        <HistoricalMap 
          incidents={filteredData} 
          selectedCountry={selectedCountry}
        />
      </Paper>
      
      {/* Incidents Table */}
      <Paper elevation={2} className="mb-6">
        <AttacksTable incidents={filteredData} />
      </Paper>
      
      {/* Charts and Visualizations */}
      {filteredData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {selectedYear === 'all' && yearlyChartData && (
            <Paper elevation={2} className="p-4 lg:col-span-2">
              <Typography variant="h6" gutterBottom>Attacks by Year</Typography>
              <Bar data={yearlyChartData} />
            </Paper>
          )}
          
          <Paper elevation={2} className="p-4">
            <Typography variant="h6" gutterBottom>Attacks by Month {selectedYear !== 'all' ? `(${selectedYear})` : ''}</Typography>
            <Line data={monthlyChartData} />
          </Paper>
          
          <Paper elevation={2} className="p-4">
            <Typography variant="h6" gutterBottom>Attack Types Distribution</Typography>
            <Pie data={typeChartData} />
          </Paper>
          
          <Paper elevation={2} className="p-4 lg:col-span-2">
            <Typography variant="h6" gutterBottom>Attacks by Region</Typography>
            <Bar 
              data={regionChartData} 
              options={{ 
                indexAxis: 'y',
                plugins: {
                  legend: {
                    display: false,
                  }
                }
              }} 
            />
          </Paper>
        </div>
      )}
      
      {/* Country Details when a country is selected */}
      {selectedCountry && (
        <Paper elevation={2} className="mb-6 p-4">
          <CountrySelector
            incidents={filteredData}
            selectedCountry={selectedCountry}
            onChange={(country) => {
              setSelectedCountry(country);
              setFilters(prev => ({ ...prev, country: country || '' }));
            }}
          />
        </Paper>
      )}
    </div>
  );
};

export default HistoricalDashboard; 