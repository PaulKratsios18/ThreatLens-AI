import React, { useState, useEffect, useMemo } from 'react';
import { Bar, Pie, Line } from 'react-chartjs-2';
import { Chart, registerables } from 'chart.js';
import HistoricalMap from './HistoricalMap';
import AttacksTable from './AttacksTable';
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
  Collapse,
  Button,
  Alert,
  AlertTitle,
  Skeleton,
  Stack,
  CircularProgress,
  Card,
  CardContent,
  CardHeader
} from '@mui/material';
import { ExpandMore, ExpandLess } from '@mui/icons-material';

Chart.register(...registerables);

const HistoricalDashboard: React.FC = () => {
  const [selectedYear, setSelectedYear] = useState<number | 'all'>('all');
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
  const [showFilters, setShowFilters] = useState<boolean>(true);
  
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
  
  const { historicalData, loadingYears, errors, fetchDataForYear } = useHistoricalData();

  // Add state to track loading status message
  const [loadingMessage, setLoadingMessage] = useState<string>('');
  
  // Define available years in descending order
  // Years are sorted in descending order
  const availableYears = useMemo(() => {
    // Years from 2021 down to 1970 in descending order
    const years = Array.from({ length: 52 }, (_, i) => 2021 - i);
    return years;
  }, []);
  
  // Handle special case for year 2001
  useEffect(() => {
    if (selectedYear === 2001) {
      setLoadingMessage('Loading data for 2001 may take longer than usual. Please be patient...');
      
      // Set a timeout to update the message after 20 seconds
      const timeoutId = setTimeout(() => {
        setLoadingMessage('Still loading 2001 data... (this year has a large dataset)');
      }, 20000);
      
      // Set another timeout for a longer wait
      const timeoutId2 = setTimeout(() => {
        setLoadingMessage('Taking longer than expected. You may want to try a different year if this continues...');
      }, 45000);
      
      return () => {
        clearTimeout(timeoutId);
        clearTimeout(timeoutId2);
        setLoadingMessage('');
      };
    }
  }, [selectedYear]);

  // Fetch data for the selected year
  useEffect(() => {
    if (selectedYear !== 'all') {
      // If we don't have data for this year and it's not already loading, fetch it
      if (!historicalData[selectedYear] && (!loadingYears || !loadingYears[selectedYear])) {
        console.log(`Initiating fetch for year ${selectedYear}`);
        fetchDataForYear(selectedYear);
      }
    } else {
      // If 'all years' is selected, make sure we load some common years
      const recentYears = [2014, 2015, 2016, 2017, 2018, 2019, 2020];
      recentYears.forEach(year => {
        if (!historicalData[year] && (!loadingYears || !loadingYears[year])) {
          console.log(`For 'all' selection, initiating fetch for year ${year}`);
          fetchDataForYear(year);
        }
      });
    }
  }, [selectedYear, historicalData, loadingYears, fetchDataForYear]);
  
  // Get loading and error state
  const loading = selectedYear === 'all' 
    ? (availableYears.some(year => loadingYears && year in loadingYears && loadingYears[year]))
    : (loadingYears && selectedYear in loadingYears && loadingYears[selectedYear]) || false;
  
  const error = selectedYear === 'all'
    ? availableYears.map(year => errors && errors[year]).filter(Boolean)[0] || null
    : (errors && errors[selectedYear]) || null;
  
  // Get data based on selected year(s)
  const rawYearData = useMemo(() => {
    if (selectedYear === 'all') {
      // Combine data from all available years
      return Object.entries(historicalData)
        .filter(([key, _]) => key !== 'all' && !isNaN(Number(key)))
        .flatMap(([_, data]) => data);
    }
    return historicalData[selectedYear] || [];
  }, [historicalData, selectedYear]);

  // Get the data for the selected year, with filtering
  const filteredData = useMemo(() => {
    // If we're loading the data for this year, return an empty array
  if (loading) {
      return [];
    }
    
    // Get the data for the selected year
    let data = rawYearData;
    
    // Apply filters
    if (filters.region) {
      data = data.filter(attack => attack.region === filters.region);
    }
    
    if (filters.country || selectedCountry) {
      const country = filters.country || selectedCountry;
      data = data.filter(attack => attack.country === country);
    }
    
    if (filters.attackType) {
      data = data.filter(attack => attack.attack_type === filters.attackType);
    }
    
    if (filters.weaponType) {
      data = data.filter(attack => attack.weapon_type === filters.weaponType);
    }
    
    if (filters.targetType) {
      data = data.filter(attack => attack.target_type === filters.targetType);
    }
    
    if (filters.groupName) {
      data = data.filter(attack => attack.group_name === filters.groupName);
    }
    
    if (filters.minCasualties > 0) {
      data = data.filter(attack => 
        (attack.num_killed || 0) + (attack.num_wounded || 0) >= filters.minCasualties
      );
    }
    
    if (filters.city) {
      data = data.filter(attack => 
        attack.city && attack.city.toLowerCase().includes(filters.city.toLowerCase())
      );
    }
    
    return data;
  }, [rawYearData, filters, selectedCountry, loading]);

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
  const handleFilterChange = (name: keyof typeof filters, value: string | number) => {
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
    return (
      <Box p={4}>
        <Typography variant="h4" component="h1" sx={{ mb: 4 }}>Historical Data Dashboard</Typography>
        
        {/* Display error message if there is one */}
        {selectedYear !== 'all' && errors && selectedYear in errors && errors[selectedYear] && (
          <Alert severity="warning" sx={{ mb: 4 }}>
            <AlertTitle>Data Loading Issue</AlertTitle>
            {errors[selectedYear]}
          </Alert>
        )}
        
        {/* Display loading message if applicable */}
        {selectedYear !== 'all' && loadingYears && selectedYear in loadingYears && loadingYears[selectedYear] && (
          <Box sx={{ mb: 4 }}>
            <Skeleton height="20px" width="100%" sx={{ mb: 2 }} />
            <Skeleton height="20px" width="80%" sx={{ mb: 2 }} />
            <Typography sx={{ color: "orange" }}>{loadingMessage}</Typography>
          </Box>
        )}
        
        <Box sx={{ display: 'flex', mb: 4, flexWrap: 'wrap', gap: 4 }}>
          {/* Year Filter Dropdown */}
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel id="year-select-label">Year</InputLabel>
            <Select 
              labelId="year-select-label"
              value={selectedYear === 'all' ? 'all' : selectedYear.toString()} 
              onChange={(e) => {
                const value = e.target.value;
                setSelectedYear(value === 'all' ? 'all' : parseInt(value));
              }}
              label="Year"
            >
              <MenuItem value="all">All Years</MenuItem>
              {availableYears.map(year => (
                <MenuItem key={year} value={year.toString()}>
                  {year} {year === 2001 ? '(Large Dataset)' : ''}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
          
          {/* Country Filter */}
          {/* ... existing country filter ... */}
        </Box>
        
        {/* Show loading indicator when data is being fetched */}
        {selectedYear !== 'all' && loadingYears && selectedYear in loadingYears && loadingYears[selectedYear] ? (
          <Box sx={{ height: '200px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <Stack spacing={2} alignItems="center">
              <CircularProgress size={60} thickness={4} />
              <Typography>{loadingMessage || 'Loading data...'}</Typography>
            </Stack>
          </Box>
        ) : (
          // Rest of the dashboard
          <Grid container spacing={3}>
            {/* ... existing dashboard components ... */}
          </Grid>
        )}
      </Box>
    );
  }

  const yearDisplay = selectedYear === 'all' ? 'All Years' : selectedYear;

  // Show a message when no data is available even after loading
  const showEmptyState = !loading && filteredData.length === 0;

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
        
        {/* Error or loading states */}
        {loading ? (
          <Box sx={{ height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <Stack spacing={2} alignItems="center">
              <CircularProgress size={60} thickness={4} />
              <Typography>{loadingMessage || `Loading data for ${yearDisplay}...`}</Typography>
            </Stack>
          </Box>
        ) : error ? (
          <Alert severity="error" sx={{ mb: 3 }}>
            <AlertTitle>Error Loading Data</AlertTitle>
            {error}
          </Alert>
        ) : showEmptyState ? (
          <Alert severity="info" sx={{ mb: 3 }}>
            <AlertTitle>No Data Available</AlertTitle>
            {selectedYear === 'all' 
              ? 'No historical data found for any years with the current filters. Try adjusting your filters.' 
              : `No historical data found for ${selectedYear} with the current filters. Try selecting a different year or adjusting your filters.`}
          </Alert>
        ) : (
          <HistoricalMap 
            incidents={filteredData} 
            selectedCountry={selectedCountry}
          />
        )}
      </Paper>
      
      {/* Incidents Table */}
      <Paper elevation={2} className="mb-6">
        {loading ? (
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Attack Incidents</Typography>
            <Stack spacing={1}>
              <Skeleton height="40px" />
              <Skeleton height="40px" />
              <Skeleton height="40px" />
              <Skeleton height="40px" />
            </Stack>
          </Box>
        ) : showEmptyState ? (
          <Box sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>Attack Incidents</Typography>
            <Typography color="text.secondary">No incidents to display.</Typography>
          </Box>
        ) : (
          <AttacksTable incidents={filteredData} />
        )}
      </Paper>
      
      {/* Charts and Visualizations */}
      {!loading && filteredData.length > 0 && (
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