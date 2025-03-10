import React, { useState, useMemo } from 'react';
import {
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  TablePagination,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Typography,
  Chip,
  Grid,
  InputAdornment,
  IconButton,
  Tooltip
} from '@mui/material';
import { Search as SearchIcon, FilterList as FilterIcon, Info as InfoIcon } from '@mui/icons-material';
import { HistoricalAttack, standardizeCountryName } from '../utils/countryUtils';

type SortableColumn = 'date' | 'country' | 'region' | 'attack_type' | 'weapon_type' | 'target_type' | 'num_killed' | 'num_wounded' | 'group_name';

interface AttacksTableProps {
  incidents: HistoricalAttack[];
}

const AttacksTable: React.FC<AttacksTableProps> = ({ incidents }) => {
  // Table state
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [orderBy, setOrderBy] = useState<SortableColumn>('date');
  const [order, setOrder] = useState<'asc' | 'desc'>('desc');
  
  // Filter state
  const [filters, setFilters] = useState({
    search: '',
    region: '',
    country: '',
    attackType: '',
    weaponType: '',
    targetType: '',
    minCasualties: 0,
    groupName: ''
  });
  const [showFilters, setShowFilters] = useState(false);

  // Standardize country names for all incidents
  const standardizedIncidents = useMemo(() => {
    return incidents.map(incident => ({
      ...incident,
      standardizedCountry: standardizeCountryName(incident.country)
    }));
  }, [incidents]);

  // Get unique values for filter dropdowns with standardized country names
  const uniqueRegions = useMemo(() => 
    [...new Set(standardizedIncidents.map(i => i.region))].sort(), 
    [standardizedIncidents]
  );
  
  const uniqueCountries = useMemo(() => {
    // Create a Map to deduplicate countries while keeping original and standardized names together
    const countryMap = new Map();
    standardizedIncidents.forEach(incident => {
      if (!countryMap.has(incident.standardizedCountry)) {
        countryMap.set(incident.standardizedCountry, {
          original: incident.country,
          standardized: incident.standardizedCountry
        });
      }
    });
    
    // Convert to array and sort
    return Array.from(countryMap.values())
      .sort((a, b) => a.standardized.localeCompare(b.standardized));
  }, [standardizedIncidents]);
  
  const uniqueAttackTypes = useMemo(() => 
    [...new Set(standardizedIncidents.map(i => i.attack_type))].sort(), 
    [standardizedIncidents]
  );
  
  const uniqueWeaponTypes = useMemo(() => 
    [...new Set(standardizedIncidents.map(i => i.weapon_type))].sort(), 
    [standardizedIncidents]
  );
  
  const uniqueTargetTypes = useMemo(() => 
    [...new Set(standardizedIncidents.map(i => i.target_type))].sort(), 
    [standardizedIncidents]
  );

  // Add unique group names for filter dropdown
  const uniqueGroups = useMemo(() => 
    [...new Set(standardizedIncidents.map(i => i.group_name || 'Unknown'))].sort(), 
    [standardizedIncidents]
  );

  // Handle sort change
  const handleRequestSort = (property: SortableColumn) => {
    const isAsc = orderBy === property && order === 'asc';
    setOrder(isAsc ? 'desc' : 'asc');
    setOrderBy(property);
  };

  // Handle page change
  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  // Handle rows per page change
  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  // Handle filter changes
  const handleFilterChange = (name: keyof typeof filters, value: any) => {
    setFilters({ ...filters, [name]: value });
    setPage(0); // Reset to first page when filters change
  };

  // Apply filters and sort data
  const filteredData = useMemo(() => {
    return standardizedIncidents
      .filter(incident => {
        const searchMatch = filters.search === '' ||
          incident.country.toLowerCase().includes(filters.search.toLowerCase()) ||
          incident.standardizedCountry.toLowerCase().includes(filters.search.toLowerCase()) ||
          incident.city.toLowerCase().includes(filters.search.toLowerCase()) ||
          incident.attack_type.toLowerCase().includes(filters.search.toLowerCase()) ||
          incident.target_type.toLowerCase().includes(filters.search.toLowerCase()) ||
          (incident.group_name && incident.group_name.toLowerCase().includes(filters.search.toLowerCase()));

        const regionMatch = filters.region === '' || incident.region === filters.region;
        const countryMatch = filters.country === '' || 
          incident.standardizedCountry === filters.country;
        const attackTypeMatch = filters.attackType === '' || incident.attack_type === filters.attackType;
        const weaponTypeMatch = filters.weaponType === '' || incident.weapon_type === filters.weaponType;
        const targetTypeMatch = filters.targetType === '' || incident.target_type === filters.targetType;
        const casualtiesMatch = (incident.num_killed + incident.num_wounded) >= filters.minCasualties;
        const groupMatch = filters.groupName === '' || incident.group_name === filters.groupName;

        return searchMatch && regionMatch && countryMatch && attackTypeMatch &&
               weaponTypeMatch && targetTypeMatch && casualtiesMatch && groupMatch;
      })
      .sort((a, b) => {
        let comparison = 0;
        
        if (orderBy === 'date') {
          const dateA = new Date(a.year, a.month - 1, a.day);
          const dateB = new Date(b.year, b.month - 1, b.day);
          comparison = dateA.getTime() - dateB.getTime();
        } else if (orderBy === 'country') {
          comparison = a.standardizedCountry.localeCompare(b.standardizedCountry);
        } else if (orderBy === 'region' || orderBy === 'attack_type' || 
                  orderBy === 'weapon_type' || orderBy === 'target_type' || orderBy === 'group_name') {
          // Handle potential undefined group_name
          const aValue = orderBy === 'group_name' ? (a[orderBy] || 'Unknown') : a[orderBy];
          const bValue = orderBy === 'group_name' ? (b[orderBy] || 'Unknown') : b[orderBy];
          comparison = aValue.localeCompare(bValue);
        } else {
          comparison = (a[orderBy] || 0) - (b[orderBy] || 0);
        }
        
        return order === 'asc' ? comparison : -comparison;
      });
  }, [standardizedIncidents, filters, orderBy, order]);

  // Format date as MM/DD/YYYY
  const formatDate = (year: number, month: number, day: number) => {
    return `${month.toString().padStart(2, '0')}/${day.toString().padStart(2, '0')}/${year}`;
  };

  return (
    <Paper elevation={3} sx={{ width: '100%', overflow: 'hidden', mt: 3 }}>
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" component="h2" gutterBottom>
          Attack Incidents
          <Chip 
            label={`${filteredData.length} incidents`} 
            size="small" 
            sx={{ ml: 2, bgcolor: 'primary.light', color: 'white' }} 
          />
        </Typography>
        
        <Box sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
          <TextField
            label="Search"
            variant="outlined"
            size="small"
            value={filters.search}
            onChange={(e) => handleFilterChange('search', e.target.value)}
            sx={{ mr: 2, flex: 1 }}
            InputProps={{
              startAdornment: (
                <InputAdornment position="start">
                  <SearchIcon fontSize="small" />
                </InputAdornment>
              ),
            }}
          />
          <IconButton onClick={() => setShowFilters(!showFilters)} color={showFilters ? "primary" : "default"}>
            <FilterIcon />
          </IconButton>
        </Box>
        
        {showFilters && (
          <Grid container spacing={2} sx={{ mb: 2 }}>
            <Grid item xs={12} sm={6} md={3}>
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
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth size="small">
                <InputLabel>Country</InputLabel>
                <Select
                  value={filters.country}
                  label="Country"
                  onChange={(e) => handleFilterChange('country', e.target.value)}
                >
                  <MenuItem value="">All Countries</MenuItem>
                  {uniqueCountries.map((country) => (
                    <MenuItem key={country.standardized} value={country.standardized}>
                      {country.standardized}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
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
            <Grid item xs={12} sm={6} md={3}>
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
            <Grid item xs={12} sm={6} md={3}>
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
            <Grid item xs={12} sm={6} md={3}>
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
            <Grid item xs={12} sm={6} md={3}>
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
          </Grid>
        )}
      </Box>
      
      <TableContainer sx={{ maxHeight: 600 }}>
        <Table stickyHeader aria-label="sticky table" size="small">
          <TableHead>
            <TableRow>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'date'}
                  direction={orderBy === 'date' ? order : 'asc'}
                  onClick={() => handleRequestSort('date')}
                >
                  Date
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'country'}
                  direction={orderBy === 'country' ? order : 'asc'}
                  onClick={() => handleRequestSort('country')}
                >
                  Country
                  <Tooltip title="Country names are standardized for consistent mapping">
                    <InfoIcon fontSize="small" sx={{ ml: 0.5, opacity: 0.6, fontSize: '0.9rem' }} />
                  </Tooltip>
                </TableSortLabel>
              </TableCell>
              <TableCell>City</TableCell>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'region'}
                  direction={orderBy === 'region' ? order : 'asc'}
                  onClick={() => handleRequestSort('region')}
                >
                  Region
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'attack_type'}
                  direction={orderBy === 'attack_type' ? order : 'asc'}
                  onClick={() => handleRequestSort('attack_type')}
                >
                  Attack Type
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'weapon_type'}
                  direction={orderBy === 'weapon_type' ? order : 'asc'}
                  onClick={() => handleRequestSort('weapon_type')}
                >
                  Weapon
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'target_type'}
                  direction={orderBy === 'target_type' ? order : 'asc'}
                  onClick={() => handleRequestSort('target_type')}
                >
                  Target
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">
                <TableSortLabel
                  active={orderBy === 'num_killed'}
                  direction={orderBy === 'num_killed' ? order : 'asc'}
                  onClick={() => handleRequestSort('num_killed')}
                >
                  Killed
                </TableSortLabel>
              </TableCell>
              <TableCell align="right">
                <TableSortLabel
                  active={orderBy === 'num_wounded'}
                  direction={orderBy === 'num_wounded' ? order : 'asc'}
                  onClick={() => handleRequestSort('num_wounded')}
                >
                  Wounded
                </TableSortLabel>
              </TableCell>
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'group_name'}
                  direction={orderBy === 'group_name' ? order : 'asc'}
                  onClick={() => handleRequestSort('group_name')}
                >
                  Group
                </TableSortLabel>
              </TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {filteredData
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((row) => (
                <TableRow
                  hover
                  key={row.id}
                  sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                >
                  <TableCell>{formatDate(row.year, row.month, row.day)}</TableCell>
                  <TableCell>
                    {row.standardizedCountry !== row.country ? (
                      <Tooltip title={`Original: ${row.country}`}>
                        <span>{row.standardizedCountry}</span>
                      </Tooltip>
                    ) : (
                      row.country
                    )}
                  </TableCell>
                  <TableCell>{row.city}</TableCell>
                  <TableCell>{row.region}</TableCell>
                  <TableCell>{row.attack_type}</TableCell>
                  <TableCell>{row.weapon_type}</TableCell>
                  <TableCell>{row.target_type}</TableCell>
                  <TableCell align="right">
                    {row.num_killed > 0 ? (
                      <Chip 
                        label={row.num_killed} 
                        size="small" 
                        color={row.num_killed > 10 ? "error" : row.num_killed > 1 ? "warning" : "success"}
                        sx={{ minWidth: 40 }}
                      />
                    ) : (
                      "0"
                    )}
                  </TableCell>
                  <TableCell align="right">
                    {row.num_wounded > 0 ? (
                      <Chip 
                        label={row.num_wounded} 
                        size="small" 
                        color={row.num_wounded > 20 ? "error" : row.num_wounded > 5 ? "warning" : "info"}
                        sx={{ minWidth: 40 }}
                      />
                    ) : (
                      "0"
                    )}
                  </TableCell>
                  <TableCell>{row.group_name || 'Unknown'}</TableCell>
                </TableRow>
              ))}
            {filteredData.length === 0 && (
              <TableRow>
                <TableCell colSpan={10} align="center" sx={{ py: 3 }}>
                  No incidents match your search criteria
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </Table>
      </TableContainer>
      <TablePagination
        rowsPerPageOptions={[10, 25, 50, 100]}
        component="div"
        count={filteredData.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </Paper>
  );
};

export default AttacksTable; 