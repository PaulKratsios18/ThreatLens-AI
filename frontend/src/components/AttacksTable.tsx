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
  Box,
  Typography,
  Chip,
  Tooltip
} from '@mui/material';
import { Info as InfoIcon } from '@mui/icons-material';
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

  // Standardize country names for all incidents
  const standardizedIncidents = useMemo(() => {
    return incidents.map(incident => ({
      ...incident,
      standardizedCountry: standardizeCountryName(incident.country)
    }));
  }, [incidents]);

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

  // Sort data
  const sortedData = useMemo(() => {
    return [...standardizedIncidents].sort((a, b) => {
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
  }, [standardizedIncidents, orderBy, order]);

  // Format date as MM/DD/YYYY
  const formatDate = (year: number, month: number, day: number) => {
    return `${month.toString().padStart(2, '0')}/${day.toString().padStart(2, '0')}/${year}`;
  };

  return (
    <Paper elevation={3} sx={{ width: '100%', overflow: 'hidden' }}>
      <Box sx={{ p: 2 }}>
        <Typography variant="h6" component="h2" gutterBottom>
          Attack Incidents
          <Chip 
            label={`${incidents.length} incidents`} 
            size="small" 
            sx={{ ml: 2, bgcolor: 'primary.light', color: 'white' }} 
          />
        </Typography>
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
              <TableCell>
                <TableSortLabel
                  active={orderBy === 'num_killed'}
                  direction={orderBy === 'num_killed' ? order : 'asc'}
                  onClick={() => handleRequestSort('num_killed')}
                >
                  Killed
                </TableSortLabel>
              </TableCell>
              <TableCell>
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
            {sortedData
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((incident) => (
                <TableRow hover role="checkbox" tabIndex={-1} key={incident.id}>
                  <TableCell>
                    {incident.month && incident.day 
                      ? formatDate(incident.year, incident.month, incident.day)
                      : incident.year
                    }
                  </TableCell>
                  <TableCell>
                    {incident.standardizedCountry}
                    {incident.standardizedCountry !== incident.country && (
                      <Tooltip title={`Original name: ${incident.country}`}>
                        <InfoIcon fontSize="small" sx={{ ml: 0.5, opacity: 0.6, fontSize: '0.9rem' }} />
                      </Tooltip>
                    )}
                  </TableCell>
                  <TableCell>{incident.city}</TableCell>
                  <TableCell>{incident.region}</TableCell>
                  <TableCell>{incident.attack_type}</TableCell>
                  <TableCell>{incident.weapon_type}</TableCell>
                  <TableCell>{incident.target_type}</TableCell>
                  <TableCell>{incident.num_killed}</TableCell>
                  <TableCell>{incident.num_wounded}</TableCell>
                  <TableCell>{incident.group_name || 'Unknown'}</TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
      </TableContainer>

      <TablePagination
        rowsPerPageOptions={[10, 25, 50, 100]}
        component="div"
        count={incidents.length}
        rowsPerPage={rowsPerPage}
        page={page}
        onPageChange={handleChangePage}
        onRowsPerPageChange={handleChangeRowsPerPage}
      />
    </Paper>
  );
};

export default AttacksTable; 