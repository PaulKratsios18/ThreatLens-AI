import React, { createContext, useContext, useState, ReactNode, useCallback } from 'react';
import { HistoricalAttack } from '../utils/countryUtils';

interface HistoricalDataContextType {
  historicalData: Record<number | 'all', HistoricalAttack[]>;
  loadingYears: Record<number | 'all', boolean>;
  errors: Record<number | 'all', string | null>;
  fetchDataForYear: (year: number | 'all') => Promise<void>;
  hasDataForYear: (year: number | 'all') => boolean;
  hasRealError: (year: number | 'all') => boolean;
}

const HistoricalDataContext = createContext<HistoricalDataContextType | undefined>(undefined);

export const useHistoricalData = () => {
  const context = useContext(HistoricalDataContext);
  if (!context) {
    throw new Error('useHistoricalData must be used within a HistoricalDataProvider');
  }
  return context;
};

interface HistoricalDataProviderProps {
  children: ReactNode;
}

export const HistoricalDataProvider: React.FC<HistoricalDataProviderProps> = ({ children }) => {
  const [historicalData, setHistoricalData] = useState<Record<number | 'all', HistoricalAttack[]>>({} as Record<number | 'all', HistoricalAttack[]>);
  const [loadingYears, setLoadingYears] = useState<Record<number | 'all', boolean>>({} as Record<number | 'all', boolean>);
  const [errors, setErrors] = useState<Record<number | 'all', string | null>>({} as Record<number | 'all', string | null>);
  const [lastFetchTime, setLastFetchTime] = useState<Record<number | 'all', number>>({} as Record<number | 'all', number>);
  
  // Helper function to handle API calls with timeout
  const fetchWithTimeout = async (url: string, options: RequestInit = {}, timeout = 60000) => {
    const controller = new AbortController();
    const id = setTimeout(() => controller.abort(), timeout);
    
    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
      });
      clearTimeout(id);
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      clearTimeout(id);
      if (error instanceof DOMException && error.name === 'AbortError') {
        throw new Error('Request timeout');
      }
      throw error;
    }
  };

  const fetchDataForYear = useCallback(async (year: number | 'all') => {
    // If already loading or we have the data and it was fetched in the last 5 minutes, skip
    const cacheTimeout = 5 * 60 * 1000; // 5 minutes
    const now = Date.now();
    const hasRecentData = lastFetchTime[year] && (now - lastFetchTime[year]) < cacheTimeout;
    
    if (loadingYears[year]) {
      console.log(`Already loading data for year ${year}`);
      return;
    }
    
    if (historicalData[year] && historicalData[year].length > 0 && hasRecentData) {
      console.log(`Using recently cached data for year ${year}`);
      return;
    }

    // Set loading state for this year
    setLoadingYears(prev => ({ ...prev, [year]: true }));
    
    // Clear any previous errors for this year
    if (errors[year]) {
      setErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[year];
        return newErrors;
      });
    }

    try {
      console.log(`Fetching data for year ${year}...`);
      const startTime = Date.now();
      
      // Set a specific timeout for year 2001 (known to be problematic)
      const timeout = year === 2001 ? 90000 : 60000; // 90 seconds for 2001, 60s for others
      
      let data;
      if (year === 'all') {
        // This would fetch all years (implementation depends on your API)
        data = await fetchWithTimeout('http://localhost:8000/historical-data-all', {}, timeout);
      } else {
        data = await fetchWithTimeout(`http://localhost:8000/historical-data?year=${year}`, {}, timeout);
      }
      
      const endTime = Date.now();
      console.log(`Data for year ${year} loaded in ${(endTime - startTime) / 1000}s`);
      
      // Update last fetch time
      setLastFetchTime(prev => ({
        ...prev,
        [year]: now
      }));
      
      // Handle potential error message in the response
      if (data.error) {
        console.warn(`Error in data for year ${year}: ${data.error}`);
        // We'll still process the incidents that were returned, but also store the error
        setErrors(prev => ({
          ...prev,
          [year]: data.error
        }));
      }
      
      // Check if we have incidents
      if (!data.incidents || data.incidents.length === 0) {
        console.warn(`No incidents found for year ${year}`);
        // Generate mock data if we hit a timeout or other issue
        if (year !== 'all' && typeof year === 'number') {
          const mockData = {
            incidents: [
              {
                id: `mock-${year}-1`,
                year: year,
                month: 1,
                day: 1,
                region: "No Data Available",
                country: "No Data Available",
                city: "No data for this year",
                latitude: 0,
                longitude: 0,
                attack_type: "No Data",
                weapon_type: "No Data",
                target_type: "No Data",
                num_killed: 0,
                num_wounded: 0,
                group_name: "No Data Available"
              }
            ]
          };
          
          // Update state with mock data
          setHistoricalData(prev => ({
            ...prev,
            [year]: mockData.incidents
          }));
          
          // Set an error for this year
          setErrors(prev => ({
            ...prev,
            [year]: "No data available for this year or timeout occurred"
          }));
        }
      } else {
        // Process the incidents to ensure we have all required fields
        const processedIncidents = data.incidents.map((incident: any) => ({
          id: incident.id || `unknown-${Math.random()}`,
          year: incident.year || year,
          month: incident.month || 0,
          day: incident.day || 0,
          region: incident.region || "Unknown",
          country: incident.country || "Unknown",
          city: incident.city || "",
          latitude: incident.latitude || 0,
          longitude: incident.longitude || 0,
          attack_type: incident.attack_type || "Unknown",
          weapon_type: incident.weapon_type || "Unknown",
          target_type: incident.target_type || "Unknown",
          num_killed: incident.num_killed || 0,
          num_wounded: incident.num_wounded || 0,
          group_name: incident.group_name || "Unknown"
        }));
      
        // Update state with the new data
        setHistoricalData(prev => ({
          ...prev,
          [year]: processedIncidents
        }));
        
        console.log(`Processed ${processedIncidents.length} incidents for year ${year}`);
      }
    } catch (error) {
      console.error(`Error fetching data for year ${year}:`, error);
      
      // Create some mock data for this error case
      if (year !== 'all' && typeof year === 'number') {
        const errorData = {
          incidents: [
            {
              id: `error-${year}-1`,
              year: year,
              month: 1,
              day: 1,
              region: "Error Loading Data",
              country: "Error",
              city: "Please try again",
              latitude: 0,
              longitude: 0,
              attack_type: "Error",
              weapon_type: "Unknown",
              target_type: "Unknown",
              num_killed: 0,
              num_wounded: 0,
              group_name: `Error: ${error instanceof Error ? error.message : String(error)}`
            }
          ]
        };
        
        // Still set some data so the UI doesn't stay in loading state
        setHistoricalData(prev => ({
          ...prev,
          [year]: errorData.incidents
        }));
      }
      
      // Set error for this year
      setErrors(prev => ({
        ...prev,
        [year]: error instanceof Error ? error.message : String(error)
      }));
    } finally {
      // Clear loading state for this year
      setLoadingYears(prev => ({ ...prev, [year]: false }));
    }
  }, [loadingYears, errors, historicalData, lastFetchTime]);

  // Provide a function to check if data exists for a year
  const hasDataForYear = useCallback((year: number | 'all'): boolean => {
    return Boolean(historicalData[year] && historicalData[year].length > 0);
  }, [historicalData]);
  
  // Provide a function to check if a year has a real error (not just empty data)
  const hasRealError = useCallback((year: number | 'all'): boolean => {
    // Check if there's an error that's not related to "no data available"
    return Boolean(errors[year] && !errors[year]?.includes("No data available"));
  }, [errors]);

  const value = {
    historicalData,
    loadingYears,
    errors,
    fetchDataForYear,
    hasDataForYear,
    hasRealError
  };

  return (
    <HistoricalDataContext.Provider value={value}>
      {children}
    </HistoricalDataContext.Provider>
  );
}; 