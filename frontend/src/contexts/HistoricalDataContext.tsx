import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { HistoricalAttack } from '../utils/countryUtils';

interface HistoricalDataContextType {
  historicalData: Record<number | 'all', HistoricalAttack[]>;
  loadingYears: Record<number | 'all', boolean>;
  errors: Record<number | 'all', string | null>;
  fetchDataForYear: (year: number | 'all') => Promise<void>;
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
  
  // Helper function to fetch with timeout
  const fetchWithTimeout = async (url: string, timeout: number = 30000) => {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
      const response = await fetch(url, { signal: controller.signal });
      clearTimeout(timeoutId);
      return response;
    } catch (error) {
      clearTimeout(timeoutId);
      throw error;
    }
  };

  const fetchDataForYear = async (year: number | 'all') => {
    // Special case for 'all' - will be handled by the HistoricalDashboard component
    if (year === 'all') return;
    
    // If we already have the data and it's not loading, don't fetch again
    if (historicalData[year] && !loadingYears[year]) {
      return;
    }

    try {
      // Set loading state for this year
      setLoadingYears(prev => ({ ...prev, [year]: true }));
      
      console.log(`Fetching data for year ${year}...`);
      
      // Set a longer timeout for years that might have more data
      const timeoutMs = year === 2001 ? 60000 : 30000; // 60 seconds for 2001, 30 seconds for others
      
      // Use our fetchWithTimeout helper
      const response = await fetchWithTimeout(
        `http://localhost:8000/historical-data?year=${year}`,
        timeoutMs
      );
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Check if there was an error reported by the API
      if (data.error) {
        console.warn(`API reported an error for year ${year}: ${data.error}`);
        setErrors(prev => ({ 
          ...prev, 
          [year]: `Server error: ${data.error}`
        }));
      } else {
        // Clear any errors
        setErrors(prev => ({ ...prev, [year]: null }));
      }
      
      // Store the fetched data by year (even if there was an error, store any incidents that were returned)
      if (data.incidents && Array.isArray(data.incidents)) {
        setHistoricalData(prev => ({
          ...prev,
          [year]: data.incidents
        }));
        console.log(`Loaded ${data.incidents.length} incidents for year ${year}`);
      }
    } catch (error) {
      console.error(`Error fetching historical data for year ${year}:`, error);
      
      // Handle different error types
      let errorMessage = 'Failed to load historical data';
      
      if (error instanceof DOMException && error.name === 'AbortError') {
        errorMessage = `Request timed out when loading data for year ${year}. Try a different year.`;
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }
      
      setErrors(prev => ({ 
        ...prev, 
        [year]: errorMessage
      }));
      
      // If we had a timeout, provide some empty data so the UI can still show something
      if (error instanceof DOMException && error.name === 'AbortError') {
        // Create a timeout incident that matches HistoricalAttack type
        const timeoutIncident: HistoricalAttack = {
          id: 0, // Use a number for id to match HistoricalAttack type
          year: year as number,
          month: 1,
          day: 1,
          region: "Error",
          country: "Data loading timeout",
          city: "Try a different year",
          latitude: 0,
          longitude: 0,
          attack_type: "Loading Timeout",
          weapon_type: "Unknown",
          target_type: "Unknown",
          num_killed: 0,
          num_wounded: 0,
          group_name: "Loading timed out"
        };
        
        setHistoricalData(prev => ({
          ...prev,
          [year]: [timeoutIncident]
        }));
      }
    } finally {
      // Clear loading state
      setLoadingYears(prev => ({ ...prev, [year]: false }));
    }
  };

  const value = {
    historicalData,
    loadingYears,
    errors,
    fetchDataForYear
  };

  return (
    <HistoricalDataContext.Provider value={value}>
      {children}
    </HistoricalDataContext.Provider>
  );
}; 