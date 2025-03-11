import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { HistoricalAttack } from '../utils/countryUtils';

interface HistoricalDataContextType {
  historicalData: Record<number, HistoricalAttack[]>;
  loadingYears: Record<number, boolean>;
  errors: Record<number, string | null>;
  fetchDataForYear: (year: number) => Promise<void>;
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
  const [historicalData, setHistoricalData] = useState<Record<number, HistoricalAttack[]>>({});
  const [loadingYears, setLoadingYears] = useState<Record<number, boolean>>({});
  const [errors, setErrors] = useState<Record<number, string | null>>({});

  const fetchDataForYear = async (year: number) => {
    // If we already have the data and it's not loading, don't fetch again
    if (historicalData[year] && !loadingYears[year]) {
      return;
    }

    try {
      // Set loading state for this year
      setLoadingYears(prev => ({ ...prev, [year]: true }));
      
      const response = await fetch(`http://localhost:8000/historical-data?year=${year}`);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Store the fetched data by year
      setHistoricalData(prev => ({
        ...prev,
        [year]: data.incidents
      }));
      
      // Clear any errors
      setErrors(prev => ({ ...prev, [year]: null }));
    } catch (error) {
      console.error(`Error fetching historical data for year ${year}:`, error);
      setErrors(prev => ({ 
        ...prev, 
        [year]: error instanceof Error ? error.message : 'Failed to load historical data' 
      }));
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