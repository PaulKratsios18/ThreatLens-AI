import React from 'react';
import DashboardTabs from './components/DashboardTabs';
import { HistoricalDataProvider } from './contexts/HistoricalDataContext';

function App() {
  return (
    <div className="App">
      <HistoricalDataProvider>
        <DashboardTabs />
      </HistoricalDataProvider>
    </div>
  );
}

export default App; 