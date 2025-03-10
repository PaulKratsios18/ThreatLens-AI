import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import '../utils/leaflet-config';
import { HistoricalAttack, CountryStats, calculateCountryStatistics } from '../utils/countryUtils';

interface HistoricalMapProps {
  incidents: HistoricalAttack[];
  selectedCountry?: string | null;
}

const HistoricalMap: React.FC<HistoricalMapProps> = ({ incidents, selectedCountry }) => {
  const [countryData, setCountryData] = useState<any>(null);
  const [countriesStats, setCountriesStats] = useState<Record<string, CountryStats>>({});
  const [maxAttacks, setMaxAttacks] = useState(0);

  // Fetch GeoJSON country boundaries
  useEffect(() => {
    fetch('https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson')
      .then(response => response.json())
      .then(data => {
        setCountryData(data);
      });
  }, []);

  // Calculate statistics for each country using the utility function
  useEffect(() => {
    if (!incidents || incidents.length === 0) return;

    const stats = calculateCountryStatistics(incidents);
    
    // Find the country with the most attacks
    let highestAttackCount = 0;
    Object.values(stats).forEach(stat => {
      if (stat.attackCount > highestAttackCount) {
        highestAttackCount = stat.attackCount;
      }
    });

    setCountriesStats(stats);
    setMaxAttacks(highestAttackCount);
  }, [incidents]);

  // Style function for GeoJSON
  const getCountryStyle = (feature: any) => {
    const countryName = feature.properties.name;
    const stats = Object.values(countriesStats).find(
      stat => stat.standardizedCountry === countryName
    );
    
    // Style for selected country
    if (selectedCountry && countryName === selectedCountry) {
      return {
        fillColor: '#3b82f6', // blue
        fillOpacity: 0.7,
        weight: 2,
        color: '#1d4ed8',
        opacity: 1
      };
    }
    
    // Default style (no attacks)
    if (!stats) {
      return {
        fillColor: '#22c55e', // green
        fillOpacity: 0.4,
        weight: 1,
        color: '#666',
        opacity: 0.5
      };
    }

    // Calculate color based on attack count relative to max
    const ratio = stats.attackCount / maxAttacks;
    
    // Color scale from green (0 attacks) to red (max attacks)
    const getColor = (ratio: number) => {
      if (ratio === 0) return '#22c55e'; // green
      if (ratio < 0.1) return '#84cc16'; // lime green
      if (ratio < 0.25) return '#eab308'; // yellow
      if (ratio < 0.5) return '#f97316'; // orange
      if (ratio < 0.75) return '#ef4444'; // red
      return '#b91c1c'; // dark red
    };

    return {
      fillColor: getColor(ratio),
      fillOpacity: 0.6,
      weight: 1,
      color: '#666',
      opacity: 0.7
    };
  };

  // Tooltip content for each country
  const onEachFeature = (feature: any, layer: any) => {
    const countryName = feature.properties.name;
    const stats = Object.values(countriesStats).find(
      stat => stat.standardizedCountry === countryName
    );
    
    if (stats) {
      layer.bindTooltip(`
        <div class="p-2">
          <strong>${countryName}</strong><br/>
          Attacks: ${stats.attackCount}<br/>
          Deaths: ${stats.deaths}<br/>
          Wounded: ${stats.wounded}<br/>
        </div>
      `);
    } else {
      layer.bindTooltip(`
        <div class="p-2">
          <strong>${countryName}</strong><br/>
          No recorded attacks
        </div>
      `);
    }
  };

  return (
    <div className="relative w-full h-[500px] rounded-lg overflow-hidden">
      <MapContainer 
        center={[20, 0]} 
        zoom={2} 
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={true}
        minZoom={1.5}
        maxZoom={6}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        {countryData && (
          <GeoJSON 
            data={countryData} 
            style={getCountryStyle}
            onEachFeature={onEachFeature}
          />
        )}
      </MapContainer>

      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-white p-2 rounded shadow-md text-sm">
        <div className="font-bold mb-1">Attack Intensity</div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#b91c1c'}}></span>
          <span>Very High</span>
        </div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#ef4444'}}></span>
          <span>High</span>
        </div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#f97316'}}></span>
          <span>Medium</span>
        </div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#eab308'}}></span>
          <span>Low</span>
        </div>
        <div className="flex items-center">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#22c55e'}}></span>
          <span>None</span>
        </div>
      </div>
    </div>
  );
};

export default HistoricalMap; 