import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, GeoJSON, Tooltip, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import '../utils/leaflet-config';
import { HistoricalAttack, CountryStats, calculateCountryStatistics, standardizeCountryName } from '../utils/countryUtils';

interface HistoricalMapProps {
  incidents: HistoricalAttack[];
  selectedCountry?: string | null;
}

// Map boundaries restrictor component
const MapBoundariesRestrictor: React.FC = () => {
  const map = useMap();
  
  useEffect(() => {
    if (!map) return;

    // Set world boundaries to restrict panning
    const southWest = L.latLng(-60, -180);
    const northEast = L.latLng(85, 180);
    const bounds = L.latLngBounds(southWest, northEast);
    
    map.setMaxBounds(bounds);
    map.on('drag', () => {
      map.panInsideBounds(bounds, { animate: false });
    });
    
    return () => {
      map.off('drag');
    };
  }, [map]);
  
  return null;
};

const HistoricalMap: React.FC<HistoricalMapProps> = ({ incidents, selectedCountry }) => {
  const [countryData, setCountryData] = useState<GeoJSON.FeatureCollection | null>(null);
  const [countriesStats, setCountriesStats] = useState<Record<string, CountryStats>>({});
  const [maxAttacks, setMaxAttacks] = useState(0);
  const [medianAttacks, setMedianAttacks] = useState(0);

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
    
    // Find the country with the most attacks and calculate median for better scaling
    let highestAttackCount = 0;
    const attackCounts: number[] = [];
    
    Object.values(stats).forEach(stat => {
      attackCounts.push(stat.attackCount);
      if (stat.attackCount > highestAttackCount) {
        highestAttackCount = stat.attackCount;
      }
    });
    
    // Calculate median if we have data
    if (attackCounts.length > 0) {
      attackCounts.sort((a, b) => a - b);
      const mid = Math.floor(attackCounts.length / 2);
      const medianValue = attackCounts.length % 2 === 0 
        ? (attackCounts[mid - 1] + attackCounts[mid]) / 2 
        : attackCounts[mid];
      setMedianAttacks(medianValue);
    }

    setCountriesStats(stats);
    setMaxAttacks(highestAttackCount);
  }, [incidents]);

  // Style function for GeoJSON
  const getCountryStyle = (feature: any): L.PathOptions => {
    const countryName = feature.properties?.name;
    const standardizedCountryName = standardizeCountryName(countryName);
    const stats = Object.values(countriesStats).find(
      stat => stat.standardizedCountry === standardizedCountryName
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
    
    // Default style (no attacks) - light gray instead of green
    if (!stats) {
      return {
        fillColor: '#e5e7eb', // light gray for countries with no data
        fillOpacity: 0.4,
        weight: 1,
        color: '#666',
        opacity: 0.5
      };
    }

    // Improved color scaling for better differentiation
    const getColor = (attackCount: number) => {
      if (attackCount === 0) return '#e5e7eb'; // light gray for zero attacks
      if (attackCount === 1) return '#dcfce7'; // very light green - exactly 1 attack
      if (attackCount <= 3) return '#86efac'; // light green - few attacks (2-3)
      if (attackCount <= 10) return '#fef08a'; // light yellow - low attacks (4-10)
      if (attackCount <= 25) return '#fdba74'; // light orange - moderate (11-25)
      if (attackCount <= 50) return '#f97316'; // orange - significant (26-50)
      if (attackCount <= 100) return '#ef4444'; // red - high (51-100)
      return '#b91c1c'; // dark red - very high (100+)
    };

    return {
      fillColor: getColor(stats.attackCount),
      fillOpacity: 0.7,
      weight: 1,
      color: '#666',
      opacity: 0.7
    };
  };

  // Tooltip content for each country
  const onEachFeature = (feature: any, layer: L.Layer): void => {
    const countryName = feature.properties?.name;
    const standardizedCountryName = standardizeCountryName(countryName);
    
    const stats = Object.values(countriesStats).find(
      stat => stat.standardizedCountry === standardizedCountryName
    );
    
    if (stats) {
      layer.bindTooltip(
        `<div class="p-2">
  <strong>${countryName}</strong><br/>
  Attacks: ${stats.attackCount}<br/>
  Deaths: ${stats.deaths}<br/>
  Wounded: ${stats.wounded}<br/>
</div>`
      );
    } else {
      layer.bindTooltip(
        `<div class="p-2">
  <strong>${countryName}</strong><br/>
  No recorded attacks
</div>`
      );
    }
  };

  return (
    <MapContainer
      center={[0, 0]}
      zoom={2}
      style={{ width: '100%', height: '300px' }}
      maxBounds={[[90, -180], [-90, 180]]}
    >
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
      />
      {countryData && (
        <GeoJSON
          data={countryData}
          style={getCountryStyle}
          onEachFeature={onEachFeature}
        />
      )}
      <MapBoundariesRestrictor />
    </MapContainer>
  );
};

export default HistoricalMap;