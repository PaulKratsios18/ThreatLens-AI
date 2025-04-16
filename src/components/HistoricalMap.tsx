import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import '../utils/leaflet-config';

// Define interfaces and types
interface HistoricalAttack {
  id: number;
  year: number;
  month: number;
  day: number;
  region: string;
  country: string;
  city: string;
  latitude: number;
  longitude: number;
  attack_type: string;
  weapon_type: string;
  target_type: string;
  num_killed: number;
  num_wounded: number;
  group_name?: string;
}

interface CountryStats {
  country: string;
  standardizedCountry: string;
  attackCount: number;
  deaths: number;
  wounded: number;
  attackTypes: Record<string, number>;
  groups?: Record<string, number>;
}

interface HistoricalMapProps {
  incidents: HistoricalAttack[];
  selectedCountry?: string | null;
}

// Calculate statistics for each country
const calculateCountryStatistics = (incidents: HistoricalAttack[]): Record<string, CountryStats> => {
  const stats: Record<string, CountryStats> = {};
  
  incidents.forEach(incident => {
    // Use the country name as is for the standardized name since we're not doing any mapping here
    const standardizedCountry = incident.country;
    
    if (!stats[standardizedCountry]) {
      stats[standardizedCountry] = {
        country: incident.country,
        standardizedCountry,
        attackCount: 0,
        deaths: 0,
        wounded: 0,
        attackTypes: {},
        groups: {}
      };
    }
    
    stats[standardizedCountry].attackCount += 1;
    stats[standardizedCountry].deaths += incident.num_killed;
    stats[standardizedCountry].wounded += incident.num_wounded;
    
    if (!stats[standardizedCountry].attackTypes[incident.attack_type]) {
      stats[standardizedCountry].attackTypes[incident.attack_type] = 0;
    }
    stats[standardizedCountry].attackTypes[incident.attack_type] += 1;
    
    if (incident.group_name && stats[standardizedCountry].groups) {
      const groupName = incident.group_name === 'Unknown' ? 'Unknown' : incident.group_name;
      stats[standardizedCountry].groups[groupName] = (stats[standardizedCountry].groups[groupName] || 0) + 1;
    }
  });
  
  return stats;
};

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
  
  // Fetch GeoJSON country boundaries
  useEffect(() => {
    fetch('https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson')
      .then(response => response.json())
      .then(data => {
        setCountryData(data);
      });
  }, []);

  // Calculate statistics for each country
  useEffect(() => {
    if (!incidents || incidents.length === 0) return;
    setCountriesStats(calculateCountryStatistics(incidents));
  }, [incidents]);

  // Style function for GeoJSON
  const getCountryStyle = (feature: GeoJSON.Feature<GeoJSON.Geometry>): L.PathOptions => {
    const countryName = feature.properties?.name;
    const stats = Object.values(countriesStats).find(
      (stat: CountryStats) => stat.standardizedCountry === countryName
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
  const onEachFeature = (feature: GeoJSON.Feature, layer: L.Layer): void => {
    const countryName = feature.properties?.name;
    const stats = Object.values(countriesStats).find(
      (stat: CountryStats) => stat.standardizedCountry === countryName
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
    <div className="relative w-full h-[500px] rounded-lg overflow-hidden">
      <MapContainer 
        center={[20, 0]} 
        zoom={2} 
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={true}
        minZoom={1.5}
        maxZoom={6}
        maxBoundsViscosity={1.0}
      >
        <MapBoundariesRestrictor />
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        {countryData && (
          <GeoJSON 
            data={countryData} 
            style={getCountryStyle as any}
            onEachFeature={onEachFeature}
          />
        )}
      </MapContainer>

      {/* Updated Legend */}
      <div className="absolute bottom-4 right-4 bg-white p-2 rounded shadow-md text-sm">
        <div className="font-bold mb-1">Attack Intensity</div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#b91c1c'}}></span>
          <span>Very High (100+)</span>
        </div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#ef4444'}}></span>
          <span>High (51-100)</span>
        </div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#f97316'}}></span>
          <span>Significant (26-50)</span>
        </div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#fdba74'}}></span>
          <span>Moderate (11-25)</span>
        </div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#fef08a'}}></span>
          <span>Low (4-10)</span>
        </div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#86efac'}}></span>
          <span>Few (2-3)</span>
        </div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#dcfce7'}}></span>
          <span>Single (1)</span>
        </div>
        <div className="flex items-center">
          <span className="inline-block w-3 h-3 mr-1" style={{backgroundColor: '#e5e7eb'}}></span>
          <span>None (0)</span>
        </div>
      </div>
    </div>
  );
};

export default HistoricalMap; 