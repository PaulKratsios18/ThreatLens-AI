import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, Tooltip, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import '../utils/leaflet-config';
import { countryToRegionMapping, standardizeCountryName, getGeoJSONCountryName } from '../utils/countryUtils';
import { Prediction } from '../types/predictions';

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

// Country coordinates for better placement on the map
const getCountryCoordinates = (country: string): [number, number] => {
  const coordinates: { [key: string]: [number, number] } = {
    // Middle East & North Africa
    'Iraq': [33.2232, 43.6793],
    'Syria': [34.8021, 38.9968],
    'Iran': [32.4279, 53.6880],
    'Egypt': [26.8206, 30.8025],
    'Libya': [26.3351, 17.2283],
    'Yemen': [15.5527, 48.5164],
    'Israel': [31.0461, 34.8516],
    
    // Sub-Saharan Africa
    'Burkina Faso': [12.2383, -1.5616],
    'Mali': [17.5707, -3.9962],
    'Niger': [17.6078, 8.0817],
    'Nigeria': [9.0820, 8.6753],
    'Somalia': [5.1521, 46.1996],
    'Democratic Republic of the Congo': [-4.0383, 21.7587],
    
    // South Asia
    'Afghanistan': [33.9391, 67.7100],
    'Pakistan': [30.3753, 69.3451],
    
    // Western Europe
    'France': [46.2276, 2.2137],
    'United Kingdom': [55.3781, -3.4360],
    'Iceland': [64.9631, -19.0208],
    
    // North America
    'United States': [37.0902, -95.7129],
    'USA': [37.0902, -95.7129],
    'Canada': [56.1304, -106.3468],
    
    // East Asia
    'Japan': [36.2048, 138.2529],
  };
  
  return coordinates[country] || [0, 0];
};

// Fallback to region coordinates
const getRegionCoordinates = (region: string): [number, number] => {
  const coordinates: { [key: string]: [number, number] } = {
    'North America': [40, -100],
    'South America': [-20, -60],
    'Central America & Caribbean': [15, -85],
    'Western Europe': [48, 10],
    'Eastern Europe': [50, 30],
    'Middle East & North Africa': [30, 35],
    'Sub-Saharan Africa': [0, 20],
    'South Asia': [25, 75],
    'Central Asia': [40, 70],
    'East Asia': [35, 115],
    'Southeast Asia': [10, 110],
    'Australasia & Oceania': [-25, 135]
  };
  return coordinates[region] || [0, 0];
};

interface PredictionMapProps {
  predictions: Prediction[];
  selectedYear: number;
  onCountrySelect: (country: string) => void;
  onRegionSelect?: (region: string) => void;
}

// Add utility functions
const getRegionForCountry = (country: string): string => {
  const standardizedName = standardizeCountryName(country);
  return countryToRegionMapping[standardizedName] || 'Unknown';
};

const getRiskLevel = (attackCount: number): string => {
  if (attackCount > 100) return 'Very High';
  if (attackCount > 50) return 'High';
  if (attackCount > 25) return 'Significant';
  if (attackCount > 10) return 'Moderate';
  if (attackCount > 3) return 'Low';
  if (attackCount > 1) return 'Few';
  if (attackCount === 1) return 'Single';
  return 'None';
};

const PredictionMap: React.FC<PredictionMapProps> = ({ 
  predictions, 
  selectedYear, 
  onCountrySelect,
  onRegionSelect 
}) => {
  const [countryData, setCountryData] = useState<GeoJSON.FeatureCollection | null>(null);
  const [selectedCountry, setSelectedCountry] = useState<string | null>(null);

  // Fetch GeoJSON country boundaries
  useEffect(() => {
    fetch('https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson')
      .then(response => response.json())
      .then(data => {
        setCountryData(data);
      });
  }, []);

  // Style function for GeoJSON
  const getCountryStyle = (feature: any): L.PathOptions => {
    const countryName = feature.properties?.name;
    // Standardize the GeoJSON country name
    const standardizedName = standardizeCountryName(countryName);
    
    // Find prediction using the standardized name
    const prediction = predictions.find(p => 
      standardizeCountryName(p.country) === standardizedName
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
    
    // Default style (no risk data) - light gray
    if (!prediction) {
      return {
        fillColor: '#e5e7eb', // light gray for countries with no data
        fillOpacity: 0.4,
        weight: 1,
        color: '#666',
        opacity: 0.5
      };
    }

    // Get color based on risk level and expected attacks
    const getColor = (riskLevel: string, attackCount: number) => {
      // First check risk level
      switch (riskLevel.toLowerCase()) {
        case 'high':
          if (attackCount > 100) return '#b91c1c'; // Very high - dark red
          return '#ef4444'; // High - red
        case 'medium':
          if (attackCount > 50) return '#f97316'; // Significant - orange
          if (attackCount > 25) return '#fdba74'; // Moderate - light orange
          return '#fef08a'; // Low - yellow
        case 'low':
          if (attackCount <= 1) return '#dcfce7'; // Very low - very light green
          if (attackCount <= 3) return '#86efac'; // Few - light green
          return '#22c55e'; // Low - green
        default:
          return '#e5e7eb'; // Unknown - light gray
      }
    };

    return {
      fillColor: getColor(prediction.risk_level, prediction.expected_attacks),
      fillOpacity: 0.7,
      weight: 1,
      color: '#666',
      opacity: 0.7
    };
  };

  // Tooltip content for each country
  const onEachFeature = (feature: any, layer: L.Layer): void => {
    const countryName = feature.properties?.name;
    // Standardize the GeoJSON country name
    const standardizedName = standardizeCountryName(countryName);
    
    // Find prediction using the standardized name
    const prediction = predictions.find(p => 
      standardizeCountryName(p.country) === standardizedName
    );
    
    if (prediction) {
      layer.bindTooltip(`
        <div class="p-2">
          <strong>${countryName}</strong> (${prediction.region_name})<br/>
          Expected Attacks: ${prediction.expected_attacks}<br/>
          GTI Score: ${typeof prediction.gti_score === 'number' ? prediction.gti_score.toFixed(1) : 'N/A'}<br/>
          Risk Level: ${prediction.risk_level}
        </div>
      `);

      // Add click handler
      layer.on({
        click: () => {
          setSelectedCountry(countryName);
          onCountrySelect(prediction.country);
        }
      });
    } else {
      layer.bindTooltip(`
        <div class="p-2">
          <strong>${countryName}</strong><br/>
          No prediction data available
        </div>
      `);
    }
  };

  return (
    <div className="w-full h-[500px] rounded-lg overflow-hidden bg-gray-100 relative">
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
            style={getCountryStyle}
            onEachFeature={onEachFeature}
          />
        )}
      </MapContainer>
      
      {/* Updated Legend to match HistoricalMap */}
      <div className="absolute bottom-4 right-4 bg-white p-2 rounded shadow-md text-sm">
        <div className="font-bold mb-1">Risk Level & Expected Attacks</div>
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

export default PredictionMap;