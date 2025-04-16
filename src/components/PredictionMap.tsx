import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import '../utils/leaflet-config';

// Define types for predictions
interface Prediction {
  country: string;
  region_name: string;
  year: number;
  expected_attacks: number;
  gti_score?: number;
  risk_level: string;
  attack_types?: Record<string, number>;
  confidence_score: number;
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

interface PredictionMapProps {
  predictions: Prediction[];
  onCountrySelect: (country: string) => void;
}

const PredictionMap: React.FC<PredictionMapProps> = ({ 
  predictions, 
  onCountrySelect
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
  const getCountryStyle = (feature: GeoJSON.Feature<GeoJSON.Geometry>): L.PathOptions => {
    const countryName = feature.properties?.name;
    const prediction = predictions.find(p => p.country === countryName);
    
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
  const onEachFeature = (feature: GeoJSON.Feature, layer: L.Layer): void => {
    const countryName = feature.properties?.name;
    const prediction = predictions.find(p => p.country === countryName);
    
    if (prediction) {
      layer.bindTooltip(
        `<div class="p-2">
          <strong>${countryName}</strong> (${prediction.region_name})<br/>
          Expected Attacks: ${prediction.expected_attacks}<br/>
          GTI Score: ${typeof prediction.gti_score === 'number' ? prediction.gti_score.toFixed(1) : 'N/A'}<br/>
          Risk Level: ${prediction.risk_level}
        </div>`
      );

      // Add click handler
      layer.on({
        click: () => {
          setSelectedCountry(countryName);
          onCountrySelect(countryName);
        }
      });
    } else {
      layer.bindTooltip(
        `<div class="p-2">
          <strong>${countryName}</strong><br/>
          No prediction data available
        </div>`
      );
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
            style={getCountryStyle as any}
            onEachFeature={onEachFeature}
          />
        )}
      </MapContainer>
      
      {/* Risk Level Legend */}
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