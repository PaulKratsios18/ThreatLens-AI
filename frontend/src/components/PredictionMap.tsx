import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Circle, Tooltip, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import '../utils/leaflet-config';
import { PredictionMapProps } from '../types/predictions';

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

const PredictionMap: React.FC<PredictionMapProps> = ({ 
  predictions, 
  selectedYear, 
  onRegionClick,
  onCountryClick 
}) => {
  // Get color based on risk level
  const getColorForRisk = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'high':
        return '#ef4444'; // red-500
      case 'medium':
        return '#f59e0b'; // amber-500
      case 'low':
        return '#22c55e'; // green-500
      default:
        return '#94a3b8'; // slate-400
    }
  };

  // Get radius based on GTI score and expected attacks
  const getCircleRadius = (gtiScore: number | undefined, expectedAttacks: number) => {
    if (gtiScore !== undefined) {
      return Math.max(gtiScore * 30000, 30000); // Base size on GTI score
    }
    // Fallback calculation based on expected attacks
    return Math.max(Math.log(expectedAttacks + 1) * 35000, 30000);
  };

  return (
    <div className="w-full h-[500px] rounded-lg overflow-hidden bg-gray-100 relative">
      <MapContainer 
        center={[20, 0]} 
        zoom={2} 
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={true}
        minZoom={2}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        {predictions.map((pred) => {
          const coordinates = getCountryCoordinates(pred.country) || getRegionCoordinates(pred.region_name);
          return (
            <Circle
              key={pred.country}
              center={coordinates}
              radius={getCircleRadius(pred.gti_score, pred.expected_attacks)}
              pathOptions={{
                color: getColorForRisk(pred.risk_level),
                fillColor: getColorForRisk(pred.risk_level),
                fillOpacity: 0.6
              }}
              eventHandlers={{
                click: () => {
                  if (onCountryClick) {
                    onCountryClick(pred.country);
                  } else if (onRegionClick) {
                    onRegionClick(pred.region_name);
                  }
                }
              }}
            >
              <Tooltip>
                <div className="p-1">
                  <strong>{pred.country}</strong> ({pred.region_name})<br/>
                  GTI Score: {pred.gti_score?.toFixed(1) || 'N/A'}<br/>
                  Expected Attacks: {pred.expected_attacks}<br/>
                  Risk Level: {pred.risk_level}
                </div>
              </Tooltip>
            </Circle>
          );
        })}
      </MapContainer>
      
      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-white p-2 rounded shadow-md text-sm">
        <div className="font-bold mb-1">Risk Level</div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 rounded-full mr-1" style={{backgroundColor: '#ef4444'}}></span>
          <span>High</span>
        </div>
        <div className="flex items-center mb-1">
          <span className="inline-block w-3 h-3 rounded-full mr-1" style={{backgroundColor: '#f59e0b'}}></span>
          <span>Medium</span>
        </div>
        <div className="flex items-center">
          <span className="inline-block w-3 h-3 rounded-full mr-1" style={{backgroundColor: '#22c55e'}}></span>
          <span>Low</span>
        </div>
      </div>
    </div>
  );
};

export default PredictionMap;