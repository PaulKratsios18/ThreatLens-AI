import React, { useEffect } from 'react';
import { MapContainer, TileLayer, Circle, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import '../utils/leaflet-config';
import { PredictionMapProps } from '../types/predictions';

const getRegionCoordinates = (region: string): [number, number] => {
  const coordinates: { [key: string]: [number, number] } = {
    'North America': [40, -100],
    'South America': [-15, -60],
    'Western Europe': [48, 10],
    'Eastern Europe': [50, 30],
    'Middle East': [27, 45],
    'North Africa': [25, 20],
    'Sub-Saharan Africa': [0, 20],
    'Central Asia': [45, 68],
    'South Asia': [20, 77],
    'East Asia': [35, 115],
    'Southeast Asia': [10, 106],
    'Oceania': [-25, 135]
  };
  return coordinates[region] || [0, 0];
};

const PredictionMap: React.FC<PredictionMapProps> = ({ 
  predictions, 
  selectedYear, 
  onRegionClick 
}) => {
  const yearPredictions = predictions.filter(p => p.year === selectedYear);

  return (
    <div className="relative w-full h-[600px] rounded-lg overflow-hidden shadow-lg">
      <MapContainer 
        center={[20, 0]} 
        zoom={2} 
        style={{ height: '100%', width: '100%', background: '#f3f4f6' }}
        scrollWheelZoom={true}
        minZoom={2}
        maxZoom={6}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        {yearPredictions.map((pred, idx) => (
          <Circle
            key={idx}
            center={getRegionCoordinates(pred.region_name)}
            radius={300000}
            pathOptions={{
              color: pred.risk_level === 'High' ? '#ef4444' : 
                     pred.risk_level === 'Medium' ? '#f59e0b' : '#22c55e',
              fillOpacity: 0.6,
              weight: 2
            }}
            eventHandlers={{
              click: () => onRegionClick(pred.region_name)
            }}
          >
            <Popup>
              <div className="p-2">
                <h3 className="font-bold">{pred.region_name}</h3>
                <p>Expected Attacks: {pred.expected_attacks}</p>
                <p>Risk Level: {pred.risk_level}</p>
                <p>Confidence: {(pred.confidence_score * 100).toFixed(1)}%</p>
              </div>
            </Popup>
          </Circle>
        ))}
      </MapContainer>
    </div>
  );
};

export default PredictionMap;