import React from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import '../utils/leaflet-config';

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
}

interface HistoricalMapProps {
  incidents: HistoricalAttack[];
}

const HistoricalMap: React.FC<HistoricalMapProps> = ({ incidents }) => {
  return (
    <div className="relative w-full h-[500px] rounded-lg overflow-hidden">
      <MapContainer 
        center={[20, 0]} 
        zoom={2} 
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={true}
        minZoom={2}
        maxZoom={6}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
        />
        {incidents.map((incident) => (
          <CircleMarker
            key={incident.id}
            center={[incident.latitude, incident.longitude]}
            radius={Math.log(incident.num_killed + incident.num_wounded + 1) * 3}
            pathOptions={{
              color: incident.num_killed > 10 ? '#ef4444' : 
                    incident.num_killed > 1 ? '#f59e0b' : '#22c55e',
              fillOpacity: 0.8
            }}
          >
            <Popup>
              <div className="p-2">
                <h3 className="font-bold">{incident.city}, {incident.country}</h3>
                <p>Date: {incident.month}/{incident.day}/{incident.year}</p>
                <p>Attack Type: {incident.attack_type}</p>
                <p>Weapon: {incident.weapon_type}</p>
                <p>Target: {incident.target_type}</p>
                <p>Casualties: {incident.num_killed} killed, {incident.num_wounded} wounded</p>
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  );
};

export default HistoricalMap; 