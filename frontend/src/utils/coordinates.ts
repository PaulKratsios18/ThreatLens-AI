interface Coordinates {
  lat: number;
  lng: number;
}

const regionCoordinates: Record<string, Coordinates> = {
  "North America": { lat: 40, lng: -100 },
  "South America": { lat: -15, lng: -60 },
  "Western Europe": { lat: 48, lng: 10 },
  "Eastern Europe": { lat: 50, lng: 30 },
  "Middle East": { lat: 27, lng: 45 },
  "North Africa": { lat: 25, lng: 20 },
  "Sub-Saharan Africa": { lat: 0, lng: 20 },
  "Central Asia": { lat: 45, lng: 68 },
  "South Asia": { lat: 20, lng: 77 },
  "East Asia": { lat: 35, lng: 115 },
  "Southeast Asia": { lat: 10, lng: 106 },
  "Oceania": { lat: -25, lng: 135 }
};

export const getRegionCoordinates = (region: string): [number, number] => {
  const coords = regionCoordinates[region];
  if (!coords) {
    console.warn(`No coordinates found for region: ${region}`);
    return [0, 0];
  }
  return [coords.lat, coords.lng];
}; 