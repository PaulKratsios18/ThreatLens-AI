// Comprehensive mapping of all country name variations to standardized names
export const countryNameMapping: Record<string, string> = {
  // North America
  "United States": "USA",
  "United States of America": "USA",
  "America": "USA",
  "USA": "USA",
  "US": "USA",
  "Canada": "Canada",
  "Mexico": "Mexico",
  
  // Europe
  "United Kingdom": "United Kingdom",
  "UK": "United Kingdom",
  "Great Britain": "United Kingdom",
  "England": "United Kingdom",
  "France": "France",
  "Germany": "Germany",
  "Italy": "Italy",
  "Spain": "Spain",
  
  // Asia
  "China": "China",
  "People's Republic of China": "China",
  "Japan": "Japan",
  "India": "India",
  "Republic of India": "India",
  "Russia": "Russia",
  "Russian Federation": "Russia",
  
  // Middle East
  "Iran": "Iran",
  "Islamic Republic of Iran": "Iran",
  "Iraq": "Iraq",
  "Saudi Arabia": "Saudi Arabia",
  "KSA": "Saudi Arabia",
  "Israel": "Israel",
  "Syria": "Syria",
  "Syrian Arab Republic": "Syria",
  
  // Africa
  "South Africa": "South Africa",
  "Nigeria": "Nigeria",
  "Egypt": "Egypt",
  "Kenya": "Kenya",
  
  // South America
  "Brazil": "Brazil",
  "Argentina": "Argentina",
  "Colombia": "Colombia",
  "Venezuela": "Venezuela",
  
  // Oceania
  "Australia": "Australia",
  "New Zealand": "New Zealand",
  
  // Others that might have variations
  "North Korea": "North Korea",
  "Democratic People's Republic of Korea": "North Korea",
  "DPRK": "North Korea",
  "South Korea": "South Korea",
  "Republic of Korea": "South Korea",
  "Vietnam": "Vietnam",
  "Viet Nam": "Vietnam",
  "Burma": "Myanmar",
  "Myanmar": "Myanmar",
  "Democratic Republic of the Congo": "DR Congo",
  "DR Congo": "DR Congo",
  "DRC": "DR Congo",
  "Congo-Kinshasa": "DR Congo",
  "Republic of the Congo": "Congo",
  "Congo-Brazzaville": "Congo",
  "Congo": "Congo",
  "UAE": "United Arab Emirates",
  "United Arab Emirates": "United Arab Emirates",
};

// Map from GeoJSON country names to standardized names
export const geoJSONtoStandardMapping: Record<string, string> = {
  "United States": "USA",
  "United States of America": "USA",
  "United Kingdom": "United Kingdom",
  "Dominican Rep.": "Dominican Republic",
  "Dem. Rep. Congo": "DR Congo",
  "Central African Rep.": "Central African Republic",
  "S. Sudan": "South Sudan",
  "Côte d'Ivoire": "Ivory Coast",
  "Bosnia and Herz.": "Bosnia and Herzegovina"
};

// Map from standardized names to GeoJSON country names
export const standardToGeoJSONMapping: Record<string, string> = {
  "USA": "United States",
  "Dominican Republic": "Dominican Rep.",
  "DR Congo": "Dem. Rep. Congo",
  "Central African Republic": "Central African Rep.",
  "South Sudan": "S. Sudan",
  "Ivory Coast": "Côte d'Ivoire",
  "Bosnia and Herzegovina": "Bosnia and Herz."
};

// Map from standardized names to socioeconomic data names
export const standardToSocioeconomicMapping: Record<string, string> = {
  "USA": "United States",
  "United Kingdom": "United Kingdom",
  "DR Congo": "Democratic Republic of the Congo"
};

// Map countries to their regions
export const countryToRegionMapping: Record<string, string> = {
  // North America
  'United States': 'North America',
  'USA': 'North America',
  'Canada': 'North America',
  'Mexico': 'North America',
  
  // Central America & Caribbean
  'Guatemala': 'Central America & Caribbean',
  'Honduras': 'Central America & Caribbean',
  'El Salvador': 'Central America & Caribbean',
  'Nicaragua': 'Central America & Caribbean',
  'Costa Rica': 'Central America & Caribbean',
  'Panama': 'Central America & Caribbean',
  'Cuba': 'Central America & Caribbean',
  'Jamaica': 'Central America & Caribbean',
  'Haiti': 'Central America & Caribbean',
  'Dominican Republic': 'Central America & Caribbean',
  'Puerto Rico': 'Central America & Caribbean',
  
  // South America
  'Colombia': 'South America',
  'Venezuela': 'South America',
  'Ecuador': 'South America',
  'Peru': 'South America',
  'Brazil': 'South America',
  'Bolivia': 'South America',
  'Paraguay': 'South America',
  'Chile': 'South America',
  'Argentina': 'South America',
  'Uruguay': 'South America',
  
  // Western Europe
  'United Kingdom': 'Western Europe',
  'Ireland': 'Western Europe',
  'France': 'Western Europe',
  'Germany': 'Western Europe',
  'Italy': 'Western Europe',
  'Spain': 'Western Europe',
  'Portugal': 'Western Europe',
  'Belgium': 'Western Europe',
  'Netherlands': 'Western Europe',
  'Switzerland': 'Western Europe',
  'Austria': 'Western Europe',
  'Sweden': 'Western Europe',
  'Norway': 'Western Europe',
  'Denmark': 'Western Europe',
  'Finland': 'Western Europe',
  'Iceland': 'Western Europe',
  'Greece': 'Western Europe',
  
  // Eastern Europe
  'Russia': 'Eastern Europe',
  'Estonia': 'Eastern Europe',
  'Latvia': 'Eastern Europe',
  'Lithuania': 'Eastern Europe',
  'Belarus': 'Eastern Europe',
  'Ukraine': 'Eastern Europe',
  'Moldova': 'Eastern Europe',
  'Poland': 'Eastern Europe',
  'Czech Republic': 'Eastern Europe',
  'Slovakia': 'Eastern Europe',
  'Hungary': 'Eastern Europe',
  'Romania': 'Eastern Europe',
  'Bulgaria': 'Eastern Europe',
  'Serbia': 'Eastern Europe',
  'Croatia': 'Eastern Europe',
  'Bosnia and Herzegovina': 'Eastern Europe',
  'North Macedonia': 'Eastern Europe',
  'Albania': 'Eastern Europe',
  
  // Middle East & North Africa
  'Turkey': 'Middle East & North Africa',
  'Syria': 'Middle East & North Africa',
  'Iraq': 'Middle East & North Africa',
  'Iran': 'Middle East & North Africa',
  'Israel': 'Middle East & North Africa',
  'Palestine': 'Middle East & North Africa',
  'Jordan': 'Middle East & North Africa',
  'Saudi Arabia': 'Middle East & North Africa',
  'Yemen': 'Middle East & North Africa',
  'Oman': 'Middle East & North Africa',
  'United Arab Emirates': 'Middle East & North Africa',
  'Qatar': 'Middle East & North Africa',
  'Kuwait': 'Middle East & North Africa',
  'Bahrain': 'Middle East & North Africa',
  'Egypt': 'Middle East & North Africa',
  'Libya': 'Middle East & North Africa',
  'Tunisia': 'Middle East & North Africa',
  'Algeria': 'Middle East & North Africa',
  'Morocco': 'Middle East & North Africa',
  
  // Sub-Saharan Africa
  'Sudan': 'Sub-Saharan Africa',
  'South Sudan': 'Sub-Saharan Africa',
  'Ethiopia': 'Sub-Saharan Africa',
  'Eritrea': 'Sub-Saharan Africa',
  'Djibouti': 'Sub-Saharan Africa',
  'Somalia': 'Sub-Saharan Africa',
  'Kenya': 'Sub-Saharan Africa',
  'Uganda': 'Sub-Saharan Africa',
  'Rwanda': 'Sub-Saharan Africa',
  'Burundi': 'Sub-Saharan Africa',
  'Tanzania': 'Sub-Saharan Africa',
  'Democratic Republic of the Congo': 'Sub-Saharan Africa',
  'Republic of the Congo': 'Sub-Saharan Africa',
  'Gabon': 'Sub-Saharan Africa',
  'Equatorial Guinea': 'Sub-Saharan Africa',
  'Cameroon': 'Sub-Saharan Africa',
  'Nigeria': 'Sub-Saharan Africa',
  'Niger': 'Sub-Saharan Africa',
  'Chad': 'Sub-Saharan Africa',
  'Mali': 'Sub-Saharan Africa',
  'Burkina Faso': 'Sub-Saharan Africa',
  'Senegal': 'Sub-Saharan Africa',
  'Guinea': 'Sub-Saharan Africa',
  'Sierra Leone': 'Sub-Saharan Africa',
  'Liberia': 'Sub-Saharan Africa',
  'Ivory Coast': 'Sub-Saharan Africa',
  'Ghana': 'Sub-Saharan Africa',
  'Togo': 'Sub-Saharan Africa',
  'Benin': 'Sub-Saharan Africa',
  'Central African Republic': 'Sub-Saharan Africa',
  'South Africa': 'Sub-Saharan Africa',
  'Namibia': 'Sub-Saharan Africa',
  'Botswana': 'Sub-Saharan Africa',
  'Zimbabwe': 'Sub-Saharan Africa',
  'Zambia': 'Sub-Saharan Africa',
  'Malawi': 'Sub-Saharan Africa',
  'Mozambique': 'Sub-Saharan Africa',
  'Madagascar': 'Sub-Saharan Africa',
  
  // Central Asia
  'Kazakhstan': 'Central Asia',
  'Uzbekistan': 'Central Asia',
  'Turkmenistan': 'Central Asia',
  'Kyrgyzstan': 'Central Asia',
  'Tajikistan': 'Central Asia',
  
  // South Asia
  'Afghanistan': 'South Asia',
  'Pakistan': 'South Asia',
  'India': 'South Asia',
  'Nepal': 'South Asia',
  'Bhutan': 'South Asia',
  'Bangladesh': 'South Asia',
  'Sri Lanka': 'South Asia',
  'Maldives': 'South Asia',
  
  // East Asia
  'China': 'East Asia',
  'Mongolia': 'East Asia',
  'North Korea': 'East Asia',
  'South Korea': 'East Asia',
  'Japan': 'East Asia',
  'Taiwan': 'East Asia',
  
  // Southeast Asia
  'Myanmar': 'Southeast Asia',
  'Thailand': 'Southeast Asia',
  'Laos': 'Southeast Asia',
  'Cambodia': 'Southeast Asia',
  'Vietnam': 'Southeast Asia',
  'Malaysia': 'Southeast Asia',
  'Singapore': 'Southeast Asia',
  'Indonesia': 'Southeast Asia',
  'Brunei': 'Southeast Asia',
  'Philippines': 'Southeast Asia',
  'East Timor': 'Southeast Asia',
  
  // Australasia & Oceania
  'Australia': 'Australasia & Oceania',
  'New Zealand': 'Australasia & Oceania',
  'Papua New Guinea': 'Australasia & Oceania',
  'Fiji': 'Australasia & Oceania',
  'Solomon Islands': 'Australasia & Oceania',
  'Vanuatu': 'Australasia & Oceania',
  'New Caledonia': 'Australasia & Oceania'
};

/**
 * Standardizes a country name according to the mapping
 * @param name The original country name
 * @returns The standardized country name or the original if not found in the mapping
 */
export function standardizeCountryName(name: string): string {
  if (!name) return name;
  
  // First check in the regular mapping
  if (countryNameMapping[name]) {
    return countryNameMapping[name];
  }
  
  // Then check if it's a GeoJSON country name
  if (geoJSONtoStandardMapping[name]) {
    return geoJSONtoStandardMapping[name];
  }
  
  // If all else fails, return the original name
  return name;
}

/**
 * Converts a standardized country name to the corresponding GeoJSON name
 * Used for mapping standardized country names to the names used in GeoJSON files
 * @param standardName The standardized country name
 * @returns The corresponding GeoJSON country name
 */
export function getGeoJSONCountryName(standardName: string): string {
  return standardToGeoJSONMapping[standardName] || standardName;
}

/**
 * Converts a standardized country name to the name used in socioeconomic data
 * @param standardName The standardized country name
 * @returns The corresponding socioeconomic data country name
 */
export function getSocioeconomicCountryName(standardName: string): string {
  return standardToSocioeconomicMapping[standardName] || standardName;
}

/**
 * Get ISO 3 country code from country name (if available)
 * This is useful for more advanced mapping applications
 */
export const getCountryCode = (): string | null => {
  // This would be expanded with a full country code mapping if needed
  return null;
};

/**
 * Groups incidents by country and calculates statistics
 */
export interface CountryStats {
  country: string;
  standardizedCountry: string;
  attackCount: number;
  deaths: number;
  wounded: number;
  attackTypes: Record<string, number>;
  groups?: Record<string, number>;
}

export interface HistoricalAttack {
  id: string | number;
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

export const calculateCountryStatistics = (incidents: HistoricalAttack[]): Record<string, CountryStats> => {
  const stats: Record<string, CountryStats> = {};
  
  incidents.forEach(incident => {
    const standardizedCountry = standardizeCountryName(incident.country);
    
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