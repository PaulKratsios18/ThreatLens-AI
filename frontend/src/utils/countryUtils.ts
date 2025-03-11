// Map GTD country names to standardized names that match GeoJSON
export const countryNameMapping: Record<string, string> = {
  // North America
  'United States': 'USA',
  'Canada': 'Canada',
  'Mexico': 'Mexico',
  
  // Central America & Caribbean
  'Guatemala': 'Guatemala',
  'El Salvador': 'El Salvador',
  'Honduras': 'Honduras',
  'Nicaragua': 'Nicaragua',
  'Costa Rica': 'Costa Rica',
  'Panama': 'Panama',
  'Cuba': 'Cuba',
  'Jamaica': 'Jamaica',
  'Haiti': 'Haiti',
  'Dominican Republic': 'Dominican Republic',
  'Trinidad and Tobago': 'Trinidad and Tobago',
  
  // South America
  'Colombia': 'Colombia',
  'Venezuela': 'Venezuela',
  'Ecuador': 'Ecuador',
  'Peru': 'Peru',
  'Bolivia': 'Bolivia',
  'Brazil': 'Brazil',
  'Paraguay': 'Paraguay',
  'Chile': 'Chile',
  'Argentina': 'Argentina',
  
  // Western Europe
  'United Kingdom': 'England',
  'Ireland': 'Ireland',
  'France': 'France',
  'Spain': 'Spain',
  'Portugal': 'Portugal',
  'Belgium': 'Belgium',
  'Netherlands': 'Netherlands',
  'Germany': 'Germany',
  'Switzerland': 'Switzerland',
  'Austria': 'Austria',
  'Italy': 'Italy',
  'Sweden': 'Sweden',
  'Norway': 'Norway',
  'Denmark': 'Denmark',
  'Finland': 'Finland',
  'Iceland': 'Iceland',
  
  // Eastern Europe
  'Greece': 'Greece',
  'Cyprus': 'Cyprus',
  'Turkey': 'Turkey',
  'Russia': 'Russia',
  'Poland': 'Poland',
  'Czech Republic': 'Czech Republic',
  'Slovakia': 'Slovakia',
  'Hungary': 'Hungary',
  'Romania': 'Romania',
  'Bulgaria': 'Bulgaria',
  'Albania': 'Albania',
  'Kosovo': 'Kosovo',
  'Serbia': 'Serbia',
  'Croatia': 'Croatia',
  'Bosnia-Herzegovina': 'Bosnia and Herzegovina',
  'Macedonia': 'North Macedonia',
  'Ukraine': 'Ukraine',
  'Belarus': 'Belarus',
  'Moldova': 'Moldova',
  'Lithuania': 'Lithuania',
  'Latvia': 'Latvia',
  'Estonia': 'Estonia',
  
  // Middle East & North Africa
  'Morocco': 'Morocco',
  'Algeria': 'Algeria',
  'Tunisia': 'Tunisia',
  'Libya': 'Libya',
  'Egypt': 'Egypt',
  'Israel': 'Israel',
  'Palestine': 'Palestine',
  'West Bank and Gaza Strip': 'Palestine',
  'Lebanon': 'Lebanon',
  'Syria': 'Syria',
  'Iraq': 'Iraq',
  'Iran': 'Iran',
  'Jordan': 'Jordan',
  'Kuwait': 'Kuwait',
  'Saudi Arabia': 'Saudi Arabia',
  'Bahrain': 'Bahrain',
  'Qatar': 'Qatar',
  'United Arab Emirates': 'United Arab Emirates',
  'Yemen': 'Yemen',
  
  // Sub-Saharan Africa
  'Mauritania': 'Mauritania',
  'Mali': 'Mali',
  'Niger': 'Niger',
  'Chad': 'Chad',
  'Sudan': 'Sudan',
  'Eritrea': 'Eritrea',
  'Ethiopia': 'Ethiopia',
  'Somalia': 'Somalia',
  'Djibouti': 'Djibouti',
  'Senegal': 'Senegal',
  'Gambia': 'Gambia',
  'Guinea-Bissau': 'Guinea-Bissau',
  'Guinea': 'Guinea',
  'Sierra Leone': 'Sierra Leone',
  'Liberia': 'Liberia',
  'Ivory Coast': 'Ivory Coast',
  'Cote d\'Ivoire': 'Ivory Coast',
  'Ghana': 'Ghana',
  'Togo': 'Togo',
  'Benin': 'Benin',
  'Nigeria': 'Nigeria',
  'Cameroon': 'Cameroon',
  'Central African Republic': 'Central African Republic',
  'South Sudan': 'South Sudan',
  'Uganda': 'Uganda',
  'Kenya': 'Kenya',
  'Tanzania': 'Tanzania',
  'Burundi': 'Burundi',
  'Rwanda': 'Rwanda',
  'Democratic Republic of the Congo': 'Democratic Republic of the Congo',
  'Republic of the Congo': 'Republic of Congo',
  'Gabon': 'Gabon',
  'Angola': 'Angola',
  'Zambia': 'Zambia',
  'Malawi': 'Malawi',
  'Mozambique': 'Mozambique',
  'Zimbabwe': 'Zimbabwe',
  'Namibia': 'Namibia',
  'Botswana': 'Botswana',
  'South Africa': 'South Africa',
  'Lesotho': 'Lesotho',
  'Swaziland': 'Swaziland',
  'Eswatini': 'Swaziland',
  
  // South Asia
  'Afghanistan': 'Afghanistan',
  'Pakistan': 'Pakistan',
  'India': 'India',
  'Nepal': 'Nepal',
  'Bangladesh': 'Bangladesh',
  'Bhutan': 'Bhutan',
  'Sri Lanka': 'Sri Lanka',
  'Maldives': 'Maldives',
  
  // Central Asia
  'Kazakhstan': 'Kazakhstan',
  'Kyrgyzstan': 'Kyrgyzstan',
  'Uzbekistan': 'Uzbekistan',
  'Turkmenistan': 'Turkmenistan',
  'Tajikistan': 'Tajikistan',
  'Mongolia': 'Mongolia',
  
  // East Asia
  'China': 'China',
  'North Korea': 'North Korea',
  'South Korea': 'South Korea',
  'Japan': 'Japan',
  'Taiwan': 'Taiwan',
  
  // Southeast Asia
  'Myanmar': 'Myanmar',
  'Thailand': 'Thailand',
  'Laos': 'Laos',
  'Cambodia': 'Cambodia',
  'Vietnam': 'Vietnam',
  'Malaysia': 'Malaysia',
  'Singapore': 'Singapore',
  'Indonesia': 'Indonesia',
  'Philippines': 'Philippines',
  'East Timor': 'Timor-Leste',
  'Timor-Leste': 'Timor-Leste',
  
  // Australasia & Oceania
  'Australia': 'Australia',
  'New Zealand': 'New Zealand',
  'Papua New Guinea': 'Papua New Guinea',
  'Solomon Islands': 'Solomon Islands',
  'Fiji': 'Fiji'
};

/**
 * Standardizes country names to match GeoJSON data
 * @param countryName The original country name
 * @returns The standardized country name
 */
export const standardizeCountryName = (countryName: string): string => {
  return countryNameMapping[countryName] || countryName;
};

/**
 * Get ISO 3 country code from country name (if available)
 * This is useful for more advanced mapping applications
 */
export const getCountryCode = (countryName: string): string | null => {
  const standardizedName = standardizeCountryName(countryName);
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