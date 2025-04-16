from typing import Dict

# Country name mapping to standardized names
COUNTRY_NAME_MAPPING: Dict[str, str] = {
    # North America
    "United States": "USA",
    "United States of America": "USA",
    "America": "USA",
    "USA": "USA",
    "US": "USA",
    "Canada": "Canada",
    "Mexico": "Mexico",
    
    # Europe
    "United Kingdom": "United Kingdom",
    "UK": "United Kingdom",
    "Great Britain": "United Kingdom",
    "England": "United Kingdom",
    "France": "France",
    "Germany": "Germany",
    "Italy": "Italy",
    "Spain": "Spain",
    
    # Asia
    "China": "China",
    "People's Republic of China": "China",
    "Japan": "Japan",
    "India": "India",
    "Republic of India": "India",
    "Russia": "Russia",
    "Russian Federation": "Russia",
    
    # Middle East
    "Iran": "Iran",
    "Islamic Republic of Iran": "Iran",
    "Iraq": "Iraq",
    "Saudi Arabia": "Saudi Arabia",
    "KSA": "Saudi Arabia",
    "Israel": "Israel",
    "Syria": "Syria",
    "Syrian Arab Republic": "Syria",
    
    # Africa
    "South Africa": "South Africa",
    "Nigeria": "Nigeria",
    "Egypt": "Egypt",
    "Kenya": "Kenya",
    
    # South America
    "Brazil": "Brazil",
    "Argentina": "Argentina",
    "Colombia": "Colombia",
    "Venezuela": "Venezuela",
    
    # Oceania
    "Australia": "Australia",
    "New Zealand": "New Zealand",
    
    # Others with variations
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
}

# Map from standardized names to socioeconomic data names
STANDARD_TO_SOCIOECONOMIC_MAPPING: Dict[str, str] = {
    "USA": "United States",
    "United Kingdom": "United Kingdom",
    "DR Congo": "Democratic Republic of the Congo"
}

# Map from standardized names to GeoJSON country names
STANDARD_TO_GEOJSON_MAPPING: Dict[str, str] = {
    "USA": "United States",
    "Dominican Republic": "Dominican Rep.",
    "DR Congo": "Dem. Rep. Congo",
    "Central African Republic": "Central African Rep.",
    "South Sudan": "S. Sudan",
    "Ivory Coast": "Côte d'Ivoire",
    "Bosnia and Herzegovina": "Bosnia and Herz."
}

def standardize_country_name(name: str) -> str:
    """
    Standardize a country name according to the mapping
    
    Args:
        name: The original country name
        
    Returns:
        The standardized country name or the original if not found in the mapping
    """
    if not name:
        return name
    
    return COUNTRY_NAME_MAPPING.get(name, name)

def get_socioeconomic_country_name(standard_name: str) -> str:
    """
    Convert a standardized country name to the name used in socioeconomic data
    
    Args:
        standard_name: The standardized country name
        
    Returns:
        The corresponding socioeconomic data country name
    """
    return STANDARD_TO_SOCIOECONOMIC_MAPPING.get(standard_name, standard_name)

def get_geojson_country_name(standard_name: str) -> str:
    """
    Convert a standardized country name to the corresponding GeoJSON name
    
    Args:
        standard_name: The standardized country name
        
    Returns:
        The corresponding GeoJSON country name
    """
    return STANDARD_TO_GEOJSON_MAPPING.get(standard_name, standard_name) 