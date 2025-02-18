class Preprocessor:
    def get_region_name(self, region_id):
        # Mapping of region IDs to names
        region_names = {
            0: "North America",
            1: "South America",
            2: "Western Europe",
            3: "Eastern Europe",
            4: "Middle East",
            5: "North Africa",
            6: "Sub-Saharan Africa",
            7: "Central Asia",
            8: "South Asia",
            9: "East Asia",
            10: "Southeast Asia",
            11: "Oceania"
        }
        return region_names.get(region_id, f"Region {region_id}")

preprocessor = Preprocessor() 