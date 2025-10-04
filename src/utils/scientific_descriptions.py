"""
Scientific descriptions and planet comparisons for exoplanet predictions
"""

import numpy as np
from typing import Dict, List, Any

class ExoplanetDescriptor:
    def __init__(self):
        # Known exoplanet characteristics for comparison
        self.reference_planets = {
            "jupiter": {
                "orbital_period_days": 4332.59,
                "transit_depth_ppm": 10000,  # Approximate for Jupiter transiting Sun
                "transit_duration_hours": 29.6,
                "description": "Gas giant similar to our Solar System's Jupiter",
                "characteristics": ["Gas giant", "Large atmosphere", "Strong magnetic field", "Many moons"]
            },
            "saturn": {
                "orbital_period_days": 10759.22,
                "transit_depth_ppm": 7500,
                "transit_duration_hours": 40.5,
                "description": "Ringed gas giant similar to Saturn",
                "characteristics": ["Ringed planet", "Gas giant", "Low density", "Many moons"]
            },
            "neptune": {
                "orbital_period_days": 60190.03,
                "transit_depth_ppm": 4000,
                "transit_duration_hours": 47.0,
                "description": "Ice giant similar to Neptune",
                "characteristics": ["Ice giant", "Strong winds", "Blue color", "Dense atmosphere"]
            },
            "uranus": {
                "orbital_period_days": 30688.5,
                "transit_depth_ppm": 3500,
                "transit_duration_hours": 42.0,
                "description": "Ice giant similar to Uranus",
                "characteristics": ["Ice giant", "Sideways rotation", "Faint rings", "Cold atmosphere"]
            },
            "earth": {
                "orbital_period_days": 365.25,
                "transit_depth_ppm": 84,  # Earth transiting Sun
                "transit_duration_hours": 13.0,
                "description": "Rocky planet similar to Earth",
                "characteristics": ["Rocky planet", "Liquid water potential", "Atmosphere", "Life potential"]
            },
            "mars": {
                "orbital_period_days": 686.98,
                "transit_depth_ppm": 23,
                "transit_duration_hours": 14.5,
                "description": "Rocky planet similar to Mars",
                "characteristics": ["Rocky planet", "Thin atmosphere", "Red color", "Polar caps"]
            },
            "venus": {
                "orbital_period_days": 224.7,
                "transit_depth_ppm": 65,
                "transit_duration_hours": 11.5,
                "description": "Rocky planet similar to Venus",
                "characteristics": ["Rocky planet", "Dense atmosphere", "Greenhouse effect", "Hot surface"]
            },
            "mercury": {
                "orbital_period_days": 87.97,
                "transit_depth_ppm": 12,
                "transit_duration_hours": 8.0,
                "description": "Rocky planet similar to Mercury",
                "characteristics": ["Rocky planet", "No atmosphere", "Extreme temperatures", "Fast orbit"]
            }
        }
        
        # Stellar types
        self.stellar_types = {
            "O": {"temp_range": (30000, 50000), "radius_range": (6.6, 20), "color": "Blue", "lifetime": "Very short"},
            "B": {"temp_range": (10000, 30000), "radius_range": (1.8, 6.6), "color": "Blue-white", "lifetime": "Short"},
            "A": {"temp_range": (7500, 10000), "radius_range": (1.4, 1.8), "color": "White", "lifetime": "Short-medium"},
            "F": {"temp_range": (6000, 7500), "radius_range": (1.15, 1.4), "color": "Yellow-white", "lifetime": "Medium"},
            "G": {"temp_range": (5200, 6000), "radius_range": (0.96, 1.15), "color": "Yellow", "lifetime": "Long (Sun-like)"},
            "K": {"temp_range": (3700, 5200), "radius_range": (0.7, 0.96), "color": "Orange", "lifetime": "Very long"},
            "M": {"temp_range": (2400, 3700), "radius_range": (0.1, 0.7), "color": "Red", "lifetime": "Extremely long"}
        }

    def get_stellar_classification(self, temp_K: float, radius_solar: float) -> Dict[str, Any]:
        """Classify the host star based on temperature and radius"""
        for spectral_type, properties in self.stellar_types.items():
            if properties["temp_range"][0] <= temp_K <= properties["temp_range"][1]:
                if properties["radius_range"][0] <= radius_solar <= properties["radius_range"][1]:
                    return {
                        "type": spectral_type,
                        "color": properties["color"],
                        "lifetime": properties["lifetime"],
                        "temperature": temp_K,
                        "radius": radius_solar
                    }
        return {"type": "Unknown", "color": "Unknown", "lifetime": "Unknown", "temperature": temp_K, "radius": radius_solar}

    def find_most_similar_planet(self, orbital_period: float, transit_depth: float, transit_duration: float) -> Dict[str, Any]:
        """Find the most similar planet in our solar system"""
        similarities = {}
        
        for planet_name, planet_data in self.reference_planets.items():
            # Calculate similarity score (lower is more similar)
            period_diff = abs(np.log10(orbital_period) - np.log10(planet_data["orbital_period_days"]))
            depth_diff = abs(np.log10(transit_depth) - np.log10(planet_data["transit_depth_ppm"]))
            duration_diff = abs(np.log10(transit_duration) - np.log10(planet_data["transit_duration_hours"]))
            
            similarity_score = (period_diff + depth_diff + duration_diff) / 3
            similarities[planet_name] = {
                "score": similarity_score,
                "data": planet_data
            }
        
        # Find the most similar planet
        most_similar = min(similarities.items(), key=lambda x: x[1]["score"])
        return most_similar

    def generate_scientific_description(self, prediction: str, probabilities: Dict[str, float], 
                                      orbital_period: float, transit_depth: float, 
                                      transit_duration: float, stellar_temp: float, 
                                      stellar_radius: float, snr: float = None) -> Dict[str, Any]:
        """Generate detailed scientific description of the exoplanet"""
        
        # Get stellar classification
        stellar_info = self.get_stellar_classification(stellar_temp, stellar_radius)
        
        # Find most similar planet
        similar_planet = self.find_most_similar_planet(orbital_period, transit_depth, transit_duration)
        
        # Calculate planet radius estimate (rough approximation)
        transit_depth_normalized = transit_depth / 1000000  # Convert ppm to fraction
        estimated_planet_radius = float(np.sqrt(transit_depth_normalized) * stellar_radius)
        
        # Determine planet type based on characteristics
        planet_type = self._classify_planet_type(estimated_planet_radius, orbital_period, stellar_temp)
        
        # Generate detailed description
        description = self._generate_detailed_description(
            prediction, probabilities, orbital_period, transit_depth, transit_duration,
            stellar_temp, stellar_radius, snr, stellar_info, similar_planet, 
            estimated_planet_radius, planet_type
        )
        
        return description

    def _classify_planet_type(self, planet_radius: float, orbital_period: float, stellar_temp: float) -> Dict[str, Any]:
        """Classify the planet type based on size and orbital characteristics"""
        if planet_radius < 0.5:
            return {"type": "Sub-Earth", "description": "Small rocky world"}
        elif planet_radius < 1.25:
            return {"type": "Earth-like", "description": "Rocky planet similar to Earth"}
        elif planet_radius < 2.5:
            return {"type": "Super-Earth", "description": "Large rocky planet"}
        elif planet_radius < 6.0:
            return {"type": "Mini-Neptune", "description": "Gas dwarf with thick atmosphere"}
        elif planet_radius < 12.0:
            return {"type": "Gas Giant", "description": "Large gas planet"}
        else:
            return {"type": "Super-Jupiter", "description": "Very large gas giant"}

    def _generate_detailed_description(self, prediction: str, probabilities: Dict[str, float],
                                     orbital_period: float, transit_depth: float, transit_duration: float,
                                     stellar_temp: float, stellar_radius: float, snr: float,
                                     stellar_info: Dict, similar_planet: tuple, 
                                     estimated_planet_radius: float, planet_type: Dict) -> Dict[str, Any]:
        """Generate the detailed scientific description"""
        
        similar_name, similar_info = similar_planet
        similar_data = similar_info["data"]
        
        # Generate confidence explanation
        confidence_explanation = self._get_confidence_explanation(probabilities, snr)
        
        # Generate stellar description
        stellar_description = self._get_stellar_description(stellar_info)
        
        # Generate orbital description
        orbital_description = self._get_orbital_description(orbital_period, similar_name, similar_data)
        
        # Generate atmospheric description
        atmospheric_description = self._get_atmospheric_description(planet_type, estimated_planet_radius, stellar_temp)
        
        # Generate habitability assessment
        habitability = self._assess_habitability(estimated_planet_radius, orbital_period, stellar_temp, stellar_radius)
        
        # Generate why/why not explanation
        why_explanation = self._get_why_explanation(prediction, planet_type, stellar_info, orbital_period)
        
        return {
            "prediction": prediction,
            "confidence_explanation": confidence_explanation,
            "stellar_description": stellar_description,
            "planet_type": planet_type,
            "estimated_radius": estimated_planet_radius,
            "orbital_description": orbital_description,
            "atmospheric_description": atmospheric_description,
            "habitability": habitability,
            "why_explanation": why_explanation,
            "similar_planet": {
                "name": similar_name.title(),
                "description": similar_data["description"],
                "characteristics": similar_data["characteristics"]
            },
            "scientific_notes": self._get_scientific_notes(transit_depth, transit_duration, snr)
        }

    def _get_confidence_explanation(self, probabilities: Dict[str, float], snr: float) -> str:
        """Explain the confidence level"""
        max_prob = max(probabilities.values())
        
        if max_prob >= 0.8:
            return f"The model is highly confident ({(max_prob*100):.1f}% probability) in this classification. "
        elif max_prob >= 0.6:
            return f"The model shows moderate confidence ({(max_prob*100):.1f}% probability) in this classification. "
        else:
            return f"The model shows low confidence ({(max_prob*100):.1f}% probability) in this classification. "
        
        if snr:
            if snr > 20:
                return "The high signal-to-noise ratio indicates very reliable measurements. "
            elif snr > 10:
                return "The good signal-to-noise ratio indicates reliable measurements. "
            else:
                return "The low signal-to-noise ratio suggests some uncertainty in the measurements. "

    def _get_stellar_description(self, stellar_info: Dict) -> str:
        """Generate stellar description"""
        return f"The host star is classified as a {stellar_info['type']}-type star with a {stellar_info['color']} color. " \
               f"With a temperature of {stellar_info['temperature']:.0f}K and radius of {stellar_info['radius']:.2f} solar radii, " \
               f"this star has a {stellar_info['lifetime']} main sequence lifetime."

    def _get_orbital_description(self, orbital_period: float, similar_name: str, similar_data: Dict) -> str:
        """Generate orbital description"""
        if orbital_period < 10:
            return f"This exoplanet orbits very close to its star with a period of {orbital_period:.2f} days, " \
                   f"similar to {similar_name.title()}'s characteristics. Such close orbits often result in tidal locking."
        elif orbital_period < 100:
            return f"This exoplanet has a moderate orbital period of {orbital_period:.2f} days, " \
                   f"showing similarities to {similar_name.title()}. This distance allows for more stable atmospheric conditions."
        else:
            return f"This exoplanet has a long orbital period of {orbital_period:.2f} days, " \
                   f"similar to {similar_name.title()}. Such distant orbits provide more stable environments."

    def _get_atmospheric_description(self, planet_type: Dict, radius: float, stellar_temp: float) -> str:
        """Generate atmospheric description"""
        if planet_type["type"] in ["Earth-like", "Super-Earth"]:
            return f"As a {planet_type['type'].lower()}, this planet likely has a rocky surface with a substantial atmosphere. " \
                   f"The atmospheric composition would depend on the planet's formation history and current conditions."
        elif planet_type["type"] in ["Gas Giant", "Super-Jupiter"]:
            return f"As a {planet_type['type'].lower()}, this planet has a massive hydrogen-helium atmosphere " \
                   f"with no solid surface. The atmosphere becomes denser with depth, eventually reaching metallic hydrogen."
        else:
            return f"This {planet_type['type'].lower()} likely has a thick atmosphere that may contain hydrogen, helium, " \
                   f"and heavier elements. The exact composition depends on the planet's formation and evolution."

    def _assess_habitability(self, radius: float, orbital_period: float, stellar_temp: float, stellar_radius: float) -> Dict[str, Any]:
        """Assess potential habitability"""
        # Calculate habitable zone (simplified)
        stellar_luminosity = (stellar_radius ** 2) * ((stellar_temp / 5778) ** 4)
        habitable_zone_inner = float(0.95 * np.sqrt(stellar_luminosity))
        habitable_zone_outer = float(1.37 * np.sqrt(stellar_luminosity))
        
        # Calculate current orbital distance (simplified)
        orbital_distance = float((orbital_period / 365.25) ** (2/3))
        
        is_in_habitable_zone = habitable_zone_inner <= orbital_distance <= habitable_zone_outer
        is_rocky = radius < 2.5
        
        if is_in_habitable_zone and is_rocky:
            habitability_score = "High"
            habitability_explanation = "This planet is located within the habitable zone and has a rocky composition, making it a promising candidate for hosting life as we know it."
        elif is_in_habitable_zone and not is_rocky:
            habitability_score = "Medium"
            habitability_explanation = "While located in the habitable zone, the gas giant nature of this planet makes it less suitable for life as we know it."
        elif not is_in_habitable_zone and is_rocky:
            habitability_score = "Low"
            habitability_explanation = "Although this is a rocky planet, its distance from the star places it outside the habitable zone."
        else:
            habitability_score = "Very Low"
            habitability_explanation = "This gas giant is located outside the habitable zone, making it unsuitable for life as we know it."
        
        return {
            "score": habitability_score,
            "explanation": habitability_explanation,
            "in_habitable_zone": bool(is_in_habitable_zone),
            "is_rocky": bool(is_rocky)
        }

    def _get_why_explanation(self, prediction: str, planet_type: Dict, stellar_info: Dict, orbital_period: float) -> str:
        """Explain why the planet was classified this way"""
        if prediction == "CONFIRMED":
            return f"This planet was classified as CONFIRMED because it exhibits clear transit signatures consistent with a {planet_type['type'].lower()}. " \
                   f"The orbital characteristics and stellar environment provide strong evidence for planetary nature."
        elif prediction == "CANDIDATE":
            return f"This object was classified as a CANDIDATE because it shows promising planetary characteristics but requires further confirmation. " \
                   f"The {planet_type['type'].lower()} classification suggests it could be a genuine exoplanet pending additional observations."
        else:  # FALSE_POSITIVE
            return f"This object was classified as a FALSE_POSITIVE because the observed characteristics are more consistent with stellar variability, " \
                   f"binary star systems, or instrumental effects rather than a genuine planetary transit."

    def _get_scientific_notes(self, transit_depth: float, transit_duration: float, snr: float) -> List[str]:
        """Generate scientific notes"""
        notes = []
        
        if transit_depth > 10000:
            notes.append("The very deep transit suggests a large planet or a small star.")
        elif transit_depth < 100:
            notes.append("The shallow transit indicates a small planet or requires very precise measurements.")
        
        if transit_duration > 20:
            notes.append("The long transit duration suggests either a large star or a distant orbit.")
        elif transit_duration < 1:
            notes.append("The short transit duration indicates either a small star or a very close orbit.")
        
        if snr and snr < 5:
            notes.append("Low signal-to-noise ratio suggests the detection may be marginal.")
        elif snr and snr > 50:
            notes.append("Excellent signal-to-noise ratio indicates very reliable detection.")
        
        return notes
