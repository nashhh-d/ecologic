
def recommend_processing(category):
    category = category.lower().replace(" ", "_")

    recommendations = {
        "food": {
            "method": "Anaerobic digestion or composting",
            "reason": "Food waste can be processed biologically under suitable conditions."
        },
        "leaf": {
            "method": "Aerobic composting",
            "reason": "Leaves can decompose into compost under suitable conditions."
        },
        "wood": {
            "method": "Wood-specific composting or biomass processing",
            "reason": "Wood decomposes slowly and may need specialized processing."
        },
        "plastic_bottles": {
            "method": "Plastic recycling assessment",
            "reason": "Plastic bottles are not suitable for ordinary composting or anaerobic digestion."
        },
        "plastic_bags": {
            "method": "Plastic film recycling assessment",
            "reason": "Plastic bags require appropriate material-specific recycling."
        },
        "metal_cans": {
            "method": "Metal recycling",
            "reason": "Metal cans should be separated for material recovery."
        },
        "paper": {
            "method": "Paper recycling assessment",
            "reason": "Clean paper can often be recycled; suitability depends on coatings and contamination."
        },
        "non_biodegradable": {
            "method": "Manual identification and sorting",
            "reason": "The material must be identified before selecting a suitable disposal route."
        }
    }

    return recommendations.get(category, {
        "method": "Manual identification required",
        "reason": "No recommendation is available for this category."
    })


if __name__ == "__main__":
    result = recommend_processing("plastic_bottles")
    print(result)
