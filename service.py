
from ecologic.inference import predict_image
from ecologic.recommender import recommend_processing


def analyze_image(image, checkpoint_path="outputs/ecologic_mobilenetv3.pth"):
    """
    Classify a waste image and recommend a processing pathway.
    """

    # Step 1: Predict the waste category using the trained CNN.
    prediction = predict_image(
        image=image,
        checkpoint_path=checkpoint_path,
        top_k=3
    )

    category = prediction["label"]
    confidence = prediction["confidence_percent"] / 100.0

    # Step 2: Generate a recommendation using the predicted category.
    recommendation = recommend_processing(category)

    # Step 3: Return everything in one structured result.
    return {
        "predicted_category": category,
        "confidence_percent": prediction["confidence_percent"],
        "model": prediction["model"],
        "top_predictions": prediction["top_predictions"],
        "recommended_method": recommendation["method"],
        "recommendation_reason": recommendation["reason"],
        "requires_review": confidence < 0.60,
    }


if __name__ == "__main__":
    from PIL import Image

    image_path = input("Enter the path to a waste image: ").strip().strip('"')

    with Image.open(image_path) as img:
        img.load()
        result = analyze_image(img)

    for key, value in result.items():
        print(f"{key}: {value}")
