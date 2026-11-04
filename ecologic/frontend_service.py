
import os
from dataclasses import dataclass

import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO


class BackendUnavailableError(Exception):
    pass


class InvalidImageError(Exception):
    pass


@dataclass
class BackendStatus:
    analysis_ready: bool
    message: str


class FrontendService:
    def __init__(self):
        self.base_url = os.getenv(
            "ECOLOGIC_API_URL",
            "http://127.0.0.1:8000"
        ).rstrip("/")

    def get_backend_status(self):
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            data = response.json()

            ready = data.get("model_checkpoint_exists", False)

            return BackendStatus(
                analysis_ready=ready,
                message=(
                    "Connected to ECOLOGIC backend."
                    if ready
                    else "Backend is running, but the model checkpoint is missing."
                )
            )

        except requests.RequestException:
            return BackendStatus(
                analysis_ready=False,
                message="Cannot connect to the ECOLOGIC backend."
            )

    def analyze_uploaded_image(self, filename, image_bytes):
        try:
            response = requests.post(
                f"{self.base_url}/analyze",
                files={
                    "file": (
                        filename,
                        image_bytes,
                        "image/jpeg"
                    )
                },
                timeout=120
            )
            response.raise_for_status()
            return response.json()

        except requests.RequestException as error:
            raise BackendUnavailableError(
                f"Could not analyze image: {error}"
            ) from error


def validate_image(filename, image_bytes):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise InvalidImageError(
            "Please upload a JPG, JPEG, or PNG image."
        )

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()

        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size

        if width <= 0 or height <= 0:
            raise InvalidImageError("The image dimensions are invalid.")

        return width, height

    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError(
            "The uploaded file is not a valid image."
        ) from error
