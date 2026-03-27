# tools/vision.py

import base64

async def analyze_image(
    image_path: str,
    question: str = "Describe this image",
) -> str:
    """Analyze an image using the foundation model's
    vision capabilities."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode()

    # Most modern LLMs support vision natively
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,"
                               f"{image_data}"
                    },
                },
            ],
        }
    ]
    # Use the fast model for vision (usually sufficient)
    # ... API call ...
    # TODO: Wire up to foundation model once ch05 is extracted
    result = "(vision analysis placeholder)"
    return result
