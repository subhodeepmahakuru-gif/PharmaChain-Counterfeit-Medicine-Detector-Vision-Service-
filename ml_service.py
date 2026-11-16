
"""
api.py

PharmaChain - Member 4: Counterfeit Medicine Detector

Flask API for ResNet-18 medicine packaging classification.

Endpoints:
    GET  /health
    POST /predict

POST /predict:
    Accepts an image using multipart/form-data.
    Form field name: image

"""

import os
import logging

from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError

from predict import Classifier


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pt"
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff"
}


# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("PharmaChainAPI")


# ==========================================
# FLASK APPLICATION
# ==========================================

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

try:

    clf = Classifier(
        path=MODEL_PATH,
        threshold=0.70
    )

    logger.info("ResNet-18 model loaded successfully.")

except Exception:

    logger.exception("Unable to initialize the classifier.")

    clf = None


# ==========================================
# HEALTH CHECK ENDPOINT
# ==========================================

@app.route("/health", methods=["GET"])
def health():

    if clf is None:

        return jsonify({

            "ok": False,

            "status": "model_unavailable",

            "message": "The trained model could not be loaded."

        }), 503

    return jsonify({

        "ok": True,

        "status": "running",

        "service": "PharmaChain Medicine Detection API",

        "model": "ResNet-18",

        "classes": clf.class_names

    }), 200


# ==========================================
# IMAGE PREDICTION ENDPOINT
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    # Check model availability
    if clf is None:

        return jsonify({

            "ok": False,

            "error": "Model is not available."

        }), 503

    # Check image field
    if "image" not in request.files:

        return jsonify({

            "ok": False,

            "error": (
                "Send an image using a form field named 'image'."
            )

        }), 400

    file = request.files["image"]

    # Check empty filename
    if not file.filename:

        return jsonify({

            "ok": False,

            "error": "No image file was selected."

        }), 400

    # Check file extension
    filename = file.filename

    extension = os.path.splitext(filename)[1].lower()

    if extension not in ALLOWED_EXTENSIONS:

        return jsonify({

            "ok": False,

            "error": "Unsupported image file extension."

        }), 400

    # Read and validate image
    try:

        image_data = file.stream.read(
            MAX_FILE_SIZE + 1
        )

        if not image_data:

            return jsonify({

                "ok": False,

                "error": "The uploaded image is empty."

            }), 400

        if len(image_data) > MAX_FILE_SIZE:

            return jsonify({

                "ok": False,

                "error": "Image exceeds the 10 MB size limit."

            }), 413

        import io

        with Image.open(
            io.BytesIO(image_data)
        ) as img:

            img.verify()

        # Reopen after verify
        with Image.open(
            io.BytesIO(image_data)
        ) as img:

            img.load()

            img = img.convert("RGB")

            # Run classifier
            result = clf.predict(img)

        # Successful prediction
        result["ok"] = True

        result["service"] = "PharmaChain"

        result["authentication_status"] = (
            "NOT_AUTHENTICATED"
        )

        logger.info(
            "Image prediction completed: %s",
            result.get("prediction")
        )

        return jsonify(result), 200

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ) as error:

        logger.warning(
            "Invalid image upload: %s",
            str(error)
        )

        return jsonify({

            "ok": False,

            "error": "That file is not a readable image."

        }), 400

    except Exception:

        logger.exception(
            "Unexpected prediction error."
        )

        return jsonify({

            "ok": False,

            "error": "Image prediction failed."

        }), 500


# ==========================================
# FILE TOO LARGE ERROR
# ==========================================

@app.errorhandler(413)
def file_too_large(error):

    return jsonify({

        "ok": False,

        "error": "Uploaded file exceeds the 10 MB limit."

    }), 413


# ==========================================
# INVALID ROUTE
# ==========================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "ok": False,

        "error": "API endpoint not found."

    }), 404


# ==========================================
# MAIN ENTRY POINT
# ==========================================

if __name__ == "__main__":

    if clf is None:

        raise SystemExit(
            "Cannot start API: model.pt could not be loaded."
        )

    logger.info(
        "Starting PharmaChain API on port 5001..."
    )

    app.run(
        host="0.0.0.0",
        port=5001,
        debug=False
    )
