
"""
PharmaChain - End-to-End Integration Testing

Tests:
1. ML service health
2. Genuine medicine image prediction
3. Counterfeit medicine image prediction
4. Member 2 backend health
5. Batch registration
6. Image verification through backend
7. Batch verdict retrieval

Required services:
    ML service:      http://127.0.0.1:5001
    Backend service: http://127.0.0.1:5000

Required dataset:
    data/test/genuine/
    data/test/counterfeit/
"""

import glob
import os
import sys
import time

import requests


# ==========================================
# CONFIGURATION
# ==========================================

ML_URL = "http://127.0.0.1:5001"
API_URL = "http://127.0.0.1:5000"

TIMEOUT = 60

TEST_DIR = "data/test"


# ==========================================
# HELPER FUNCTIONS
# ==========================================

def first_image(pattern):
    """Return the first available image matching a pattern."""

    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No test images found for: {pattern}\n"
            "Run the dataset preparation step first."
        )

    return files[0]


def check_service(url, service_name):
    """Check whether a service is running."""

    try:

        response = requests.get(
            url,
            timeout=5
        )

        response.raise_for_status()

        print(f"[OK] {service_name} is running.")

        return True

    except requests.exceptions.RequestException as error:

        print(
            f"[ERROR] {service_name} is unavailable: {error}"
        )

        return False


def post_prediction(image_path):
    """Send an image to the ML prediction service."""

    if not os.path.isfile(image_path):
        raise FileNotFoundError(image_path)

    with open(image_path, "rb") as image:

        response = requests.post(
            ML_URL + "/predict",
            files={
                "image": (
                    os.path.basename(image_path),
                    image
                )
            },
            timeout=TIMEOUT
        )

    response.raise_for_status()

    return response.json()


def register_batch(batch_id):
    """Register a batch with Member 2's backend."""

    payload = {

        "batch_id": batch_id,

        "drug_name": "Vaccine X",

        "location": "Factory"

    }

    response = requests.post(
        API_URL + "/register-batch",
        json=payload,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    return response.json()


def verify_image(batch_id, image_path):
    """Send an image to Member 2's verification endpoint."""

    with open(image_path, "rb") as image:

        response = requests.post(
            API_URL + "/verify-image",

            data={
                "batch_id": batch_id
            },

            files={
                "image": (
                    os.path.basename(image_path),
                    image
                )
            },

            timeout=TIMEOUT
        )

    response.raise_for_status()

    return response.json()


def get_batch_verdict(batch_id):
    """Retrieve the batch verification result."""

    response = requests.get(
        f"{API_URL}/verify/{batch_id}",
        timeout=TIMEOUT
    )

    response.raise_for_status()

    return response.json()


# ==========================================
# MAIN TEST PIPELINE
# ==========================================

def main():

    print("\n==========================================")
    print(" PHARMACHAIN INTEGRATION TEST")
    print("==========================================\n")

    # --------------------------------------
    # STEP 1: Find test images
    # --------------------------------------

    print("[STEP 1] Finding test images...")

    try:

        genuine_image = first_image(
            os.path.join(
                TEST_DIR,
                "genuine",
                "*"
            )
        )

        counterfeit_image = first_image(
            os.path.join(
                TEST_DIR,
                "counterfeit",
                "*"
            )
        )

    except FileNotFoundError as error:

        sys.exit(str(error))

    print("[OK] Genuine image:", genuine_image)

    print("[OK] Counterfeit image:", counterfeit_image)

    # --------------------------------------
    # STEP 2: Check ML service
    # --------------------------------------

    print("\n[STEP 2] Checking ML service...")

    if not check_service(
        ML_URL + "/health",
        "ML service"
    ):

        sys.exit(
            "Start the ML service first using:\n"
            "python ml_service.py"
        )

    # --------------------------------------
    # STEP 3: Test ML predictions
    # --------------------------------------

    print("\n[STEP 3] Testing ML predictions...")

    try:

        genuine_result = post_prediction(
            genuine_image
        )

        print("\nGenuine image prediction:")

        print(genuine_result)

        counterfeit_result = post_prediction(
            counterfeit_image
        )

        print("\nCounterfeit image prediction:")

        print(counterfeit_result)

    except requests.exceptions.RequestException as error:

        sys.exit(
            f"ML prediction request failed: {error}"
        )

    except ValueError as error:

        sys.exit(
            f"Invalid ML service response: {error}"
        )

    # --------------------------------------
    # STEP 4: Check backend service
    # --------------------------------------

    print("\n[STEP 4] Checking Member 2 backend...")

    if not check_service(
        API_URL + "/health",
        "Member 2 backend"
    ):

        print(
            "\nML tests completed successfully."
        )

        print(
            "Backend tests skipped because "
            "Member 2's backend is unavailable."
        )

        return

    # --------------------------------------
    # STEP 5: Generate batch IDs
    # --------------------------------------

    print("\n[STEP 5] Registering test batches...")

    timestamp = int(time.time())

    test_batches = {

        "genuine": f"IMG-{timestamp}-G",

        "counterfeit": f"IMG-{timestamp}-C"

    }

    # --------------------------------------
    # STEP 6: Register batches
    # --------------------------------------

    for name, batch_id in test_batches.items():

        try:

            result = register_batch(batch_id)

            print(
                f"[OK] Registered {name} batch: "
                f"{batch_id}"
            )

            print("Response:", result)

        except requests.exceptions.RequestException as error:

            print(
                f"[ERROR] Could not register "
                f"{name} batch: {error}"
            )

            return

    # --------------------------------------
    # STEP 7: Verify medicine images
    # --------------------------------------

    print("\n[STEP 7] Testing backend image verification...")

    test_images = {

        "genuine": genuine_image,

        "counterfeit": counterfeit_image

    }

    for name, image_path in test_images.items():

        batch_id = test_batches[name]

        try:

            result = verify_image(
                batch_id,
                image_path
            )

            print(
                f"\n{name.upper()} image backend response:"
            )

            print(result)

            if result.get("ok") is False:

                print(
                    "[WARNING] Backend reported "
                    "verification failure."
                )

            # ----------------------------------
            # STEP 8: Retrieve batch verdict
            # ----------------------------------

            verdict_data = get_batch_verdict(
                batch_id
            )

            verdict = verdict_data.get(
                "verdict",
                "not_available"
            )

            print(
                f"\n{name.upper()} batch verdict: "
                f"{verdict}"
            )

        except requests.exceptions.RequestException as error:

            print(
                f"[ERROR] Backend verification failed "
                f"for {name}: {error}"
            )

        except ValueError as error:

            print(
                f"[ERROR] Invalid backend JSON response: "
                f"{error}"
            )

    # --------------------------------------
    # STEP 9: Completion
    # --------------------------------------

    print("\n==========================================")
    print(" PHARMACHAIN INTEGRATION TEST COMPLETED")
    print("==========================================")

    print("\nML service:", ML_URL)

    print("Backend service:", API_URL)

    print("\nReview the returned predictions and batch records.")

    print(
        "Note: Model predictions and batch verdicts "
        "must not be treated as proof of medicine authenticity."
    )


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":

    main()
