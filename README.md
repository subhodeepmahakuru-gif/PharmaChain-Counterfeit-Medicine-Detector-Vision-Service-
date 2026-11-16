# PharmaChain: Counterfeit Medicine Detector (Vision Service)

The computer vision component ("the eyes") of the **PharmaChain** system. This service uses a **ResNet-18** transfer learning model to classify photographs of medicine packaging as either `genuine` or `counterfeit`, exposing predictions via a lightweight Flask REST API.

## System Architecture

```text
[Packaging Photo] ──► [Member 5 Portal] ──► [Member 2 API] ──► [Vision Service (Port 5001)]
                                                                       │
                                                   Returns {"label", "confidence"}
                                                                       │
                    (If label == "counterfeit" && confidence >= 0.7) ──▼
                                                         [Blockchain State Updated]


Project Structure
requirements.txt — Python dependencies (torch, torchvision, pillow, numpy, scikit-learn, flask, requests).

make_demo_data.py — Generates synthetic medicine box images with simulated defects (misspellings, missing holograms, shifted logos, blurs) to verify the pipeline.

make_counterfeits.py — Takes real genuine packaging photos (real_genuine/), splits them by source photo to prevent data leakage, and generates synthesized counterfeit variants.

train.py — Fine-tunes a pretrained ResNet-18 model, saves the best weights to model.pt, and outputs evaluation metrics to metrics.json.

predict.py — Standalone CLI script to run inference on a single image.

ml_service.py — Flask API server running on port 5001 with /health and /predict endpoints.

test_service.py — Integration test script verifying both the local ML service and the end-to-end flow through Member 2's backend.

Setup & Installation
Clone the repository and navigate to the project folder:

Bash


git clone [https://github.com/YOUR-USERNAME/pharmachain-vision.git](https://github.com/YOUR-USERNAME/pharmachain-vision.git)
cd pharmachain-vision
Create and activate a virtual environment:

Bash


python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
Install dependencies:

Bash


pip install -r requirements.txt
Usage
1. Dataset Generation & Training
Option A: Quick Pipeline Test (Demo Data)

Bash


python make_demo_data.py
python train.py --epochs 6
Option B: Real Photos + Synthesized Fakes
Place at least 60 genuine packaging photos into a real_genuine/ folder, then run:

Bash


python make_counterfeits.py --variants 6
python train.py --epochs 8 --unfreeze layer4
2. Single Image Inference (CLI)
Bash


python predict.py path/to/photo.jpg
3. Start the API Service
Bash


python ml_service.py
The service will start on http://127.0.0.1:5001.

4. Run Integration Tests
With ml_service.py running in a separate terminal:

Bash


python test_service.py
API Endpoints
GET /health
Checks if the service is running and returns the loaded class names.

Response:

JSON


{
  "ok": true,
  "classes": ["counterfeit", "genuine"]
}
POST /predict
Classifies an uploaded packaging image.

Request: multipart/form-data with the image file in the image field.

Example (cURL):

Bash


curl -F "image=@box.jpg" [http://127.0.0.1:5001/predict](http://127.0.0.1:5001/predict)
Response (200 OK):

JSON


{
  "ok": true,
  "label": "counterfeit",
  "confidence": 0.9142,
  "scores": {
    "counterfeit": 0.9142,
    "genuine": 0.0858
  }
}
Model Design Notes & Limitations
No Flips or Color Jitter: Standard data augmentations like horizontal flips and color jitter are intentionally excluded during training, as mirrored text and color shifts are primary indicators of counterfeit packaging.

Data Leakage Prevention: make_counterfeits.py splits original photos into train, val, and test sets before generating synthetic variants so derived images never cross splits.

Known Limitations: Models trained on synthesized defects may not generalize to all real-world counterfeits without real regulator samples, and raw softmax confidence scores are uncalibrated probabilities.
