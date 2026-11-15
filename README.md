1. Project Overview
Role: Build the "eyes" of PharmaChain—an image classifier that inspects medicine packaging photos to determine whether they are genuine or counterfeit.   
Member4_Counterfeit_Guide.pdf

Integration: Member 2's /verify-image endpoint calls this service, and Member 5's portal uses its photo tab. If a pack is classified as counterfeit with a confidence 
≥
0.7
≥0.7, Member 2's API triggers a blockchain update (set()) to flag the batch as compromised.   
Member4_Counterfeit_Guide.pdf
+ 3

2. Technologies Used
Language & Core Libraries: Python 3.10+, PyTorch, torchvision, Pillow, scikit-learn.   
Member4_Counterfeit_Guide.pdf

Backend & API: Flask, requests.   
Member4_Counterfeit_Guide.pdf
+ 1

Architecture: ResNet-18 (transfer learning using ImageNet pretrained weights).   
Member4_Counterfeit_Guide.pdf
+ 2

Environment: CPU laptop (or Google Colab with a GPU for larger datasets).   
Member4_Counterfeit_Guide.pdf
+ 2

3. System Workflow
Plaintext


[Photo Uploaded] --> [Member 5 Portal] --> [Member 2 API (/verify-image)] 
                                                    │
                                                    ▼
[Blockchain Batch: COMPROMISED] <── (Confidence >= 0.7) ── [Member 4 ML Service (Port 5001)]
Input: A photo of a medicine pack is uploaded via Member 5’s portal.   
Member4_Counterfeit_Guide.pdf

API Call: Member 2's backend sends a POST request with the image to Member 4's service ([http://127.0.0.1:5001/predict](http://127.0.0.1:5001/predict)).   
Member4_Counterfeit_Guide.pdf
+ 2

Classification: The ResNet-18 model evaluates the image and returns a JSON response containing the label (genuine or counterfeit), confidence score, and class probabilities (scores).   
Member4_Counterfeit_Guide.pdf
+ 2

Action: If the label is counterfeit and the confidence is 
≥
0.7
≥0.7, Member 2's API updates the blockchain state for that batch to COMPROMISED.   
Member4_Counterfeit_Guide.pdf
+ 2

4. Main Files
requirements.txt: Lists dependencies (torch, torchvision, pillow, numpy, scikit-learn, flask, requests).   
Member4_Counterfeit_Guide.pdf

make_demo_data.py (Option A): Generates a synthetic dataset of fake and genuine medicine boxes using code to quickly prove the pipeline works.   
Member4_Counterfeit_Guide.pdf
+ 1

make_counterfeits.py (Option B): Takes real genuine packaging photos and programmatically applies realistic defects (blurs, fades, tints, covered patches, shear, JPEG damage, noise) to build a robust training dataset while splitting files by source photo to prevent leakage.   
Member4_Counterfeit_Guide.pdf
+ 2

train.py: Handles ResNet-18 transfer learning, applies custom augmentations (omitting horizontal flips and color jitter to preserve text/color clues), saves the best validation model to model.pt, and outputs metrics.json.   
Member4_Counterfeit_Guide.pdf
+ 2

predict.py: A command-line script that loads model.pt and classifies a single photo.   
Member4_Counterfeit_Guide.pdf
+ 1

ml_service.py: A Flask API running on port 5001 providing health check (/health) and prediction (/predict) endpoints.   
Member4_Counterfeit_Guide.pdf
+ 1

test_service.py: Tests the local ML service and verifies the full end-to-end pathway through Member 2's API and the blockchain.   
Member4_Counterfeit_Guide.pdf
+ 1

5. Setup Instructions
Bash


# 1. Create and enter project directory
mkdir C:\pharmachain\pharmachain-vision
cd C:\pharmachain\pharmachain-vision

# 2. Set up virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Create requirements.txt and install dependencies
# (Paste requirements from documentation into requirements.txt)
pip install -r requirements.txt
(Note: If PowerShell blocks script execution, run Set-ExecutionPolicy -Scope CurrentUser RemoteSigned or use Command Prompt venv\Scripts\activate.bat).   
Member4_Counterfeit_Guide.pdf

6. Demo Commands
Step A: Prove pipeline with demo data & train   
Member4_Counterfeit_Guide.pdf

Bash


python make_demo_data.py
python train.py --epochs 6
Step B: Build real dataset & retrain (Optional)   
Member4_Counterfeit_Guide.pdf

Bash


# Put real genuine photos in real_genuine/ folder first
python make_counterfeits.py --variants 6
python train.py --epochs 8 --unfreeze layer4
Step C: Classify a single photo   
Member4_Counterfeit_Guide.pdf

Bash


python predict.py data\test\counterfeit\counterfeit_000.png
Step D: Start the API service   
Member4_Counterfeit_Guide.pdf

Bash


python ml_service.py
Test the full path (in a separate terminal):   
Member4_Counterfeit_Guide.pdf

Bash
python test_service.py
7. Testing Results
Demo Data Metrics: On synthetic demo data, test accuracy hits roughly 1.000 with clean diagonal confusion matrices ([[30, 0], [0, 30]]).   
Member4_Counterfeit_Guide.pdf

Service Test Output:

Genuine photo: {'confidence': 1.0, 'label': 'genuine', 'ok': True} 
→
→ Batch verdict: GENUINE.   
Member4_Counterfeit_Guide.pdf

Counterfeit photo: {'confidence': 0.9999, 'label': 'counterfeit', 'ok': True} 
→
→ Batch verdict: COMPROMISED.   
Member4_Counterfeit_Guide.pdf

8. Important Notes & Limitations
Data Leakage Prevention: make_counterfeits.py splits source photos into train/val/test splits before generating variants, ensuring a base photo never overlaps across splits.   
Member4_Counterfeit_Guide.pdf
+ 1

Augmentation Restrictions: Training augmentations intentionally exclude horizontal flips and color jitters, as mirrored text and shifted colors are primary indicators of counterfeits.   
Member4_Counterfeit_Guide.pdf
+ 1

Report Limitations to State:

Synthesized counterfeits mean the model learns specific artificial defects rather than all real-world variants.   
Member4_Counterfeit_Guide.pdf
+ 1

Small datasets from limited brands restrict overall generalization.   
Member4_Counterfeit_Guide.pdf
+ 1

Softmax confidence is a design threshold (
0.7
0.7) rather than a calibrated true probability.   
Member4_Counterfeit_Guide.pdf
+ 1

9. Member 4 Contribution Summary
Developed the complete computer vision pipeline (dataset generation/augmentation, ResNet-18 transfer learning model, evaluation scripts, and single-image prediction tool).   
Member4_Counterfeit_Guide.pdf
+ 4

Deployed and maintained the ml_service.py Flask API on port 5001 to integrate seamlessly with Member 2's verification workflow and Member 5's UI portal.   
Member4_Counterfeit_Guide.pdf
+ 1

Validated end-to-end integration with local blockchain states (GENUINE vs. COMPROMISED).   
Member4_Counterfeit_Guide.pdf
+ 2
