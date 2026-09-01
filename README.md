# Final Year Project: AI Disaster Damage Assessment

An AI-powered system for assessing structural damage from natural disasters using satellite/aerial imagery, deep learning segmentation, and interactive geospatial visualization.

---

## Team Members

| Name | Role |
|------|------|
| Mohamed Saif B | Team Leader (TL) |
| Dinesh D | Member |
| Diya Angeline S P | Member |
| Manjima M | Member |

## Domain

**AI/ML** | **Geospatial AI (Geo AI)**

## Project Type

AI + Computer Vision + Full Stack Web Application + GIS Mapping + Explainable AI (XAI) + Research Project

## Programming Languages

Python, TypeScript, SQL, HTML, CSS

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **AI / ML** | PyTorch, U-Net, NumPy, Pandas, Pillow, Matplotlib |
| **Backend** | Python, FastAPI, Uvicorn |
| **Frontend** | Next.js 16 (App Router), React 19, TypeScript, Tailwind CSS v4 |
| **Mapping** | Leaflet, React-Leaflet |
| **Charts** | Recharts |
| **Database** | Supabase (PostgreSQL) |
| **AI Service** | Google Gemini (planned) |

---

## Dataset

**xBD Dataset** — A large-scale benchmark dataset for building damage assessment from satellite imagery. Contains pre-disaster and post-disaster image pairs with pixel-level damage annotations.

**Damage Classes:**

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | Background | Non-building pixels |
| 1 | No Damage | Buildings with no visible damage |
| 2 | Minor Damage | Light structural damage |
| 3 | Major Damage | Severe structural damage |
| 4 | Destroyed | Completely destroyed structures |

---

## Project Structure

```
disaster-damage-assessment/
├── ai/                          # Machine Learning Pipeline
│   ├── unet.py                  # U-Net model architecture
│   ├── dataset.py               # PyTorch dataset (XBDDataset)
│   ├── train_validate.py        # Training + validation loop
│   ├── evaluate_model.py        # Model evaluation (IoU, Accuracy)
│   ├── predict_damage.py        # Single-sample damage prediction
│   ├── analyze_predictions.py   # Generic prediction pipeline
│   ├── validate_predictions.py  # Multi-sample validation to CSV
│   ├── visualize_sample.py      # Sample data visualization
│   ├── visualize_predictions.py # Prediction result visualization
│   ├── analyze_classes.py       # Class distribution analysis
│   ├── checkpoints/             # Saved model weights
│   │   └── best_model.pth
│   └── requirements.txt
│
├── backend/                     # Backend API
│   ├── app/
│   │   └── main.py              # FastAPI application
│   ├── tests/
│   ├── .env / .env.example
│   └── requirements.txt
│
├── frontend/                    # Frontend Web Application
│   ├── src/app/
│   │   ├── page.tsx             # Main page
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── package.json
│   └── tsconfig.json
│
└── outputs/                     # Generated Artifacts
    ├── damage_predictions/      # Damage masks & visualizations
    ├── predictions/             # Sample prediction images
    └── validation_results/      # Validation reports (CSV)
```

---

## Completed Work

### AI / ML Pipeline
- [x] U-Net architecture implementation (encoder-decoder with skip connections)
- [x] Custom PyTorch dataset loader for xBD satellite imagery
- [x] Image preprocessing pipeline (6-channel input: pre + post disaster RGB)
- [x] Training pipeline with weighted CrossEntropyLoss for class imbalance
- [x] Multi-sample validation with CSV report generation
- [x] Single and batch damage prediction scripts
- [x] Confusion matrix, pixel accuracy, and IoU evaluation metrics
- [x] Prediction visualization (4-panel: pre, post, prediction, overlay)
- [x] Class distribution analysis
- [x] Trained model checkpoint (`best_model.pth`)

### Frontend
- [x] Next.js 16 project scaffolding with App Router
- [x] React 19 + TypeScript configuration
- [x] Tailwind CSS v4 styling setup
- [x] Leaflet / React-Leaflet installed for GIS mapping
- [x] Recharts installed for data visualization
- [x] Axios installed for API communication
- [x] Lucide React icons

### Backend
- [x] FastAPI application skeleton
- [x] Health check endpoint (`GET /health`)
- [x] Supabase credentials configured
- [x] Google Gemini API key configured

---

## Still To Do

### AI / ML Pipeline
- [ ] Implement Explainable AI (XAI) — Grad-CAM / SHAP for prediction interpretability
- [ ] Improve model performance (current Mean IoU ~0.19–0.49)
- [ ] Hyperparameter tuning and experiment tracking
- [ ] Model export for inference (ONNX / TorchScript)
- [ ] Batch prediction endpoint for the backend
- [ ] Data augmentation pipeline

### Backend
- [ ] AI inference API endpoints (upload images → get damage assessment)
- [ ] Supabase database integration (users, projects, assessment history)
- [ ] Image upload and storage handling
- [ ] User authentication and authorization
- [ ] Assessment results storage and retrieval
- [ ] Google Gemini integration for automated damage report generation
- [ ] CORS configuration for frontend communication
- [ ] Rate limiting and input validation

### Frontend
- [ ] Interactive GIS map view with Leaflet (damage overlay on map)
- [ ] Image upload component (pre + post disaster pair)
- [ ] Damage assessment result dashboard
- [ ] Damage classification visualization (color-coded masks)
- [ ] Historical assessments and search functionality
- [ ] Responsive design and UI/UX polish
- [ ] Charts and analytics dashboard (Recharts)
- [ ] User authentication pages (login / register)
- [ ] Explainable AI (XAI) visualization panel
- [ ] Real-time prediction status updates

### DevOps & Deployment
- [ ] Docker containerization
- [ ] CI/CD pipeline setup
- [ ] Frontend deployment (Vercel)
- [ ] Backend deployment
- [ ] Environment variable management
- [ ] Production database setup

---

## Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+
- npm or yarn
- CUDA-enabled GPU (recommended for model training)

### AI Pipeline Setup

```bash
cd ai
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
python train_validate.py     # Train the model
python predict_damage.py     # Run predictions
```

### Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env         # Configure environment variables
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`.

---

## Model Architecture

**U-Net** with 6-channel input (pre-disaster RGB + post-disaster RGB):

- **Encoder:** 4 blocks (64 → 128 → 256 → 512 channels) with double convolution + max pooling
- **Bottleneck:** 1024 channels
- **Decoder:** 4 blocks with transposed convolutions + skip connections
- **Output:** 5-class pixel-wise segmentation map

**Training Configuration:**
- Optimizer: Adam (lr = 1e-4)
- Loss: Weighted CrossEntropyLoss (class-weighted for imbalance)
- Input Size: 256 × 256
- Batch Size: 2

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Pixel Accuracy | Overall percentage of correctly classified pixels |
| IoU (Intersection over Union) | Per-class overlap between prediction and ground truth |
| Mean IoU | Average IoU across all classes |
| Confusion Matrix | Detailed class-wise prediction breakdown |

---

## License

This project is part of an academic final year project. For educational use only.

---

*Built with passion for disaster resilience and AI-driven response.*
