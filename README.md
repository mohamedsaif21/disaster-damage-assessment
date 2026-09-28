# AI Disaster Damage Assessment & Decision Support System

An AI-powered disaster damage assessment system that analyzes pre-disaster and post-disaster satellite/aerial imagery to identify and classify building damage.

The project combines **Deep Learning, Computer Vision, Geospatial AI, Explainable AI, and Full-Stack Web Development** into an end-to-end disaster assessment platform.

---

# Project Overview

Natural disasters such as hurricanes, floods, wildfires, earthquakes, and tsunamis can cause large-scale structural damage.

Traditional damage assessment is often:

- Manual
- Time-consuming
- Difficult to scale
- Dependent on field inspections
- Difficult to perform immediately after a disaster

This project aims to build an AI-assisted system that compares **pre-disaster and post-disaster imagery**, identifies damaged structures, classifies damage severity, and provides the results through a web-based assessment platform.

The system is being developed as both:

1. An **AI/Computer Vision research project**
2. A **full-stack disaster damage assessment application**

---

# Project Information

## Project Type

**Final Year Project**

## Domain

- Artificial Intelligence
- Machine Learning
- Computer Vision
- Geospatial AI (GeoAI)
- Remote Sensing
- Explainable AI (XAI)
- Full-Stack Web Development

## Team

| Name | Role |
|---|---|
| Mohamed Saif B | Team Leader |
| Dinesh D | Member |
| Diya Angeline S P | Member |
| Manjima M | Member |

---

# Core Objective

The primary objective is to develop a system capable of:

1. Accepting pre-disaster and post-disaster images.
2. Validating the image pair.
3. Preprocessing the images for the trained model.
4. Detecting structural damage using a deep-learning segmentation model.
5. Classifying damage into multiple severity classes.
6. Calculating damage statistics.
7. Returning structured assessment results through a FastAPI backend.
8. Storing assessment information using Supabase.
9. Providing the results through a web interface.
10. Supporting future GIS visualization and explainable AI analysis.

---

# Current System Architecture

The current implemented pipeline is:

```text
                    USER
                      │
                      ▼
             Before + After Images
                      │
                      ▼
                FastAPI API
                      │
                      ▼
             Image Validation
                      │
                      ▼
               Preprocessing
                      │
                      ▼
          Change-Aware U-Net
                      │
                      ▼
          Pixel-wise Prediction
                      │
                      ▼
             Damage Statistics
                      │
                      ▼
             Assessment Service
                      │
                      ▼
              Structured JSON
                      │
                      ▼
                  Supabase
                      │
                      ▼
              Frontend Dashboard
                      │
                      ▼
             GIS / Visualization

