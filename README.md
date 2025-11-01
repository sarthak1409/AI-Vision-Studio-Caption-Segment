# 🧠 AI Vision Studio: Caption & Segment

> **Internship Submission – Zidio Development**  
> Developed by **Sarthak Maddi**
> 
## 🎬 Demo Video

[![Watch the video](https://img.youtube.com/vi/DOz9BiiU-VY/maxresdefault.jpg)](https://youtu.be/DOz9BiiU-VY)

A modern **Streamlit-based AI web app** that combines **Image Captioning** and **Instance Segmentation** using state-of-the-art deep learning models.  
This project integrates **BLIP + CLIP** for intelligent caption generation and **Mask R-CNN** for precise object segmentation — all within a sleek, animated UI.

---

## 🚀 Features

- 🧠 **Deep Caption Generation** (BLIP + CLIP reranking)
- 🎯 **Smart Image Segmentation** (Mask R-CNN pretrained on COCO)
- 🖼️ **Interactive Image Comparison** (Original vs Segmented)
- 💾 **Download Segmented Results**
- ⚙️ **Adjustable Parameters**
  - Segmentation confidence threshold
  - Number of caption suggestions
- 🌈 **Aurora Gradient UI** with glassmorphism styling

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Frontend** | Streamlit |
| **Backend / Models** | PyTorch, Hugging Face Transformers |
| **Image Handling** | Pillow (PIL), OpenCV, NumPy |
| **Visualization** | Streamlit Image Comparison |
| **Deployment** | Streamlit Cloud / Localhost |

---

## 📁 Folder Structure

```
internship_zidio/
├── sample_images/               # Sample input images
├── app.py                       # Main Streamlit app (UI + logic)
├── caption_model.py             # BLIP + CLIP captioning module
├── segment_model.py             # Mask R-CNN segmentation module
├── requirements.txt             # Required dependencies
└── README.md                    # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/internship_zidio.git
   cd internship_zidio
   ```

2. **Create Virtual Environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # (on macOS/Linux)
   venv\Scripts\activate       # (on Windows)
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the App**
   ```bash
   streamlit run app.py
   ```

5. **Access the App**
   Open your browser and visit:
   👉 [http://localhost:8501](http://localhost:8501)

---

## 🧠 Model Details

| Component         | Model Used                                       | Source                        |
| ----------------- | ------------------------------------------------ | ----------------------------- |
| Captioning        | **BLIP (Salesforce/blip-image-captioning-base)** | Hugging Face                  |
| Caption Reranking | **CLIP (openai/clip-vit-base-patch32)**          | Hugging Face                  |
| Segmentation      | **Mask R-CNN (ResNet-50 FPN)**                   | TorchVision Pretrained Models |

Each image uploaded passes through BLIP for caption generation and CLIP for semantic reranking.  
For segmentation, Mask R-CNN identifies objects and overlays colored masks with confidence thresholds.

---

## 💡 How It Works

1. **Upload an Image**
2. **Step 1: Deep Captioning**
   * BLIP generates multiple candidate captions.
   * CLIP scores each caption and selects the most relevant one.
3. **Step 2: Instance Segmentation**
   * Mask R-CNN detects and labels objects.
   * Overlays masks and bounding boxes.
4. **Visual Comparison**
   * View side-by-side comparison of original and segmented images.
5. **Download Results**
   * Save your segmented image in `.png` format.

---

## 🎨 UI Highlights

* Dynamic **Aurora Gradient Background**
* Smooth animations & **Glassmorphic containers**
* Clean sidebar with parameter sliders
* Footer credits and branding for Zidio Internship

---

## 📦 Requirements

All dependencies are listed in [`requirements.txt`](requirements.txt):

```
torch
torchvision
transformers
pillow
opencv-python
matplotlib
streamlit
numpy
timm
ftfy
streamlit-image-comparison
```

---

## 📸 Sample Demo

> Upload any image from `sample_images/` and try the caption + segmentation features.  

---

## 🙌 Credits

**Developed by:** Sarthak Maddi  
**Organization:** Zidio Development
**Year:** 2025

---

## 🛡️ License

This project is open-source and available for academic and research use under the MIT License.
