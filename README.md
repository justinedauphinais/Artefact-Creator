# Artefact Creator (Python / OpenCV)

A lightweight toolkit to inject **visual artefacts** into images or videos for testing, augmentation, or demonstration purposes (e.g., for computer vision robustness or media forensics).

Supports the following effects:
- `chromatic_aberration`
- `color_banding`
- `compression` (lossy)
- `dust_particles`
- `gaussian_noise`
- `lens_distortion`
- `motion_blur`
- `salt_pepper_noise`
- `vignetting`

## ✨ Features

- Apply **single** or **stacked** artefacts to images or entire videos.
- Consistent, reproducible results with optional random seed.
- CLI **and** Python API.
- Batch processing of folders.

---

## 🧰 Requirements

- Python 3.9+  
- Packages: `opencv-python`, `numpy`  

Install:
```bash
pip install -r requirements.txt
