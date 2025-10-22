# Professional MPR Viewer with AI

A modern **Medical Image Viewer** built with **PyQt5**, **NumPy**, **Nibabel**, and **TensorFlow** that supports visualization of **NIfTI** and **DICOM** volumes in **Axial, Sagittal, Coronal, and Oblique** planes.  
It also includes an **AI-based orientation classifier** and tools for measurement, ROI extraction, and mask overlay.

---

## 🧠 Features

- **Multi-Planar Reconstruction (MPR):** Axial, Sagittal, Coronal, and Oblique views.  
- **AI Classification:** Automatically classifies NIfTI file orientation (Axial, Coronal, Sagittal).  
- **DICOM to NIfTI conversion** (via `dicom2nifti`).  
- **Brightness and Contrast control per view.**  
- **Zoom and playback controls** (cine mode).  
- **ROI selection and export** as new NIfTI files.  
- **Measurement tool** for pixel distance.  
- **Mask overlay and contour visualization.**  
- **Resizable interface** with a modern dark QSS theme.

---

## ⚙️ Requirements

Install the required dependencies using pip:

```bash
pip install PyQt5 numpy nibabel dicom2nifti pydicom scikit-image scipy tensorflow matplotlib
```

> ⚠️ If you face missing-library errors, the app will display a message with the correct install command.

---

## 📁 File Structure

```
main.py              # Main GUI and logic
nifti_orientation_classifier_full_model.h5   # (Optional) Default AI model file
```

---

## 🚀 Usage

1. **Run the app:**

   ```bash
   python main.py
   ```

2. **Load Data:**
   - **NIfTI File:** Click **“Load NIfTI”** and select a `.nii` or `.nii.gz` file.
   - **DICOM Folder:** Click **“Load DICOM (.dcm)”** and choose a folder containing `.dcm` files.  
     They will be converted automatically to NIfTI.

3. **AI File Classifier:**
   - Click **“Classify NIfTI File”** to detect its anatomical orientation.
   - You can switch models using **“Change AI Model.”**

4. **Manipulate Views:**
   - Adjust tilt angles in **Oblique Plane Controls.**
   - Zoom, pan, or modify brightness and contrast interactively.
   - Enable **Measurement Mode** to measure distances.

5. **ROI Tools:**
   - Select slice ranges and **apply or export ROI** as a new NIfTI file.
   - **Reset** to restore the full volume.

6. **Mask Overlay:**
   - Load mask `.nii` files and visualize outlines or surface boundaries.

---

## 🧩 Notes

- The AI model path is defined in:
  ```python
  self.model_path = r"D:\Programing\Biomedical Projects\Task 2\Files\Ai Model Weight\nifti_orientation_classifier_full_model.h5"
  ```
  Update it to your local model file path if needed.

- Temporary converted or exported files are saved under:
  ```
  D:\Programing\Biomedical Projects\Task 2\Files\Temp
  ```

- If you open large datasets, initial loading may take a few seconds.

---

## 🧰 Tech Stack

- **Python 3.8+**
- **PyQt5** — GUI and visualization
- **NumPy / SciPy** — numerical operations
- **Nibabel** — NIfTI handling
- **dicom2nifti / pydicom** — DICOM conversion
- **scikit-image** — contour extraction
- **TensorFlow / Keras** — AI model inference

---

## 🩺 Screenshot

**

---

## 📜 License

This project is for **educational and research purposes** only.  
Not intended for clinical use.
