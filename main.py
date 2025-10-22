import sys
import os
import math
import traceback
import numpy as np
import pydicom
import dicom2nifti
import shutil
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication

# required libs
try:
    import nibabel as nib
except Exception:
    print("ERROR: nibabel missing. Install: pip install nibabel")
    sys.exit(1)

try:
    from PyQt5.QtWidgets import (
        QApplication, QComboBox, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QFileDialog, QLabel, QSlider, QMessageBox,
        QGridLayout, QGroupBox, QSpinBox, QDoubleSpinBox, QScrollArea,
        QSplitter  # <-- تمت إضافة QSplitter
    )
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
except Exception:
    print("ERROR: PyQt5 missing. Install: pip install PyQt5")
    sys.exit(1)

try:
    from scipy.ndimage import map_coordinates
except Exception:
    print("ERROR: scipy missing. Install: pip install scipy")
    sys.exit(1)

# optional: pydicom (for single .dcm file load)
try:
    import pydicom
    DICOM_AVAILABLE = True
except Exception:
    DICOM_AVAILABLE = False

# skimage for contours
try:
    from skimage import measure
except Exception:
    print("ERROR: scikit-image missing. Install: pip install scikit-image")
    sys.exit(1)

# --- NEW: AI Model Dependencies ---
try:
    import tensorflow as tf
    # This is needed for the new file-based classification
    from tensorflow.keras.preprocessing import image as keras_image_proc
except Exception:
    print("ERROR: tensorflow missing. Install: pip install tensorflow")
    sys.exit(1)
# --- End of AI Dependencies ---


# Modern QSS Stylesheet
QSS_STYLE = """
QWidget {
    background-color: #2b2b2b;
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10pt;
}

QMainWindow {
    background-color: #1b1b1b;
}

QPushButton {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 8px 16px;
    color: #e0e0e0;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #4a4a4a;
    border: 1px solid #6a6a6a;
}

QPushButton:pressed {
    background-color: #2a2a2a;
}

QPushButton:disabled {
    background-color: #2a2a2a;
    color: #666666;
}

QGroupBox {
    border: 2px solid #555555;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: bold;
    color: #61afef;
    background-color: #222222;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
}

QSlider::groove:horizontal {
    border: 1px solid #555555;
    height: 6px;
    background: #3d3d3d;
    margin: 2px 0;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #61afef;
    border: 1px solid #4a90d9;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QSlider::handle:horizontal:hover {
    background: #7ec0ff;
}

QComboBox {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 5px;
    min-width: 100px;
}

QComboBox::drop-down {
    border: none;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #e0e0e0;
    margin-right: 5px;
}

QComboBox QAbstractItemView {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    selection-background-color: #4a90d9;
}

QLabel {
    color: #e0e0e0;
    padding: 2px;
    background-color: transparent;
}

QStatusBar {
    background-color: #2b2b2b;
    color: #98c379;
    border-top: 1px solid #555555;
}

QSpinBox {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px;
    color: #e0e0e0;
}

QSpinBox::up-button, QSpinBox::down-button {
    background-color: #4a4a4a;
    border: 1px solid #555555;
}

QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background-color: #5a5a5a;
}

QCheckBox {
    spacing: 5px;
    color: #e0e0e0;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #555555;
    border-radius: 3px;
    background-color: #3d3d3d;
}

QCheckBox::indicator:checked {
    background-color: #61afef;
    border-color: #4a90d9;
}

QCheckBox::indicator:hover {
    border-color: #7ec0ff;
}

QScrollArea {
    border: none;
    background-color: #2b2b2b;
}

QScrollBar:vertical {
    background-color: #2b2b2b;
    width: 12px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #4a4a4a;
    min-height: 20px;
    border-radius: 6px;
    margin: 2px;
}

QScrollBar::handle:vertical:hover {
    background-color: #5a5a5a;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}

/* --- NEW: Style for the QSplitter handle --- */
QSplitter::handle:horizontal {
    background-color: #4a4a4a;
    width: 4px;
    margin: 1px 0;
}

QSplitter::handle:horizontal:hover {
    background-color: #61afef;
}
/* ------------------------------------------- */
"""


# Color mapping for views
COLORS = {
    'Axial': '#2A7BD8',    # Blue
    'Coronal': '#E53935',  # Red
    'Sagittal': '#F9A825', # Yellow
    'Oblique': '#2ECC71'   # Green
}


# ---------- Utility clamp ----------
def clamp(v, a, b):
    return max(a, min(b, v))


# ---------- Canvas (QLabel-based) ----------
class ImageCanvas(QLabel):
    """Simple image canvas that displays a 2D numpy array (grayscale)
       with two colored reference lines (v and h) and optionally an oblique projection line.
    """
    def __init__(self, view_name, parent_viewer):
        super().__init__()
        self.view_name = view_name
        self.parent_viewer = parent_viewer
        self.image_data = None
        self.v_pos = 0.0
        self.h_pos = 0.0
        self.zoom = 1.0
        self.dragging = None
        self.setMinimumSize(360, 300)
        self.setStyleSheet("border:1px solid #555555; background-color: #1e1e1e;")
        self.setAlignment(Qt.AlignCenter)
        self.active = False

    def set_image(self, arr, v_pos=None, h_pos=None):
        if arr is None or getattr(arr, "size", 0) == 0:
            self.image_data = None
            self.clear()
            return
        img = np.asarray(arr, dtype=np.float32)
        if img.ndim > 2:
            img = np.squeeze(img)
        if img.ndim != 2:
            img = np.zeros((10, 10), dtype=np.float32)
        self.image_data = img
        h, w = img.shape
        if v_pos is None:
            v_pos = w // 2
        if h_pos is None:
            h_pos = h // 2
        self.v_pos = clamp(float(v_pos), 0, w - 1)
        self.h_pos = clamp(float(h_pos), 0, h - 1)
        self.update_display()

    def update_display(self, oblique_direction_2d=None, boundary_contours=None, measurement_points=None):
        if self.image_data is None:
            self.clear()
            return
        try:
            img = self.image_data.copy()
            
            # Apply brightness and contrast from parent viewer
            view_idx = {'Axial': 0, 'Sagittal': 1, 'Coronal': 2, 'Oblique': 3}.get(self.view_name, 0)
            brightness = self.parent_viewer.brightness_values[view_idx]
            contrast = self.parent_viewer.contrast_values[view_idx]
            
            # Normalize the data to 0-1 range first
            if img.max() > img.min():
                normalized_data = (img - np.min(img)) / (np.max(img) - np.min(img))
            else:
                normalized_data = np.zeros_like(img, dtype=np.float32)
            
            # Apply contrast first (centered at 0.5)
            contrasted = np.clip((normalized_data - 0.5) * contrast + 0.5, 0, 1)
            
            # Then apply brightness (normalize brightness to range [-1, 1])
            brightness_normalized = brightness / 100.0
            adjusted = np.clip(contrasted + brightness_normalized, 0, 1)
            
            # Convert to 0-255 range for display
            arr8 = (adjusted * 255).astype(np.uint8)

            h, w = arr8.shape
            bytes_per_line = w
            qimg = QImage(arr8.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            
            # Apply colormap if not grayscale
            if self.parent_viewer.current_colormap != 'gray':
                # Convert grayscale to RGB using colormap
                from matplotlib import cm
                cmap = cm.get_cmap(self.parent_viewer.current_colormap)
                rgba = cmap(arr8 / 255.0)
                rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                h, w = rgb.shape[:2]
                bytes_per_line = w * 3
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            sw = int(w * self.zoom); sh = int(h * self.zoom)
            if sw <= 0 or sh <= 0:
                return
            qimg = qimg.scaled(sw, sh, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            pix = QPixmap.fromImage(qimg)
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)

            if self.view_name == 'Axial':
                color_v = QColor(COLORS['Sagittal'])
                color_h = QColor(COLORS['Coronal'])
            elif self.view_name == 'Sagittal':
                color_v = QColor(COLORS['Coronal'])
                color_h = QColor(COLORS['Axial'])
            elif self.view_name == 'Coronal':
                color_v = QColor(COLORS['Sagittal'])
                color_h = QColor(COLORS['Axial'])
            else:
                color_v = color_h = QColor('#61afef')

            pen_v = QPen(color_v)
            pen_v.setWidth(2)
            painter.setPen(pen_v)
            cx = int(clamp(self.v_pos * self.zoom, 0, pix.width() - 1))
            painter.drawLine(cx, 0, cx, pix.height())

            pen_h = QPen(color_h)
            pen_h.setWidth(2)
            painter.setPen(pen_h)
            cy = int(clamp(self.h_pos * self.zoom, 0, pix.height() - 1))
            painter.drawLine(0, cy, pix.width(), cy)

            square_w, square_h = 18, 12
            margin = 6
            painter.fillRect(pix.width() - square_w - margin, margin, square_w, square_h, QColor(COLORS.get(self.view_name, '#FFFFFF')))

            if oblique_direction_2d is not None:
                dx, dy = oblique_direction_2d
                mag = math.hypot(dx, dy)
                if mag > 1e-6:
                    ndx, ndy = dx / mag, dy / mag
                    length = int(min(pix.width(), pix.height()) * 0.6)
                    cxp = pix.width() // 2
                    cyp = pix.height() // 2
                    x1 = int(cxp - ndx * length / 2)
                    y1 = int(cyp - ndy * length / 2)
                    x2 = int(cxp + ndx * length / 2)
                    y2 = int(cyp + ndy * length / 2)
                    pen_o = QPen(QColor(COLORS['Oblique']))
                    pen_o.setWidth(2)
                    painter.setPen(pen_o)
                    painter.drawLine(x1, y1, x2, y2)

            if boundary_contours:
                color = getattr(self.parent_viewer, "mask_color", QColor('#00FF00'))
                pen_b = QPen(color)
                pen_b.setWidth(2)
                painter.setPen(pen_b)
                img_w, img_h = w, h
                pix_w, pix_h = pix.width(), pix.height()
                offset_x = (pix_w - img_w * self.zoom) / 2.0
                offset_y = (pix_h - img_h * self.zoom) / 2.0
                for cnt in boundary_contours:
                    if cnt.size == 0:
                        continue
                    prev = None
                    for (r, c) in cnt:
                        x = int(clamp(c * self.zoom + 0.5 + offset_x, 0, pix_w - 1))
                        y = int(clamp(r * self.zoom + 0.5 + offset_y, 0, pix_h - 1))
                        if prev is not None:
                            painter.drawLine(prev[0], prev[1], x, y)
                        prev = (x, y)
                    if len(cnt) >= 2:
                        r0, c0 = cnt[0]
                        x0 = int(clamp(c0 * self.zoom + 0.5 + offset_x, 0, pix_w - 1))
                        y0 = int(clamp(r0 * self.zoom + 0.5 + offset_y, 0, pix_h - 1))
                        painter.drawLine(prev[0], prev[1], x0, y0)

            # Draw measurement points and lines
            if measurement_points and len(measurement_points) > 0:
                pen_m = QPen(QColor('#FF00FF'))  # Magenta for measurements
                pen_m.setWidth(3)
                painter.setPen(pen_m)
                
                for i, (px, py) in enumerate(measurement_points):
                    # Convert image coords to pixmap coords
                    x = int(clamp(px * self.zoom + 0.5, 0, pix.width() - 1))
                    y = int(clamp(py * self.zoom + 0.5, 0, pix.height() - 1))
                    
                    # Draw point
                    painter.drawEllipse(x - 5, y - 5, 10, 10)
                    
                    # Draw line between points
                    if i > 0:
                        prev_px, prev_py = measurement_points[i-1]
                        prev_x = int(clamp(prev_px * self.zoom + 0.5, 0, pix.width() - 1))
                        prev_y = int(clamp(prev_py * self.zoom + 0.5, 0, pix.height() - 1))
                        painter.drawLine(prev_x, prev_y, x, y)

            painter.end()
            self.setPixmap(pix)
        except Exception as e:
            print(f"Error in update_display ({self.view_name}): {e}")
            traceback.print_exc()

    def mousePressEvent(self, event):
        if self.image_data is None or event.button() != Qt.LeftButton:
            return
        
        # Check if in measurement mode
        if self.parent_viewer.measurement_mode:
            pix = self.pixmap()
            if pix is None:
                return
            lw, lh = self.width(), self.height()
            pw, ph = pix.width(), pix.height()
            offx = (lw - pw) / 2.0
            offy = (lh - ph) / 2.0
            mx = (event.x() - offx) / self.zoom
            my = (event.y() - offy) / self.zoom
            mx = clamp(mx, 0, self.image_data.shape[1] - 1)
            my = clamp(my, 0, self.image_data.shape[0] - 1)
            
            # Add measurement point
            self.parent_viewer.add_measurement_point(self.view_name, mx, my)
            return
        
        p = self.parent_viewer
        pix = self.pixmap()
        if pix is None:
            return
        lw, lh = self.width(), self.height()
        pw, ph = pix.width(), pix.height()
        offx = (lw - pw) / 2.0
        offy = (lh - ph) / 2.0
        mx = (event.x() - offx) / self.zoom
        my = (event.y() - offy) / self.zoom
        mx = clamp(mx, 0, self.image_data.shape[1] - 1)
        my = clamp(my, 0, self.image_data.shape[0] - 1)

        tol_px = 8
        v_widget_x = int(self.v_pos * self.zoom + offx)
        h_widget_y = int(self.h_pos * self.zoom + offy)
        dx_v = abs(event.x() - v_widget_x)
        dy_h = abs(event.y() - h_widget_y)

        ob_dir = p.get_oblique_projection_on_view(self.view_name)
        is_near_oblique = False
        if ob_dir is not None:
            obdx, obdy = ob_dir
            mag = math.hypot(obdx, obdy)
            if mag > 1e-6:
                ndx, ndy = obdx / mag, obdy / mag
                cx = int(pw // 2 + offx)
                cy = int(ph // 2 + offy)
                vx = event.x() - cx; vy = event.y() - cy
                perp = abs(-ndy * vx + ndx * vy)
                if perp <= tol_px:
                    is_near_oblique = True

        if is_near_oblique:
            self.dragging = 'oblique'
        else:
            if dx_v <= tol_px and dx_v <= dy_h:
                self.dragging = 'v'
            elif dy_h <= tol_px and dy_h < dx_v:
                self.dragging = 'h'
            else:
                if abs(mx - self.v_pos) < abs(my - self.h_pos):
                    self.dragging = 'v'
                else:
                    self.dragging = 'h'

        p.handle_drag_from_view(self.view_name, self.dragging, mx, my)

    def mouseMoveEvent(self, event):
        if self.image_data is None or self.dragging is None:
            return
        pix = self.pixmap()
        if pix is None:
            return
        lw, lh = self.width(), self.height()
        pw, ph = pix.width(), pix.height()
        offx = (lw - pw) / 2.0
        offy = (lh - ph) / 2.0
        mx = (event.x() - offx) / self.zoom
        my = (event.y() - offy) / self.zoom
        mx = clamp(mx, 0, self.image_data.shape[1] - 1)
        my = clamp(my, 0, self.image_data.shape[0] - 1)
        p = self.parent_viewer
        p.handle_drag_from_view(self.view_name, self.dragging, mx, my)

    def mouseReleaseEvent(self, event):
        self.dragging = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom = min(8.0, self.zoom * 1.15)
        else:
            self.zoom = max(0.2, self.zoom / 1.15)
        self.update_display()


# ---------- Main application ----------
class MPRViewer(QMainWindow):

    # --- AI Model Settings ---
    # These are now instance variables set in __init__
    
    IMG_SIZE = 128
    CLASSES = ['axial', 'coronal', 'sagittal']
    # -------------------------
    def load_mask_outline(self):
        """Loads a NIfTI mask file and overlays its contours."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Mask NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if not path:
            return
        try:
            nii = nib.load(path)
            mask_data = np.asarray(nii.dataobj)
            if mask_data.ndim == 4:
                mask_data = mask_data[..., 0]
            if mask_data.ndim != 3:
                QMessageBox.critical(self, "Error", f"Invalid mask shape: {mask_data.shape}")
                return
            self.mask_volume = (mask_data > 0).astype(np.uint8)
            self.statusBar().showMessage(f"Loaded mask: {os.path.basename(path)}")
            self.update_all_views()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load mask:\n{e}")

    def change_mask_color(self):
        color = QColorDialog.getColor(self.mask_color, self, "Select Mask Outline Color")
        if color.isValid():
            self.mask_color = color
            self.update_all_views()

    def __init__(self):


        super().__init__()
        self.mask_volume = None
        self.mask_color = QColor(255, 0, 0)  # Red by default
        self.setWindowTitle("Professional MPR Viewer with AI")
        self.setGeometry(100, 100, 1600, 900)

        # data
        self.volume = None
        self.original_volume = None  # Keep original for reset
        self.combined_mask = None
        self.shape = None
        self.temp_roi_file = None  # Track temporary ROI file

        # crosshair in voxel coords (x,y,z)
        self.cross = [0, 0, 0]
        self.slice_idx = [0, 0, 0, 0]  # Added 4th for oblique

        # oblique angles
        self.oblique_tilt_x = 0.0
        self.oblique_tilt_y = 30.0

        # playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)
        self.play_states = {'Axial': False, 'Sagittal': False, 'Coronal': False, 'Oblique': False}
        self.global_play = False  # Global play state

        # Measurement mode
        self.measurement_mode = False
        self.measurement_points = {'Axial': [], 'Sagittal': [], 'Coronal': [], 'Oblique': []}
        self.pixel_spacing = [1.0, 1.0, 1.0]  # default spacing in mm
        
        # Brightness and contrast per view
        self.brightness_values = [0, 0, 0, 0]  # Axial, Sagittal, Coronal, Oblique
        self.contrast_values = [1.0, 1.0, 1.0, 1.0]
        
        # Colormap
        self.current_colormap = 'gray'

        # --- AI Model ---
        # 🔴 !! IMPORTANT: UPDATE THIS PATH !! 🔴
        # This is now the *default* model path
        self.model_path = r"D:\Programing\Biomedical Projects\Task 2\Files\Ai Model Weight\nifti_orientation_classifier_full_model.h5"
        # 🔴 !! IMPORTANT: UPDATE THIS PATH !! 🔴
        self.temp_save_path = r"D:\Programing\Biomedical Projects\Task 2\Files\Temp"

        self.ai_model = None

        self.init_ui()
        self.load_ai_model() # Load the *default* AI model on startup

    def load_mask_outline(self):
        """Loads a NIfTI mask file and overlays its contours."""
        path, _ = QFileDialog.getOpenFileName(self, "Select Mask NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if not path:
            return
        try:
            nii = nib.load(path)
            mask_data = np.asarray(nii.dataobj)
            if mask_data.ndim == 4:
                mask_data = mask_data[..., 0]
            if mask_data.ndim != 3:
                QMessageBox.critical(self, "Error", f"Invalid mask shape: {mask_data.shape}")
                return
            self.mask_volume = (mask_data > 0).astype(np.uint8)
            self.statusBar().showMessage(f"Loaded mask: {os.path.basename(path)}")
            self.update_all_views()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load mask:\n{e}")

    def change_mask_color(self):
        color = QColorDialog.getColor(self.mask_color, self, "Select Mask Outline Color")
        if color.isValid():
            self.mask_color = color
            # أعد تحديث كل العروض لإجبار الـ QPainter على إعادة الرسم باللون الجديد
            self.update_all_views()
            # تحديث يدوي إضافي لضمان إعادة الرسم الفوري
            for canvas in [self.canvas_axial, self.canvas_sagittal, self.canvas_coronal, self.canvas_oblique]:
                if canvas:
                    canvas.update()

    def init_ui(self):
        # Apply modern styling
        self.setStyleSheet(QSS_STYLE)
        
        # --- 🔴 CHANGED: Use QSplitter for resizable sidebar ---
        # central = QWidget() # OLD
        # self.setCentralWidget(central) # OLD
        
        # Main horizontal layout: sidebar on left, viewports on right
        # main_layout = QHBoxLayout(central) # OLD
        
        # --- NEW: Use a QSplitter for resizable panels ---
        main_splitter = QSplitter(Qt.Horizontal, self)
        self.setCentralWidget(main_splitter)
        # --------------------------------------------------
        
        # ===== LEFT SIDEBAR =====
        # Create scrollable left control panel
        self.control_scroll = QScrollArea()
        self.control_scroll.setWidgetResizable(True)
        self.control_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.control_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # --- REMOVED fixed widths ---
        # self.control_scroll.setMaximumWidth(340) # OLD
        
        # --- We can set a *minimum* reasonable width ---
        self.control_scroll.setMinimumWidth(300) 
        
        self.control_panel = QWidget()
        self.control_layout = QVBoxLayout()
        self.control_panel.setLayout(self.control_layout)
        
        # Set the control panel as the scroll area's widget
        self.control_scroll.setWidget(self.control_panel)

        # Load buttons
        self.btn_load_nii = QPushButton("Load NIfTI")
        self.btn_load_nii.clicked.connect(self.load_nifti)
        self.control_layout.addWidget(self.btn_load_nii)

        self.btn_load_dcm = QPushButton("Load DICOM (.dcm)")
        self.btn_load_dcm.clicked.connect(self.load_dicom_file)
        if not DICOM_AVAILABLE:
            self.btn_load_dcm.setEnabled(False)
            self.btn_load_dcm.setToolTip("Install pydicom to enable DICOM support")
        self.control_layout.addWidget(self.btn_load_dcm)

        # --- UPDATED: AI File Classifier ---
        ai_group = QGroupBox("AI File Classifier")
        ai_layout = QVBoxLayout()
        
        self.btn_classify_file = QPushButton("Classify NIfTI File")
        self.btn_classify_file.clicked.connect(self.classify_nifti_file)
        ai_layout.addWidget(self.btn_classify_file)

        # --- NEW: Button to change model ---
        self.btn_load_model = QPushButton("Change AI Model")
        self.btn_load_model.clicked.connect(self.select_ai_model)
        ai_layout.addWidget(self.btn_load_model)
        # -----------------------------------

        ai_group.setLayout(ai_layout)
        self.control_layout.addWidget(ai_group)
        # -------------------------------

        # Global Cine Controls
        cine_group = QGroupBox("Global Cine Controls")
        cine_layout = QVBoxLayout()
        
        cine_buttons = QHBoxLayout()
        self.btn_global_play = QPushButton("Play All")
        self.btn_global_play.setCheckable(True)
        self.btn_global_play.clicked.connect(self.toggle_global_play)
        self.btn_global_step_back = QPushButton("◀◀ Step Back")
        self.btn_global_step_back.clicked.connect(lambda: self.global_step(-1))
        self.btn_global_step_fwd = QPushButton("Step Forward ▶▶")
        self.btn_global_step_fwd.clicked.connect(lambda: self.global_step(1))
        
        cine_buttons.addWidget(self.btn_global_step_back)
        cine_buttons.addWidget(self.btn_global_play)
        cine_buttons.addWidget(self.btn_global_step_fwd)
        cine_layout.addLayout(cine_buttons)
        
        cine_group.setLayout(cine_layout)
        self.control_layout.addWidget(cine_group)

        # Playback speed controls
        speed_group = QGroupBox("Playback Speed")
        speed_layout = QVBoxLayout()
        speed_layout.addWidget(QLabel("Speed (FPS):"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(30)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.update_timer_interval)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("10 FPS")
        speed_layout.addWidget(self.speed_label)
        speed_group.setLayout(speed_layout)
        self.control_layout.addWidget(speed_group)

        # Oblique controls
        oblique_group = QGroupBox("Oblique Plane Controls")
        oblique_layout = QVBoxLayout()
        
        oblique_layout.addWidget(QLabel("Tilt X (Pitch):"))
        self.tilt_x_slider = QSlider(Qt.Horizontal)
        self.tilt_x_slider.setMinimum(-89)
        self.tilt_x_slider.setMaximum(89)
        self.tilt_x_slider.setValue(int(self.oblique_tilt_x))
        self.tilt_x_slider.valueChanged.connect(self.tilt_x_changed)
        oblique_layout.addWidget(self.tilt_x_slider)
        self.tilt_x_label = QLabel("0°")
        oblique_layout.addWidget(self.tilt_x_label)

        oblique_layout.addWidget(QLabel("Tilt Y (Yaw):"))
        self.tilt_y_slider = QSlider(Qt.Horizontal)
        self.tilt_y_slider.setMinimum(-89)
        self.tilt_y_slider.setMaximum(89)
        self.tilt_y_slider.setValue(int(self.oblique_tilt_y))
        self.tilt_y_slider.valueChanged.connect(self.tilt_y_changed)
        oblique_layout.addWidget(self.tilt_y_slider)
        self.tilt_y_label = QLabel("30°")
        oblique_layout.addWidget(self.tilt_y_label)
        
        oblique_group.setLayout(oblique_layout)
        self.control_layout.addWidget(oblique_group)

        # Zoom controls
        zoom_group = QGroupBox("Zoom Controls")
        zoom_layout = QVBoxLayout()
        zoom_layout.addWidget(QLabel("Zoom Level:"))
        
        zoom_buttons = QHBoxLayout()
        btn_zoom_in = QPushButton("Zoom In (+)")
        btn_zoom_in.clicked.connect(self.zoom_in_all)
        btn_zoom_out = QPushButton("Zoom Out (-)")
        btn_zoom_out.clicked.connect(self.zoom_out_all)
        btn_zoom_reset = QPushButton("Reset Zoom")
        btn_zoom_reset.clicked.connect(self.reset_zoom_all)
        zoom_buttons.addWidget(btn_zoom_in)
        zoom_buttons.addWidget(btn_zoom_out)
        zoom_layout.addLayout(zoom_buttons)
        zoom_layout.addWidget(btn_zoom_reset)
        
        zoom_group.setLayout(zoom_layout)
        self.control_layout.addWidget(zoom_group)

        # Brightness and Contrast controls
        bc_group = QGroupBox("Brightness & Contrast")
        bc_layout = QVBoxLayout()
        
        bc_layout.addWidget(QLabel("Brightness:"))
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum(-100)
        self.brightness_slider.setMaximum(100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        bc_layout.addWidget(self.brightness_slider)
        self.brightness_label = QLabel("0")
        bc_layout.addWidget(self.brightness_label)
        
        bc_layout.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setMinimum(10)
        self.contrast_slider.setMaximum(300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_contrast)
        bc_layout.addWidget(self.contrast_slider)
        self.contrast_label = QLabel("1.0")
        bc_layout.addWidget(self.contrast_label)
        
        btn_reset_bc = QPushButton("Reset B/C")
        btn_reset_bc.clicked.connect(self.reset_brightness_contrast)
        bc_layout.addWidget(btn_reset_bc)
        
        bc_group.setLayout(bc_layout)
        self.control_layout.addWidget(bc_group)

        # Measurement tools
        measure_group = QGroupBox("Measurement Tools")
        measure_layout = QVBoxLayout()
        
        self.btn_measure = QPushButton("Enable Measurement")
        self.btn_measure.setCheckable(True)
        self.btn_measure.clicked.connect(self.toggle_measurement_mode)
        measure_layout.addWidget(self.btn_measure)
        
        self.measure_label = QLabel("Click two points to measure")
        self.measure_label.setWordWrap(True)
        measure_layout.addWidget(self.measure_label)
        
        btn_clear_measure = QPushButton("Clear Measurements")
        btn_clear_measure.clicked.connect(self.clear_measurements)
        measure_layout.addWidget(btn_clear_measure)
        
        measure_group.setLayout(measure_layout)
        self.control_layout.addWidget(measure_group)

        # Colormap selection
        colormap_group = QGroupBox("Colormap")
        colormap_layout = QVBoxLayout()
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['gray', 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'jet', 'hot', 'cool', 'bone'])
        self.colormap_combo.currentTextChanged.connect(self.update_colormap)
        colormap_layout.addWidget(self.colormap_combo)
        colormap_group.setLayout(colormap_layout)
        self.control_layout.addWidget(colormap_group)

        # ROI export
        roi_group = QGroupBox("ROI Export")
        roi_layout = QVBoxLayout()
        roi_layout.addWidget(QLabel("Axial Slice Range"))
        roi_layout.addWidget(QLabel("Start slice:"))
        self.roi_start = QSpinBox()
        self.roi_start.setMinimum(0)
        roi_layout.addWidget(self.roi_start)
        roi_layout.addWidget(QLabel("End slice:"))
        self.roi_end = QSpinBox()
        self.roi_end.setMinimum(0)
        roi_layout.addWidget(self.roi_end)
        
        roi_buttons = QHBoxLayout()
        self.btn_apply_roi = QPushButton("Apply ROI")
        self.btn_apply_roi.clicked.connect(self.apply_roi)
        self.btn_export_roi = QPushButton("Export ROI")
        self.btn_export_roi.clicked.connect(self.export_roi)
        roi_buttons.addWidget(self.btn_apply_roi)
        roi_buttons.addWidget(self.btn_export_roi)
        roi_layout.addLayout(roi_buttons)
        
        self.btn_reset_roi = QPushButton("Reset to Original")
        self.btn_reset_roi.clicked.connect(self.reset_to_original)
        self.btn_reset_roi.setEnabled(False)
        roi_layout.addWidget(self.btn_reset_roi)
        
        roi_group.setLayout(roi_layout)
        self.control_layout.addWidget(roi_group)

        # === Mask / Outline Overlay ===
        mask_group = QGroupBox("Mask Outline Overlay")
        mask_layout = QVBoxLayout()

        self.btn_load_mask_outline = QPushButton("Load Mask NIfTI")
        self.btn_load_mask_outline.clicked.connect(self.load_mask_outline)
        mask_layout.addWidget(self.btn_load_mask_outline)

        self.btn_change_mask_color = QPushButton("Change Outline Color")
        self.btn_change_mask_color.clicked.connect(self.change_mask_color)
        mask_layout.addWidget(self.btn_change_mask_color)

        mask_group.setLayout(mask_layout)
        self.control_layout.addWidget(mask_group)

        # === Mask / Outline Loader ===
        mask_group = QGroupBox("Mask Outline Overlay")
        mask_layout = QVBoxLayout()

        self.btn_load_mask_outline = QPushButton("Load Mask NIfTI")
        self.btn_load_mask_outline.clicked.connect(self.load_mask_outline)
        mask_layout.addWidget(self.btn_load_mask_outline)

        self.btn_change_mask_color = QPushButton("Change Outline Color")
        self.btn_change_mask_color.clicked.connect(self.change_mask_color)
        mask_layout.addWidget(self.btn_change_mask_color)

        mask_group.setLayout(mask_layout)
        self.control_layout.addWidget(mask_group)

        # Add stretch to push controls to top
        self.control_layout.addStretch()
        
        # Add scrollable control panel to main splitter
        # main_layout.addWidget(self.control_scroll) # OLD
        main_splitter.addWidget(self.control_scroll) # NEW

        # ===== RIGHT VIEWPORT PANEL =====
        viewport_panel = QWidget()
        viewport_layout = QVBoxLayout(viewport_panel)

        # Viewport grid
        grid = QGridLayout()
        self.canvas_axial = ImageCanvas('Axial', self)
        self.canvas_sagittal = ImageCanvas('Sagittal', self)
        self.canvas_coronal = ImageCanvas('Coronal', self)
        self.canvas_oblique = ImageCanvas('Oblique', self)

        grid.addWidget(self.wrap_with_controls(self.canvas_axial, 'Axial'), 0, 0)
        grid.addWidget(self.wrap_with_controls(self.canvas_sagittal, 'Sagittal'), 0, 1)
        grid.addWidget(self.wrap_with_controls(self.canvas_coronal, 'Coronal'), 1, 0)
        grid.addWidget(self.wrap_with_controls(self.canvas_oblique, 'Oblique'), 1, 1)

        viewport_layout.addLayout(grid)
        
        # Add viewport panel to main splitter
        # main_layout.addWidget(viewport_panel) # OLD
        main_splitter.addWidget(viewport_panel) # NEW

        # --- NEW: Set initial sizes for the splitter ---
        # Give the controls a reasonable starting size and the viewports the rest
        main_splitter.setSizes([340, 1260]) # (340 for sidebar, rest for viewports, ~1600 total)
        main_splitter.setStretchFactor(1, 1) # Make the viewport panel (index 1) expand with window
        # --------------------------------------------------

        self.statusBar().showMessage("Ready")

    def wrap_with_controls(self, canvas, title):
        cont = QWidget()
        cont.setStyleSheet("background-color: #2b2b2b;")
        vl = QVBoxLayout(cont)
        vl.setContentsMargins(0, 0, 0, 0)
        
        # header
        header = QWidget()
        header.setStyleSheet("background-color: #2b2b2b;")
        hh = QHBoxLayout(header)
        hh.setContentsMargins(5, 5, 5, 5)
        sq = QLabel()
        sq.setFixedSize(18, 12)
        sq.setStyleSheet(f"background-color: {COLORS.get(title,'#fff')}; border:1px solid #555;")
        lab = QLabel(f"<b>{title} View</b>")
        lab.setStyleSheet("color:#61afef;")
        hh.addWidget(sq)
        hh.addWidget(lab)
        
        # --- Add Predict Button for individual slices ---
        if title in ['Axial', 'Sagittal', 'Coronal']:
            btn_predict = QPushButton("Predict Slice")
            # Make the button smaller to fit the header
            btn_predict.setStyleSheet("padding: 4px 8px; font-weight: normal; margin-left: 10px;")
            # Use lambda to pass title to the prediction function
            btn_predict.clicked.connect(lambda checked, t=title: self.run_prediction(t))
            hh.addWidget(btn_predict)
        # -------------------------------

        hh.addStretch()
        
        if title == 'Oblique':
            lab_mode = QLabel("Mode:")
            self.oblique_mode = QComboBox()
            self.oblique_mode.addItems(["Oblique View", "Surface Boundary"])
            self.oblique_mode.currentTextChanged.connect(self.on_oblique_mode_changed)
            hh.addWidget(lab_mode)
            hh.addWidget(self.oblique_mode)
            hh.addStretch()
        vl.addWidget(header)
        vl.addWidget(canvas)

        # Add controls for all views including Oblique
        ctrl = QWidget()
        ctrl.setStyleSheet("background-color: #2b2b2b;")
        ch = QHBoxLayout(ctrl)
        ch.setContentsMargins(4, 4, 4, 4)
        back = QPushButton("◀")
        play = QPushButton("Play")
        play.setCheckable(True)
        fwd = QPushButton("▶")
        slider = QSlider(Qt.Horizontal)
        ch.addWidget(back)
        ch.addWidget(play)
        ch.addWidget(fwd)
        ch.addWidget(QLabel("Slice:"))
        ch.addWidget(slider)
        vl.addWidget(ctrl)

        if title == 'Axial':
            back.clicked.connect(lambda: self.step_view('Axial', -1))
            fwd.clicked.connect(lambda: self.step_view('Axial', 1))
            play.clicked.connect(lambda checked: self.toggle_play('Axial', checked, play))
            slider.valueChanged.connect(lambda v: self.slider_moved('Axial', v))
            self.axial_slider = slider
            self.axial_play_btn = play
        elif title == 'Sagittal':
            back.clicked.connect(lambda: self.step_view('Sagittal', -1))
            fwd.clicked.connect(lambda: self.step_view('Sagittal', 1))
            play.clicked.connect(lambda checked: self.toggle_play('Sagittal', checked, play))
            slider.valueChanged.connect(lambda v: self.slider_moved('Sagittal', v))
            self.sag_slider = slider
            self.sag_play_btn = play
        elif title == 'Coronal':
            back.clicked.connect(lambda: self.step_view('Coronal', -1))
            fwd.clicked.connect(lambda: self.step_view('Coronal', 1))
            play.clicked.connect(lambda checked: self.toggle_play('Coronal', checked, play))
            slider.valueChanged.connect(lambda v: self.slider_moved('Coronal', v))
            self.cor_slider = slider
            self.cor_play_btn = play
        elif title == 'Oblique':
            back.clicked.connect(lambda: self.step_view('Oblique', -1))
            fwd.clicked.connect(lambda: self.step_view('Oblique', 1))
            play.clicked.connect(lambda checked: self.toggle_play('Oblique', checked, play))
            slider.valueChanged.connect(lambda v: self.slider_moved('Oblique', v))
            self.oblique_slider = slider
            self.oblique_play_btn = play

        return cont
    
    def load_nifti_direct(self, path):
        """
        Loads a NIfTI file from a given path — used after DICOM conversion.
        """
        import nibabel as nib
        import numpy as np
        from PyQt5.QtWidgets import QMessageBox

        try:
            nii = nib.load(path)
            data = np.asarray(nii.dataobj)

            # معالجة الأبعاد (3D فقط)
            if data.ndim == 4:
                data = data[..., 0]
            if data.ndim == 2:
                data = data[..., np.newaxis]
            if data.ndim != 3:
                QMessageBox.critical(self, "Error", f"Unsupported NIfTI shape: {data.shape}")
                return

            # تحميل البيانات في خصائص الكلاس
            self.volume = data.astype(np.float32)
            self.original_volume = self.volume.copy()
            nx, ny, nz = self.volume.shape
            self.shape = (nx, ny, nz)
            self.cross = [nx // 2, ny // 2, nz // 2]
            self.slice_idx = [nz // 2, nx // 2, ny // 2, 0]

            # تحديث الواجهة (نفس فكرة load_nifti)
            self.roi_start.setMaximum(nz - 1)
            self.roi_end.setMaximum(nz - 1)
            self.roi_end.setValue(nz - 1)
            self.update_all_views()

            self.statusBar().showMessage(f"✅ Loaded NIfTI from DICOM: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load converted NIfTI:\n{e}")
            print(e)

    def load_nifti(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)")
        if not path:
            return
        self.statusBar().showMessage(f"Loading {os.path.basename(path)}...")
        QApplication.processEvents()
        try:
            nii = nib.load(path)
            data = np.asarray(nii.dataobj)
            if data.ndim == 4:
                data = data[..., 0]
                QMessageBox.information(self, "Info", "4D volume: using first timepoint")
            if data.ndim == 2:
                data = data[..., np.newaxis]
            if data.ndim != 3:
                QMessageBox.critical(self, "Error", f"Unsupported shape: {data.shape}")
                return
            self.volume = data.astype(np.float32)
            self.original_volume = self.volume.copy()  # Keep original copy
            nx, ny, nz = self.volume.shape
            self.shape = (nx, ny, nz)
            self.cross = [nx//2, ny//2, nz//2]
            self.slice_idx = [nz//2, nx//2, ny//2, 0]  # Added 4th element for oblique
            
            if self.axial_slider:
                self.axial_slider.blockSignals(True)
                self.axial_slider.setMinimum(0)
                self.axial_slider.setMaximum(nz-1)
                self.axial_slider.setValue(self.slice_idx[0])
                self.axial_slider.blockSignals(False)
            if self.sag_slider:
                self.sag_slider.blockSignals(True)
                self.sag_slider.setMinimum(0)
                self.sag_slider.setMaximum(nx-1)
                self.sag_slider.setValue(self.slice_idx[1])
                self.sag_slider.blockSignals(False)
            if self.cor_slider:
                self.cor_slider.blockSignals(True)
                self.cor_slider.setMinimum(0)
                self.cor_slider.setMaximum(ny-1)
                self.cor_slider.setValue(self.slice_idx[2])
                self.cor_slider.blockSignals(False)
            if self.oblique_slider:
                self.oblique_slider.blockSignals(True)
                self.oblique_slider.setMinimum(0)
                self.oblique_slider.setMaximum(100)  # Percentage-based for oblique depth
                self.oblique_slider.setValue(50)
                self.oblique_slider.blockSignals(False)
            
            self.roi_start.setMaximum(nz-1)
            self.roi_end.setMaximum(nz-1)
            self.roi_end.setValue(nz-1)
            
            # Try to extract pixel spacing from NIfTI header
            try:
                pixdim = nii.header['pixdim']
                self.pixel_spacing = [float(pixdim[1]), float(pixdim[2]), float(pixdim[3])]
                self.statusBar().showMessage(f"Loaded volume: {self.volume.shape}, spacing: {self.pixel_spacing} mm")
            except:
                self.pixel_spacing = [1.0, 1.0, 1.0]
                self.statusBar().showMessage(f"Loaded volume: {self.volume.shape} (default spacing)")
            
            self.update_all_views()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load NIfTI: {e}")
            print(e)
        finally:
            pass

    def load_dicom_file(self):
        # المسار الثابت اللي هنحفظ فيه NIfTI الناتج
        output_folder = r"D:\Programing\Biomedical Projects\Task 2\Files\Temp\Nifti File"

        # 1️⃣ يختار المستخدم مجلد DICOM
        dicom_folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not dicom_folder:
            return

        # 2️⃣ حذف أي ملفات قديمة من فولدر الإخراج
        if os.path.exists(output_folder):
            try:
                shutil.rmtree(output_folder)
            except Exception as e:
                QMessageBox.warning(self, "Warning", f"Failed to clear output folder: {e}")
        os.makedirs(output_folder, exist_ok=True)

        self.statusBar().showMessage(f"Converting DICOM folder to NIfTI... Please wait")
        QApplication.processEvents()

        try:
            # 3️⃣ تحويل ملفات DICOM إلى NIfTI (مع إعادة التوجيه لتفادي الصور المقلوبة)
            dicom2nifti.convert_directory(dicom_folder, output_folder, compression=True, reorient=True)

            # 4️⃣ الحصول على اسم الملف الناتج
            nii_files = [f for f in os.listdir(output_folder) if f.endswith(('.nii', '.nii.gz'))]
            if not nii_files:
                QMessageBox.critical(self, "Error", "Conversion failed: No NIfTI file found.")
                return

            nifti_path = os.path.join(output_folder, nii_files[0])
            self.statusBar().showMessage(f"NIfTI generated: {os.path.basename(nifti_path)}")

            # 5️⃣ تحميل الملف الناتج مباشرة في العارض
            # Note: load_nifti_direct should exist; if not, consider calling self.load_nifti(nifti_path) or implement it.
            self.load_nifti_direct(nifti_path)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to convert DICOM folder: {e}")
            print(e)

            # Fallback: allow user to open a single .dcm file if pydicom is available
            if not DICOM_AVAILABLE:
                QMessageBox.warning(self, "Warning", "pydicom not available")
                return

            path, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "", "DICOM Files (*.dcm)")
            if not path:
                return
            try:
                ds = pydicom.dcmread(path)
                arr = ds.pixel_array.astype(np.float32)
                if arr.ndim == 2:
                    arr = arr[:, :, np.newaxis]
                if arr.ndim == 3:
                    self.volume = arr.astype(np.float32)
                    self.original_volume = self.volume.copy()
                    nx, ny, nz = self.volume.shape
                    self.shape = (nx, ny, nz)
                    self.cross = [nx//2, ny//2, nz//2]
                    self.slice_idx = [nz//2, nx//2, ny//2, 0]
                    self.update_all_views()
                    self.statusBar().showMessage("Loaded DICOM pixel_array (single file)")
                else:
                    QMessageBox.information(self, "Info", "Loaded DICOM but shape not 2D/3D")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to read DICOM: {e}")
                print(e)

    def get_oblique_plane_normal(self):
        tx = math.radians(self.oblique_tilt_x)
        ty = math.radians(self.oblique_tilt_y)
        Rx = np.array([[1,0,0],[0, math.cos(tx), -math.sin(tx)],[0, math.sin(tx), math.cos(tx)]], dtype=np.float32)
        Ry = np.array([[math.cos(ty), 0, math.sin(ty)],[0,1,0],[-math.sin(ty),0, math.cos(ty)]], dtype=np.float32)
        n0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        n = Ry.dot(Rx.dot(n0))
        return n

    def get_oblique_projection_on_view(self, view_name):
        n = self.get_oblique_plane_normal()
        if self.volume is None:
            return None
        if view_name == 'Axial':
            view_normal = np.array([0.0,0.0,1.0])
            d = np.cross(n, view_normal)
            return (float(d[0]), float(-d[1]))
        elif view_name == 'Sagittal':
            view_normal = np.array([1.0,0.0,0.0])
            d = np.cross(n, view_normal)
            return (float(d[1]), float(d[2]))
        elif view_name == 'Coronal':
            view_normal = np.array([0.0,1.0,0.0])
            d = np.cross(n, view_normal)
            return (float(d[0]), float(d[2]))
        else:
            return None

    def sample_oblique(self, center, width=None, height=None):
        vol = self.volume
        nx, ny, nz = vol.shape
        if width is None: width = ny
        if height is None: height = nx
        n = self.get_oblique_plane_normal()
        up = np.array([0.0,1.0,0.0])
        if abs(np.dot(n, up)) > 0.9:
            up = np.array([1.0,0.0,0.0])
        e1 = np.cross(up, n); e1 = e1 / (np.linalg.norm(e1) + 1e-12)
        e2 = np.cross(n, e1); e2 = e2 / (np.linalg.norm(e2) + 1e-12)
        i_coords = (np.arange(height) - (height - 1) / 2.0)
        j_coords = (np.arange(width) - (width - 1) / 2.0)
        ii, jj = np.meshgrid(i_coords, j_coords, indexing='ij')
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        X = cx + ii * e1[0] + jj * e2[0]
        Y = cy + ii * e1[1] + jj * e2[1]
        Z = cz + ii * e1[2] + jj * e2[2]
        coords = np.vstack((X.ravel(), Y.ravel(), Z.ravel()))
        sampled = map_coordinates(vol, coords, order=1, mode='nearest')
        sampled = sampled.reshape((height, width))
        return sampled

    def handle_drag_from_view(self, view_name, dragging_type, img_x, img_y):
        if self.volume is None:
            return
        nx, ny, nz = self.shape
        if view_name == 'Axial':
            if dragging_type == 'v':
                self.cross[0] = int(clamp(round(img_x), 0, nx - 1))
            elif dragging_type == 'h':
                self.cross[1] = int(clamp(round(img_y), 0, ny - 1))
            elif dragging_type == 'oblique':
                w = self.canvas_axial.image_data.shape[1]; h = self.canvas_axial.image_data.shape[0]
                cx = w / 2.0; cy = h / 2.0
                ang = math.degrees(math.atan2(img_y - cy, img_x - cx))
                self.oblique_tilt_y = clamp(ang, -89, 89)
                self.tilt_y_slider.blockSignals(True)
                self.tilt_y_slider.setValue(int(self.oblique_tilt_y))
                self.tilt_y_slider.blockSignals(False)
                self.tilt_y_label.setText(f"{int(self.oblique_tilt_y)}°")
        elif view_name == 'Sagittal':
            if dragging_type == 'v':
                self.cross[1] = int(clamp(round(img_x), 0, ny - 1))
            elif dragging_type == 'h':
                self.cross[2] = int(clamp(round(img_y), 0, nz - 1))
            elif dragging_type == 'oblique':
                w = self.canvas_sagittal.image_data.shape[1]; h = self.canvas_sagittal.image_data.shape[0]
                cx = w / 2.0; cy = h / 2.0
                ang = math.degrees(math.atan2(img_y - cy, img_x - cx))
                self.oblique_tilt_x = clamp(-ang, -89, 89)
                self.tilt_x_slider.blockSignals(True)
                self.tilt_x_slider.setValue(int(self.oblique_tilt_x))
                self.tilt_x_slider.blockSignals(False)
                self.tilt_x_label.setText(f"{int(self.oblique_tilt_x)}°")
        elif view_name == 'Coronal':
            if dragging_type == 'v':
                self.cross[0] = int(clamp(round(img_x), 0, nx - 1))
            elif dragging_type == 'h':
                self.cross[2] = int(clamp(round(img_y), 0, nz - 1))
            elif dragging_type == 'oblique':
                w = self.canvas_coronal.image_data.shape[1]; h = self.canvas_coronal.image_data.shape[0]
                cx = w / 2.0; cy = h / 2.0
                ang = math.degrees(math.atan2(img_y - cy, img_x - cx))
                self.oblique_tilt_x = clamp(ang, -89, 89)
                self.tilt_x_slider.blockSignals(True)
                self.tilt_x_slider.setValue(int(self.oblique_tilt_x))
                self.tilt_x_slider.blockSignals(False)
                self.tilt_x_label.setText(f"{int(self.oblique_tilt_x)}°")

        self.slice_idx[0] = int(clamp(self.cross[2], 0, nz - 1))
        self.slice_idx[1] = int(clamp(self.cross[0], 0, nx - 1))
        self.slice_idx[2] = int(clamp(self.cross[1], 0, ny - 1))
        self.update_all_views()

    def slider_moved(self, view_name, val):
        if self.volume is None: return
        nx, ny, nz = self.shape
        if view_name == 'Axial':
            self.slice_idx[0] = clamp(val, 0, nz-1); self.cross[2] = int(self.slice_idx[0])
        elif view_name == 'Sagittal':
            self.slice_idx[1] = clamp(val, 0, nx-1); self.cross[0] = int(self.slice_idx[1])
        elif view_name == 'Coronal':
            self.slice_idx[2] = clamp(val, 0, ny-1); self.cross[1] = int(self.slice_idx[2])
        elif view_name == 'Oblique':
            self.slice_idx[3] = clamp(val, 0, 100)  # Store percentage
        self.update_all_views()

    def toggle_play(self, view_name, checked, btn):
        self.play_states[view_name] = checked
        btn.setText('Pause' if checked else 'Play')
        self.refresh_timer()

    def refresh_timer(self):
        any_play = any(self.play_states.values())
        if any_play and not self.timer.isActive():
            self.update_timer_interval(); self.timer.start()
        elif not any_play and self.timer.isActive():
            self.timer.stop()

    def update_timer_interval(self):
        fps = max(1, self.speed_slider.value())
        self.timer.setInterval(int(1000.0 / fps))
        self.speed_label.setText(f"{fps} FPS")

    def on_timer(self):
        if self.play_states.get('Axial'):
            self.step_view('Axial', 1)
        if self.play_states.get('Sagittal'):
            self.step_view('Sagittal', 1)
        if self.play_states.get('Coronal'):
            self.step_view('Coronal', 1)
        if self.play_states.get('Oblique'):
            self.step_view('Oblique', 1)

    def step_view(self, view_name, step):
        if self.volume is None: return
        nx, ny, nz = self.shape
        if view_name == 'Axial':
            self.slice_idx[0] = int(clamp(self.slice_idx[0] + step, 0, nz - 1)); self.cross[2] = self.slice_idx[0]
        elif view_name == 'Sagittal':
            self.slice_idx[1] = int(clamp(self.slice_idx[1] + step, 0, nx - 1)); self.cross[0] = self.slice_idx[1]
        elif view_name == 'Coronal':
            self.slice_idx[2] = int(clamp(self.slice_idx[2] + step, 0, ny - 1)); self.cross[1] = self.slice_idx[2]
        elif view_name == 'Oblique':
            self.slice_idx[3] = int(clamp(self.slice_idx[3] + step, 0, 100))
        self.update_all_views()

    def tilt_x_changed(self, v):
        self.oblique_tilt_x = float(v)
        self.tilt_x_label.setText(f"{v}°")
        self.update_all_views()
        
    def tilt_y_changed(self, v):
        self.oblique_tilt_y = float(v)
        self.tilt_y_label.setText(f"{v}°")
        self.update_all_views()
    
    def update_all_views(self):
        if self.volume is None:
            return

        nx, ny, nz = self.shape
        self.cross[0] = int(clamp(self.cross[0], 0, nx-1))
        self.cross[1] = int(clamp(self.cross[1], 0, ny-1))
        self.cross[2] = int(clamp(self.cross[2], 0, nz-1))
        self.slice_idx[0] = int(clamp(self.slice_idx[0], 0, nz-1))
        self.slice_idx[1] = int(clamp(self.slice_idx[1], 0, nx-1))
        self.slice_idx[2] = int(clamp(self.slice_idx[2], 0, ny-1))

        # ===== Get slices =====
        axial = self.volume[:, :, self.slice_idx[0]]
        sagittal = self.volume[self.slice_idx[1], :, :]
        coronal = self.volume[:, self.slice_idx[2], :]
        ob = self.sample_oblique(center=self.cross, width=coronal.shape[1], height=coronal.shape[0])

        # ===== Set base images =====
        self.canvas_axial.set_image(axial.T, v_pos=self.cross[0], h_pos=self.cross[1])
        self.canvas_sagittal.set_image(sagittal.T, v_pos=self.cross[1], h_pos=self.cross[2])
        self.canvas_coronal.set_image(coronal.T, v_pos=self.cross[0], h_pos=self.cross[2])
        self.canvas_oblique.set_image(ob.T, v_pos=ob.shape[1]//2, h_pos=ob.shape[0]//2)

        proj_ax = self.get_oblique_projection_on_view('Axial')
        proj_sg = self.get_oblique_projection_on_view('Sagittal')
        proj_cr = self.get_oblique_projection_on_view('Coronal')

        # ===== Surface Boundary (existing feature) =====
        boundary_contours = None
        if getattr(self, 'oblique_mode', None) and self.oblique_mode.currentText() == "Surface Boundary" and self.combined_mask is not None:
            z = int(self.slice_idx[0])
            mask_slice = self.combined_mask[:, :, z]
            contours = measure.find_contours(mask_slice.astype(np.uint8), 0.5)
            boundary_contours = contours
            self.canvas_oblique.update_display(
                oblique_direction_2d=None,
                boundary_contours=boundary_contours,
                measurement_points=self.measurement_points['Oblique']
            )

        # ===== NEW: Show NIfTI Mask only in Oblique View and only when Surface Mode is active =====
        elif getattr(self, "current_mode", "") == "Surface Mode" and self.mask_volume is not None:
            try:
                from skimage import measure
                mask_contours_oblique = measure.find_contours(
                    self.mask_volume[:, :, self.slice_idx[0]], 0.5
                )
            except Exception as e:
                print("Mask contour error (oblique):", e)
                mask_contours_oblique = None

            # Axial, Sagittal, Coronal — without mask
            self.canvas_axial.update_display(
                oblique_direction_2d=proj_ax,
                measurement_points=self.measurement_points['Axial']
            )
            self.canvas_sagittal.update_display(
                oblique_direction_2d=proj_sg,
                measurement_points=self.measurement_points['Sagittal']
            )
            self.canvas_coronal.update_display(
                oblique_direction_2d=proj_cr,
                measurement_points=self.measurement_points['Coronal']
            )

            # Oblique — show mask outline
            self.canvas_oblique.update_display(
                oblique_direction_2d=None,
                boundary_contours=mask_contours_oblique,
                measurement_points=self.measurement_points['Oblique']
            )

        # ===== Default: No mask =====
        else:
            self.canvas_axial.update_display(
                oblique_direction_2d=proj_ax,
                measurement_points=self.measurement_points['Axial']
            )
            self.canvas_sagittal.update_display(
                oblique_direction_2d=proj_sg,
                measurement_points=self.measurement_points['Sagittal']
            )
            self.canvas_coronal.update_display(
                oblique_direction_2d=proj_cr,
                measurement_points=self.measurement_points['Coronal']
            )
            self.canvas_oblique.update_display(
                oblique_direction_2d=None,
                measurement_points=self.measurement_points['Oblique']
            )

        # ===== Sync sliders =====
        if self.axial_slider:
            self.axial_slider.blockSignals(True); self.axial_slider.setValue(self.slice_idx[0]); self.axial_slider.blockSignals(False)
        if self.sag_slider:
            self.sag_slider.blockSignals(True); self.sag_slider.setValue(self.slice_idx[1]); self.sag_slider.blockSignals(False)
        if self.cor_slider:
            self.cor_slider.blockSignals(True); self.cor_slider.setValue(self.slice_idx[2]); self.cor_slider.blockSignals(False)
        if self.oblique_slider:
            self.oblique_slider.blockSignals(True); self.oblique_slider.setValue(int(self.slice_idx[3])); self.oblique_slider.blockSignals(False)

    def on_oblique_mode_changed(self, text):
        if text == "Surface Boundary":
            files, _ = QFileDialog.getOpenFileNames(self, "Select NIfTI mask files (1 or more)", "", "NIfTI Files (*.nii *.nii.gz)")
            if not files:
                self.oblique_mode.blockSignals(True)
                self.oblique_mode.setCurrentText("Oblique View")
                self.oblique_mode.blockSignals(False)
                return
            combined = None
            for f in files:
                try:
                    nii = nib.load(f)
                    m = np.asarray(nii.dataobj)
                    if m.ndim == 2:
                        m = m[:, :, np.newaxis]
                    if m.ndim != 3:
                        QMessageBox.warning(self, "Warning", f"Skipping file (unsupported shape): {os.path.basename(f)}")
                        continue
                    if combined is None:
                        combined = (m != 0)
                    else:
                        if combined.shape != m.shape:
                            QMessageBox.warning(self, "Warning", f"Mask {os.path.basename(f)} shape {m.shape} mismatches previous {combined.shape}; skipping")
                            continue
                        combined = np.logical_or(combined, (m != 0))
                except Exception as e:
                    QMessageBox.warning(self, "Warning", f"Failed to load mask {os.path.basename(f)}: {e}")
            if combined is None:
                QMessageBox.information(self, "Info", "No valid masks loaded; reverting to Oblique View")
                self.oblique_mode.blockSignals(True)
                self.oblique_mode.setCurrentText("Oblique View")
                self.oblique_mode.blockSignals(False)
                return
            self.combined_mask = combined.astype(np.uint8)
            if self.volume is None:
                self.volume = np.zeros_like(self.combined_mask, dtype=np.float32)
                self.original_volume = self.volume.copy()
                self.shape = self.volume.shape
                self.cross = [self.shape[0]//2, self.shape[1]//2, self.shape[2]//2]
                self.slice_idx = [self.shape[2]//2, self.shape[0]//2, self.shape[1]//2, 0]
            else:
                if self.combined_mask.shape != self.volume.shape:
                    QMessageBox.warning(self, "Warning", "Combined mask shape does not match current volume.")
            self.update_all_views()
        else:
            self.update_all_views()

    def export_roi(self):
        if self.volume is None:
            QMessageBox.warning(self, "Warning", "No volume loaded")
            return
        start = self.roi_start.value(); end = self.roi_end.value()
        if start >= end:
            QMessageBox.warning(self, "Warning", "Start must be < End")
            return
        if end > self.volume.shape[2] - 1:
            QMessageBox.warning(self, "Warning", f"End exceeds depth ({self.volume.shape[2]})")
            return
        roi = self.volume[:, :, start:end+1]
        path, _ = QFileDialog.getSaveFileName(self, "Save ROI", "", "NIfTI Files (*.nii.gz *.nii)")
        if not path:
            return
        try:
            nib.save(nib.Nifti1Image(roi, np.eye(4)), path)
            QMessageBox.information(self, "Saved", f"Saved ROI to {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save ROI: {e}")

    def apply_roi(self):
        """Apply ROI - extract and display ROI as temporary volume"""
        if self.volume is None:
            QMessageBox.warning(self, "Warning", "No volume loaded")
            return
        start = self.roi_start.value()
        end = self.roi_end.value()
        if start >= end:
            QMessageBox.warning(self, "Warning", "Start must be < End")
            return
        if end > self.original_volume.shape[2] - 1:
            QMessageBox.warning(self, "Warning", f"End exceeds depth ({self.original_volume.shape[2]})")
            return
        
        # Extract ROI
        roi = self.original_volume[:, :, start:end+1].copy()
        
        # Save as temporary file
        import tempfile
        if self.temp_roi_file and os.path.exists(self.temp_roi_file):
            try:
                os.remove(self.temp_roi_file)
            except:
                pass
        
        fd, self.temp_roi_file = tempfile.mkstemp(suffix='.nii.gz')
        os.close(fd)
        
        try:
            nib.save(nib.Nifti1Image(roi, np.eye(4)), self.temp_roi_file)
            
            # Load the ROI as current volume
            self.volume = roi.astype(np.float32)
            nx, ny, nz = self.volume.shape
            self.shape = (nx, ny, nz)
            self.cross = [nx//2, ny//2, nz//2]
            self.slice_idx = [nz//2, nx//2, ny//2, 0]
            
            # Update sliders
            if self.axial_slider:
                self.axial_slider.blockSignals(True)
                self.axial_slider.setMaximum(nz-1)
                self.axial_slider.setValue(self.slice_idx[0])
                self.axial_slider.blockSignals(False)
            if self.sag_slider:
                self.sag_slider.blockSignals(True)
                self.sag_slider.setMaximum(nx-1)
                self.sag_slider.setValue(self.slice_idx[1])
                self.sag_slider.blockSignals(False)
            if self.cor_slider:
                self.cor_slider.blockSignals(True)
                self.cor_slider.setMaximum(ny-1)
                self.cor_slider.setValue(self.slice_idx[2])
                self.cor_slider.blockSignals(False)
            
            # Update ROI controls
            self.roi_start.setMaximum(nz-1)
            self.roi_end.setMaximum(nz-1)
            self.roi_end.setValue(nz-1)
            
            # Enable reset button
            self.btn_reset_roi.setEnabled(True)
            
            self.update_all_views()
            self.statusBar().showMessage(f"Applied ROI: slices {start}-{end} (now viewing {nz} slices)")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply ROI: {e}")
            if self.temp_roi_file and os.path.exists(self.temp_roi_file):
                try:
                    os.remove(self.temp_roi_file)
                except:
                    pass
            self.temp_roi_file = None

    def reset_to_original(self):
        """Reset to original volume"""
        if self.original_volume is None:
            QMessageBox.warning(self, "Warning", "No original volume available")
            return
        
        # Clean up temp file
        if self.temp_roi_file and os.path.exists(self.temp_roi_file):
            try:
                os.remove(self.temp_roi_file)
            except:
                pass
        self.temp_roi_file = None
        
        # Restore original volume
        self.volume = self.original_volume.copy()
        nx, ny, nz = self.volume.shape
        self.shape = (nx, ny, nz)
        self.cross = [nx//2, ny//2, nz//2]
        self.slice_idx = [nz//2, nx//2, ny//2, 0]
        
        # Update sliders
        if self.axial_slider:
            self.axial_slider.blockSignals(True)
            self.axial_slider.setMaximum(nz-1)
            self.axial_slider.setValue(self.slice_idx[0])
            self.axial_slider.blockSignals(False)
        if self.sag_slider:
            self.sag_slider.blockSignals(True)
            self.sag_slider.setMaximum(nx-1)
            self.sag_slider.setValue(self.slice_idx[1])
            self.sag_slider.blockSignals(False)
        if self.cor_slider:
            self.cor_slider.blockSignals(True)
            self.cor_slider.setMaximum(ny-1)
            self.cor_slider.setValue(self.slice_idx[2])
            self.cor_slider.blockSignals(False)
        
        # Update ROI controls
        self.roi_start.setMaximum(nz-1)
        self.roi_end.setMaximum(nz-1)
        self.roi_end.setValue(nz-1)
        
        # Disable reset button
        self.btn_reset_roi.setEnabled(False)
        
        self.update_all_views()
        self.statusBar().showMessage(f"Reset to original volume: {self.volume.shape}")

    # ===== GLOBAL CINE CONTROLS =====
    def toggle_global_play(self, checked):
        """Toggle play/pause for all views simultaneously"""
        self.global_play = checked
        
        if checked:
            self.btn_global_play.setText("Pause All")
            # Start all views playing
            self.play_states = {'Axial': True, 'Sagittal': True, 'Coronal': True, 'Oblique': True}
            if hasattr(self, 'axial_play_btn'):
                self.axial_play_btn.setChecked(True)
                self.axial_play_btn.setText('Pause')
            if hasattr(self, 'sag_play_btn'):
                self.sag_play_btn.setChecked(True)
                self.sag_play_btn.setText('Pause')
            if hasattr(self, 'cor_play_btn'):
                self.cor_play_btn.setChecked(True)
                self.cor_play_btn.setText('Pause')
            if hasattr(self, 'oblique_play_btn'):
                self.oblique_play_btn.setChecked(True)
                self.oblique_play_btn.setText('Pause')
        else:
            self.btn_global_play.setText("Play All")
            # Stop all views
            self.play_states = {'Axial': False, 'Sagittal': False, 'Coronal': False, 'Oblique': False}
            if hasattr(self, 'axial_play_btn'):
                self.axial_play_btn.setChecked(False)
                self.axial_play_btn.setText('Play')
            if hasattr(self, 'sag_play_btn'):
                self.sag_play_btn.setChecked(False)
                self.sag_play_btn.setText('Play')
            if hasattr(self, 'cor_play_btn'):
                self.cor_play_btn.setChecked(False)
                self.cor_play_btn.setText('Play')
            if hasattr(self, 'oblique_play_btn'):
                self.oblique_play_btn.setChecked(False)
                self.oblique_play_btn.setText('Play')
        
        self.refresh_timer()
        self.statusBar().showMessage("Global play: " + ("ON" if checked else "OFF"))

    def global_step(self, direction):
        """Step all views forward or backward simultaneously"""
        if self.volume is None:
            return
        
        self.step_view('Axial', direction)
        self.step_view('Sagittal', direction)
        self.step_view('Coronal', direction)
        self.step_view('Oblique', direction)
        
        self.statusBar().showMessage(f"All views stepped {'forward' if direction > 0 else 'backward'}")

    # ===== ZOOM CONTROLS =====
    def zoom_in_all(self):
        """Zoom in all viewports"""
        self.canvas_axial.zoom = min(8.0, self.canvas_axial.zoom * 1.2)
        self.canvas_sagittal.zoom = min(8.0, self.canvas_sagittal.zoom * 1.2)
        self.canvas_coronal.zoom = min(8.0, self.canvas_coronal.zoom * 1.2)
        self.canvas_oblique.zoom = min(8.0, self.canvas_oblique.zoom * 1.2)
        self.update_all_views()

    def zoom_out_all(self):
        """Zoom out all viewports"""
        self.canvas_axial.zoom = max(0.2, self.canvas_axial.zoom / 1.2)
        self.canvas_sagittal.zoom = max(0.2, self.canvas_sagittal.zoom / 1.2)
        self.canvas_coronal.zoom = max(0.2, self.canvas_coronal.zoom / 1.2)
        self.canvas_oblique.zoom = max(0.2, self.canvas_oblique.zoom / 1.2)
        self.update_all_views()

    def reset_zoom_all(self):
        """Reset zoom to 1.0 for all viewports"""
        self.canvas_axial.zoom = 1.0
        self.canvas_sagittal.zoom = 1.0
        self.canvas_coronal.zoom = 1.0
        self.canvas_oblique.zoom = 1.0
        self.update_all_views()

    # ===== BRIGHTNESS & CONTRAST CONTROLS =====
    def update_brightness(self, value):
        """Update brightness for all views"""
        self.brightness_values = [value] * 4
        self.brightness_label.setText(f"{value}")
        self.update_all_views()

    def update_contrast(self, value):
        """Update contrast for all views"""
        contrast = value / 100.0
        self.contrast_values = [contrast] * 4
        self.contrast_label.setText(f"{contrast:.2f}")
        self.update_all_views()

    def reset_brightness_contrast(self):
        """Reset brightness and contrast to defaults"""
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.brightness_values = [0, 0, 0, 0]
        self.contrast_values = [1.0, 1.0, 1.0, 1.0]
        self.update_all_views()

    # ===== MEASUREMENT TOOLS =====
    def toggle_measurement_mode(self, checked):
        """Toggle measurement mode on/off"""
        self.measurement_mode = checked
        if checked:
            self.btn_measure.setText("Disable Measurement")
            self.measure_label.setText("Click two points to measure distance")
            self.statusBar().showMessage("Measurement mode enabled - click two points on any view")
        else:
            self.btn_measure.setText("Enable Measurement")
            self.measure_label.setText("Click two points to measure")
            self.statusBar().showMessage("Measurement mode disabled")

    def add_measurement_point(self, view_name, x, y):
        """Add a measurement point and calculate distance if two points exist"""
        points = self.measurement_points[view_name]
        points.append((x, y))
        
        if len(points) == 2:
            # Calculate distance
            p1, p2 = points[0], points[1]
            
            # Get appropriate pixel spacing based on view
            if view_name == 'Axial':
                spacing_x = self.pixel_spacing[0]
                spacing_y = self.pixel_spacing[1]
            elif view_name == 'Sagittal':
                spacing_x = self.pixel_spacing[1]
                spacing_y = self.pixel_spacing[2]
            elif view_name == 'Coronal':
                spacing_x = self.pixel_spacing[0]
                spacing_y = self.pixel_spacing[2]
            else:  # Oblique
                spacing_x = self.pixel_spacing[0]
                spacing_y = self.pixel_spacing[1]
            
            dx = (p2[0] - p1[0]) * spacing_x
            dy = (p2[1] - p1[1]) * spacing_y
            distance = math.sqrt(dx**2 + dy**2)
            
            self.measure_label.setText(f"{view_name}: {distance:.2f} mm\n({len(points)} points)")
            self.statusBar().showMessage(f"Distance: {distance:.2f} mm on {view_name} view")
        elif len(points) > 2:
            # Reset to new measurement
            self.measurement_points[view_name] = [(x, y)]
            self.measure_label.setText(f"{view_name}: Point 1 marked")
        else:
            self.measure_label.setText(f"{view_name}: Point {len(points)} marked")
        
        self.update_all_views()

    def clear_measurements(self):
        """Clear all measurement points"""
        self.measurement_points = {'Axial': [], 'Sagittal': [], 'Coronal': [], 'Oblique': []}
        self.measure_label.setText("Click two points to measure")
        self.statusBar().showMessage("Measurements cleared")
        self.update_all_views()

    # ===== COLORMAP CONTROL =====
    def update_colormap(self, colormap_name):
        """Update the colormap for all views"""
        self.current_colormap = colormap_name
        self.statusBar().showMessage(f"Colormap changed to: {colormap_name}")
        self.update_all_views()

    # ========================================
    # --- AI MODEL FUNCTIONS ---
    # ========================================
    
    # --- NEW: Function to let user select a new model ---
    def select_ai_model(self):
        """
        Opens a file dialog for the user to select a new .h5 model file.
        If selected, it updates self.model_path and reloads the model.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Load AI Model", "", "HDF5 Files (*.h5)")
        if path:
            self.model_path = path  # Update the instance variable
            self.load_ai_model()    # Reload the model with the new path
        else:
            self.statusBar().showMessage("AI model selection cancelled.")

    # --- UPDATED: Now uses self.model_path ---
    def load_ai_model(self):
        """Loads the Keras AI model from the path stored in self.model_path."""
        
        # Check if the path is set and exists
        if not self.model_path or not os.path.exists(self.model_path):
            self.statusBar().showMessage(f"AI Model path not set or file not found: {self.model_path}")
            # Only show popup if the UI is already built (i.e., not on first launch)
            if hasattr(self, 'btn_classify_file'): 
                QMessageBox.warning(self, "AI Model Error", 
                                    f"Model file not found. Please select a valid model using 'Change AI Model'.\n\nPath: {self.model_path}")
            return
        
        try:
            self.statusBar().showMessage(f"Loading AI model: {os.path.basename(self.model_path)}...")
            self.ai_model = tf.keras.models.load_model(self.model_path, compile=False)
            self.statusBar().showMessage("AI model loaded successfully.")
        except Exception as e:
            self.ai_model = None
            self.statusBar().showMessage(f"Failed to load AI model: {e}")
            QMessageBox.critical(self, "AI Model Error", f"Failed to load model:\n{e}")
            print(f"Failed to load AI model: {e}")

    # --- In-Memory Prediction (for 'Predict Slice' buttons) ---
    def preprocess_array(self, slice_data):
        """
        Prepares a 2D numpy array (from the viewer) for the AI model.
        Converts (h, w) numpy array to (1, 128, 128, 1) tensor.
        """
        try:
            # 1. Ensure data is float32
            arr = np.asarray(slice_data, dtype=np.float32)
            
            # 2. Normalize from its own min/max to [0, 1]
            if arr.max() > arr.min():
                arr = (arr - arr.min()) / (arr.max() - arr.min())
            else:
                arr = np.zeros_like(arr) # Avoid division by zero
            
            # 3. Add channel dimension: (h, w) -> (h, w, 1)
            arr = np.expand_dims(arr, axis=-1)
            
            # 4. Add batch dimension: (h, w, 1) -> (1, h, w, 1)
            arr_batch = np.expand_dims(arr, axis=0)
            
            # 5. Resize using TensorFlow to (1, 128, 128, 1)
            resized_tensor = tf.image.resize(arr_batch, (self.IMG_SIZE, self.IMG_SIZE))
            
            return resized_tensor
        except Exception as e:
            print(f"Error in preprocess_array: {e}")
            traceback.print_exc()
            return None

    def run_prediction(self, view_name):
        """
        Gets the current slice, preprocesses it, and runs the AI prediction.
        Displays the result in a QMessageBox.
        """
        if self.ai_model is None:
            QMessageBox.warning(self, "AI Error", "The AI model is not loaded. Please select a model using 'Change AI Model'.")
            return
            
        if self.volume is None:
            QMessageBox.warning(self, "AI Error", "Please load a NIfTI volume first.")
            return

        # 1. Get the raw image data from the correct canvas
        slice_data = None
        if view_name == 'Axial':
            slice_data = self.canvas_axial.image_data 
        elif view_name == 'Sagittal':
            slice_data = self.canvas_sagittal.image_data
        elif view_name == 'Coronal':
            slice_data = self.canvas_coronal.image_data
        
        if slice_data is None:
            QMessageBox.critical(self, "AI Error", "Could not retrieve slice data for prediction.")
            return

        # 2. Preprocess the numpy array
        self.statusBar().showMessage(f"Preprocessing {view_name} slice for AI...")
        processed_data = self.preprocess_array(slice_data)
        
        if processed_data is None:
            QMessageBox.critical(self, "AI Error", "Failed to preprocess the image array.")
            self.statusBar().showMessage("AI preprocessing failed.")
            return

        # 3. Run prediction
        try:
            self.statusBar().showMessage(f"Running prediction on {view_name} slice...")
            prediction = self.ai_model.predict(processed_data)
            
            predicted_index = np.argmax(prediction)
            predicted_class = self.CLASSES[predicted_index]
            confidence = np.max(prediction) * 100
            
            # 4. Show results
            result_text = (f"View Tested: {view_name}\n\n"
                           f"Predicted Class: {predicted_class.upper()}\n"
                           f"Confidence: {confidence:.2f}%")
                           
            QMessageBox.information(self, "AI Slice Prediction Result", result_text)
            self.statusBar().showMessage(f"AI Result: {predicted_class.upper()} ({confidence:.2f}%)")

        except Exception as e:
            QMessageBox.critical(self, "AI Prediction Error", f"An error occurred during prediction:\n{e}")
            self.statusBar().showMessage("AI prediction failed.")
            print(f"AI Prediction Error: {e}")
            traceback.print_exc()

    # --- File-based AI Classification Functions (for 'Classify NIfTI File' button) ---

    def preprocess_image(self, img_path):
        """
        Loads and prepares a single IMAGE FILE for the model.
        (This is from your original AI script)
        """
        try:
            # Use the aliased import from Keras
            img = keras_image_proc.load_img(img_path, 
                                            target_size=(self.IMG_SIZE, self.IMG_SIZE), 
                                            color_mode='grayscale')
            img_array = keras_image_proc.img_to_array(img)
            img_array /= 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
        except Exception as e:
            print(f"\n[ERROR] Could not process image: {e}")
            self.statusBar().showMessage(f"AI Error: Could not process temp image: {e}")
            return None

    def classify_nifti_file(self):
        """
        Takes the middle axial slice, saves it as a PNG,
        and runs the AI model on that FILE to classify orientation.
        """
        if self.original_volume is None:
            QMessageBox.warning(self, "AI Error", "Please load a NIfTI volume first.")
            return

        if self.ai_model is None:
            QMessageBox.warning(self, "AI Error", "The AI model is not loaded. Please select a model using 'Change AI Model'.")
            return

        try:
            # 1. Ensure temp directory exists
            os.makedirs(self.temp_save_path, exist_ok=True)
            temp_file_path = os.path.join(self.temp_save_path, "temp_classify_slice.png")

            # 2. Get middle axial slice from ORIGINAL volume
            nx, ny, nz = self.original_volume.shape
            middle_slice_index = nz // 2
            
            # Get slice data: shape (nx, ny)
            slice_data = self.original_volume[:, :, middle_slice_index].astype(np.float32)

            # 3. Normalize and convert to 8-bit for saving
            if slice_data.max() > slice_data.min():
                slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
            
            arr_8bit = (slice_data * 255).astype(np.uint8)
            
            # Save the image transposed (w, h) or (ny, nx) to match
            # the orientation shown in the Axial view.
            
            # --- 🔴 ERROR FIX HERE 🔴 ---
            # We must use .copy() after transposing to ensure
            # the data is in a contiguous C-order buffer for QImage
            arr_8bit_transposed = arr_8bit.T.copy() 
            # --------------------------
            
            h_t, w_t = arr_8bit_transposed.shape
            bytes_per_line = w_t
            
            qimg = QImage(arr_8bit_transposed.data, w_t, h_t, bytes_per_line, QImage.Format_Grayscale8)
            
            if not qimg.save(temp_file_path, "PNG"):
                QMessageBox.critical(self, "Save Error", f"Failed to save temporary PNG file to:\n{temp_file_path}")
                return

            self.statusBar().showMessage(f"Saved temp slice to {temp_file_path}")

            # 4. Preprocess the SAVED FILE
            processed_image = self.preprocess_image(temp_file_path)

            if processed_image is None:
                QMessageBox.critical(self, "AI Error", "Failed to preprocess the saved temporary image.")
                return

            # 5. Run prediction
            self.statusBar().showMessage(f"Classifying NIfTI file orientation...")
            prediction = self.ai_model.predict(processed_image)
            
            predicted_index = np.argmax(prediction)
            predicted_class = self.CLASSES[predicted_index]
            confidence = np.max(prediction) * 100

            # 6. Show results
            result_text = (f"NIfTI File Classification\n\n"
                           f"Predicted Orientation: {predicted_class.upper()}\n"
                           f"Confidence: {confidence:.2f}%\n\n"
                           f"(Based on middle axial slice saved to {temp_file_path})")
                           
            QMessageBox.information(self, "AI File Classification Result", result_text)
            self.statusBar().showMessage(f"AI File Classification: {predicted_class.upper()} ({confidence:.2f}%)")

        except Exception as e:
            QMessageBox.critical(self, "AI Classification Error", f"An error occurred:\n{e}")
            self.statusBar().showMessage("AI classification failed.")
            print(f"AI Classification Error: {e}")
            traceback.print_exc()

def main():
    app = QApplication(sys.argv)
    w = MPRViewer()
    w.showMaximized()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()