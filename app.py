import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os

# ----------------------------
# CONFIG
# ----------------------------
YOLO_MODEL_PATH = "models/best.pt"
YOLO_VAL_FOLDER = "D:\\myproject\\brain-tumor-using-yolo\\val"
PIXEL_SPACING = (0.4688, 0.4688)  # mm per pixel
CONFIDENCE = 0.5

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------
@st.cache_resource
def load_yolo():
    return YOLO(YOLO_MODEL_PATH)

model = load_yolo()

# ----------------------------
# YOLO: SINGLE IMAGE
# ----------------------------
def detect_single_image(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return None, []

    results = model.predict(img_bgr, conf=CONFIDENCE, verbose=False)
    annotated = img_bgr.copy()
    detections = []

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes else []
        label = "No tumor detected"

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            w_mm = (x2 - x1) * PIXEL_SPACING[0]
            h_mm = (y2 - y1) * PIXEL_SPACING[1]

            label = f"Tumor: {w_mm:.1f}mm × {h_mm:.1f}mm"
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            detections.append({"w_mm": w_mm, "h_mm": h_mm})

        cv2.rectangle(annotated, (10,10), (420,55), (0,0,0), -1)
        cv2.putText(annotated, label, (20,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated), detections

# ----------------------------
# YOLO: FOLDER MODE
# ----------------------------
def detect_folder(folder_path, max_images=200, popup=True):
    images = [f for f in os.listdir(folder_path)
              if f.lower().endswith((".jpg",".png",".jpeg",".bmp",".tiff"))]

    results = []
    win = "Brain Tumor Detection (ESC to stop)"

    if popup:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    for i, fname in enumerate(images):
        if i >= max_images:
            break

        path = os.path.join(folder_path, fname)
        img = cv2.imread(path)
        if img is None:
            continue

        res = model.predict(img, conf=CONFIDENCE, verbose=False)
        annotated = img.copy()
        detections = []
        label = "No tumor detected"

        for r in res:
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes else []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                w_mm = (x2 - x1) * PIXEL_SPACING[0]
                h_mm = (y2 - y1) * PIXEL_SPACING[1]

                label = f"Tumor: {w_mm:.1f}mm × {h_mm:.1f}mm"
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
                detections.append({"w_mm": w_mm, "h_mm": h_mm})

        cv2.rectangle(annotated, (10,10), (420,55), (0,0,0), -1)
        cv2.putText(annotated, label, (20,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        if popup:
            cv2.imshow(win, annotated)
            if cv2.waitKey(300) & 0xFF == 27:
                break

        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        results.append((fname, Image.fromarray(annotated), detections))

    if popup:
        cv2.destroyAllWindows()

    return results

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
st.title("🧠 Brain Tumor Detection (YOLO)")

option = st.radio("Choose mode", ("Upload MRI Image", "Run on Validation Folder"))

if option == "Upload MRI Image":
    uploaded = st.file_uploader("Upload MRI image", type=["jpg","png","jpeg","bmp","tiff"])
    if st.button("Detect Tumor"):
        if uploaded:
            img, dets = detect_single_image(uploaded.getvalue())
            st.image(img, use_column_width=True)
            if dets:
                for i,d in enumerate(dets,1):
                    st.write(f"**Tumor {i}:** {d['w_mm']:.1f} mm × {d['h_mm']:.1f} mm")
        else:
            st.warning("Please upload an image.")

else:
    st.info("Runs YOLO on validation folder and opens popup window.")
    if st.button("Run on Folder"):
        results = detect_folder(YOLO_VAL_FOLDER)
        st.success(f"Processed {len(results)} images")
        cols = st.columns(4)
        for i,(name,img,dets) in enumerate(results):
            cols[i % 4].image(img, caption=f"{name} | {len(dets)} tumors", use_column_width=True)
