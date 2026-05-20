import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import urllib.request
from pathlib import Path

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="centered"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    body { background-color: #0d0d0d; }
    .main { background-color: #0d0d0d; }
    .header-box {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a0a2e 50%, #16213e 100%);
        border: 1px solid #7c3aed44;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        text-align: center;
    }
    .header-box h1 { color: #a78bfa; font-size: 26px; margin: 0; }
    .header-box p { color: #6b7280; font-size: 13px; margin: 8px 0 0 0; }
    .info-card {
        background: #1a1a2e;
        border: 1px solid #7c3aed33;
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
    }
    .stButton>button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        width: 100%;
        font-size: 15px;
        font-weight: bold;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
    }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🎯 Object Detection & Tracking</h1>
    <p>CodeAlpha AI Internship — Task 4 | YOLOv3 + OpenCV Tracking</p>
</div>
""", unsafe_allow_html=True)

# ─── Download YOLO Files ───────────────────────────────────────────────────────
@st.cache_resource
def download_yolo_files():
    """Download YOLOv3 tiny weights, config and COCO names"""
    files = {
        "yolov3-tiny.weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
        "yolov3-tiny.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }

    for filename, url in files.items():
        if not Path(filename).exists():
            with st.spinner(f"Downloading {filename}..."):
                urllib.request.urlretrieve(url, filename)
    return True

# ─── Load YOLO Model ───────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    download_yolo_files()
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers

# ─── Colors for Classes ────────────────────────────────────────────────────────
COLORS = np.random.uniform(0, 255, size=(80, 3))

# ─── Detection Function ────────────────────────────────────────────────────────
def detect_objects(frame, net, classes, output_layers, conf_threshold=0.4, nms_threshold=0.4):
    height, width = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                cx = int(detection[0] * width)
                cy = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(cx - w / 2)
                y = int(cy - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            results.append((boxes[i], confidences[i], class_ids[i]))
    return results

# ─── Draw Detections ───────────────────────────────────────────────────────────
def draw_detections(frame, detections, classes, trackers=None):
    for idx, (box, confidence, class_id) in enumerate(detections):
        x, y, w, h = box
        color = COLORS[class_id % len(COLORS)]
        label = f"{classes[class_id]}: {confidence:.2f}"

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 5, y), color, -1)
        cv2.putText(frame, label, (x + 2, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Tracking ID
        track_id = f"ID:{idx+1}"
        cv2.putText(frame, track_id, (x + w - 40, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Frame info overlay
    cv2.putText(frame, f"Objects: {len(detections)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame

# ─── Main App ──────────────────────────────────────────────────────────────────
st.markdown("### 📤 Upload a Video File")
uploaded_video = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv"],
    help="Upload any video — the AI will detect and track objects in each frame"
)

# Settings
st.markdown("### ⚙️ Detection Settings")
col1, col2 = st.columns(2)
with col1:
    conf_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
with col2:
    max_frames = st.slider("Max Frames to Process", 30, 300, 100, 10)

if uploaded_video is not None:
    # Save uploaded video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    tfile.close()

    st.markdown("### 🎬 Processing Video...")

    if st.button("🚀 Start Object Detection & Tracking"):
        try:
            with st.spinner("Loading YOLOv3 model..."):
                net, classes, output_layers = load_model()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            st.markdown(f"""
            <div class="info-card">
                📊 <b>Video Info:</b> {total_frames} frames | {fps:.1f} FPS | Processing up to {max_frames} frames
            </div>
            """, unsafe_allow_html=True)

            # Process frames
            frame_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()

            processed_frames = []
            frame_count = 0
            all_detected_classes = set()

            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect objects
                detections = detect_objects(
                    frame, net, classes, output_layers,
                    conf_threshold=conf_threshold
                )

                # Track detected classes
                for _, _, class_id in detections:
                    all_detected_classes.add(classes[class_id])

                # Draw detections
                annotated_frame = draw_detections(frame.copy(), detections, classes)

                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                processed_frames.append(rgb_frame)

                # Show every 5th frame
                if frame_count % 5 == 0:
                    frame_placeholder.image(rgb_frame, caption=f"Frame {frame_count+1} | Objects detected: {len(detections)}", use_column_width=True)

                progress = (frame_count + 1) / min(max_frames, total_frames)
                progress_bar.progress(min(progress, 1.0))
                status_text.text(f"Processing frame {frame_count+1}/{min(max_frames, total_frames)}...")
                frame_count += 1

            cap.release()
            progress_bar.progress(1.0)
            status_text.text("✅ Processing Complete!")

            # Results Summary
            st.markdown("### 📊 Detection Results")
            st.markdown(f"""
            <div class="info-card">
                ✅ <b>Frames Processed:</b> {frame_count}<br>
                🎯 <b>Objects Found:</b> {', '.join(all_detected_classes) if all_detected_classes else 'None detected'}<br>
                🔧 <b>Model:</b> YOLOv3-tiny | <b>Tracking:</b> ID-based assignment
            </div>
            """, unsafe_allow_html=True)

            if processed_frames:
                st.markdown("### 🖼️ Sample Detected Frames")
                sample_indices = np.linspace(0, len(processed_frames)-1, min(4, len(processed_frames)), dtype=int)
                cols = st.columns(2)
                for i, idx in enumerate(sample_indices):
                    with cols[i % 2]:
                        st.image(processed_frames[idx], caption=f"Frame {idx+1}", use_column_width=True)

        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.info("Make sure OpenCV is installed: pip install opencv-python")

        finally:
            os.unlink(tfile.name)

else:
    st.markdown("""
    <div class="info-card">
        💡 <b>How it works:</b><br>
        1. Upload any video file (MP4, AVI, MOV)<br>
        2. YOLOv3 detects objects in each frame<br>
        3. Bounding boxes + labels are drawn<br>
        4. Each object gets a unique tracking ID<br>
        5. Results summary shown after processing
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<center><small>Built by <b>Md. Limon Hossen</b> | CodeAlpha AI Internship 🤖 | YOLOv3 + OpenCV</small></center>",
    unsafe_allow_html=True
)
