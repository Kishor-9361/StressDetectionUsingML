"""
One-time script. Run once to create face_indicators_train.csv and face_indicators_test.csv.
These are used to train the lightweight 18-feature face expert.
After running, delete this script — it is not needed at runtime.
"""
import mediapipe as mp
import cv2
import numpy as np
import os
import csv

mp_face_mesh = mp.solutions.face_mesh

def dist(pts, a, b):
    return float(np.linalg.norm(pts[a] - pts[b]))

def compute_18_indicators(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # Use the FaceMeshWrapper or a standard local fallback FaceMesh context manager:
    try:
        # standard media pipeSolutions FaceMesh fallback:
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                    refine_landmarks=False,
                                    min_detection_confidence=0.5) as fm:
            res = fm.process(rgb)
    except Exception as e:
        print(f"Error initializing standard FaceMesh: {e}")
        return None

    if not res.multi_face_landmarks:
        return None

    lm  = res.multi_face_landmarks[0].landmark
    pts = np.array([[l.x * w, l.y * h] for l in lm])

    faceH = dist(pts, 10, 152) + 1e-6
    faceW = dist(pts, 234, 454) + 1e-6
    iod   = dist(pts, 33, 263) + 1e-6

    earL = (dist(pts, 159, 145) + dist(pts, 158, 153)) / (2 * dist(pts, 33, 133) + 1e-6)
    earR = (dist(pts, 386, 374) + dist(pts, 385, 380)) / (2 * dist(pts, 362, 263) + 1e-6)
    avgEAR = (earL + earR) / 2

    return [
        earL,
        earR,
        avgEAR,
        0.0,                                          # blink_velocity — static image, set 0
        dist(pts, 55, 159) / faceH,                  # brow_descent_left
        dist(pts, 285, 386) / faceH,                 # brow_descent_right
        abs(dist(pts, 55, 159) - dist(pts, 285, 386)) / faceH,  # brow_asymmetry
        dist(pts, 13, 14) / (dist(pts, 61, 291) + 1e-6),        # lip_compression
        faceW / faceH,                                # jaw_tension
        (dist(pts, 61, 4) + dist(pts, 291, 4)) / (2 * faceH),   # mouth_corner_pull
        dist(pts, 10, 151) / faceH,                  # forehead_tension
        faceH / iod,                                  # face_height_norm
        0.0,                                          # head_tilt — compute if needed
        0.0,                                          # temporal_x_var — static image
        0.0,                                          # temporal_y_var — static image
        avgEAR,                                       # eye_openness_ratio
        0.9,                                          # landmark_confidence (detected)
        dist(pts, 4, 50) / faceH,                    # nose_wrinkle
    ]

def process_split(root_dir, split, output_csv):
    rows = []
    for label_name, label_val in [('stress', 1), ('nostress', 0)]:
        folder = os.path.join(root_dir, split, label_name)
        if not os.path.exists(folder):
            print(f'WARNING: {folder} not found')
            continue
        images = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f'Processing {len(images)} images from {folder}...')
        for img_name in images:
            path = os.path.join(folder, img_name)
            indicators = compute_18_indicators(path)
            if indicators is not None:
                rows.append(indicators + [label_val])

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [
            'left_ear', 'right_ear', 'avg_ear', 'blink_velocity',
            'brow_descent_left', 'brow_descent_right', 'brow_asymmetry',
            'lip_compression', 'jaw_tension', 'mouth_corner_pull',
            'forehead_tension', 'face_height_norm', 'head_tilt',
            'temporal_x_var', 'temporal_y_var', 'eye_openness_ratio',
            'landmark_confidence', 'nose_wrinkle', 'label',
        ]
        writer.writerow(header)
        writer.writerows(rows)
    print(f'Saved {len(rows)} rows to {output_csv}')

if __name__ == '__main__':
    FACES_ROOT = r'../datasets/facesData'  # updated path
    process_split(FACES_ROOT, 'train', 'face_indicators_train.csv')
    process_split(FACES_ROOT, 'test',  'face_indicators_test.csv')
    print('Done.')
