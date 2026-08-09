import cv2
import numpy as np
from ultralytics import YOLO, settings
import json
import os

# ---------------------------------------------------------
# Configuration & Setup
# ---------------------------------------------------------
'''
Nose
Left Eye
Right Eye
Left Ear
Right Ear
Left Shoulder
Right Shoulder
Left Elbow
Right Elbow
Left Wrist
Right Wrist
Left Hip
Right Hip
Left Knee
Right Knee
Left Ankle
Right Ankle
'''

settings.update({
    "weights_dir": "./iofiles"
})
det_model = YOLO('yolo11x.pt')
pose_model = YOLO('yolo11x-pose.pt')
device = 'cpu'
image_path = './iofiles/f99a3cd6-93c4-43aa-a155-e76b03578dd8.jpg'
save_img_path = f'./iofiles/cropped_{image_path.split("/")[2]}'
save_kpts_path = './iofiles/normalized_keypoints.json'

# New 13-point skeleton (Index 0 is the consolidated Face point)
NEW_SKELETON = [
    (0, 1), (0, 2), (1, 2),          # Face to shoulders & shoulder line
    (1, 3), (3, 5), (2, 4), (4, 6),  # Arms
    (1, 7), (2, 8), (7, 8),          # Torso
    (7, 9), (9, 11), (8, 10), (10, 12) # Legs
]

KEYPOINT_NAMES = [
    "Face", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", 
    "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee", "R_Knee", 
    "L_Ankle", "R_Ankle"
]


def manual_annotation_gui(image):
    """
    Opens a GUI to allow the user to manually click the 13 keypoints 
    if the YOLO model fails to detect a person.
    """
    window_name = "Manual Annotation - Click the requested points"
    
    state = {
        'pts': [],
        'current_idx': 0
    }
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if state['current_idx'] < len(KEYPOINT_NAMES):
                state['pts'].append([float(x), float(y)])
                state['current_idx'] += 1
                
                if state['current_idx'] < len(KEYPOINT_NAMES):
                    print(f"[{state['current_idx'] + 1}/13] Please click on: **{KEYPOINT_NAMES[state['current_idx']]}**")
                else:
                    print("\nAll keypoints placed! Press ENTER in the image window to continue.")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- MANUAL ANNOTATION MODE ---")
    print(f"[1/13] Please click on: **{KEYPOINT_NAMES[0]}**")

    while True:
        display_img = image.copy()
        
        # Draw placed points
        for i, pt in enumerate(state['pts']):
            cv2.circle(display_img, (int(pt[0]), int(pt[1])), 5, (0, 255, 255), -1)
            cv2.putText(display_img, str(i), (int(pt[0]) + 5, int(pt[1]) - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow(window_name, display_img)
        
        key = cv2.waitKey(1) & 0xFF
        # Break on Enter (13) only if all points are placed
        if key == 13 and state['current_idx'] >= len(KEYPOINT_NAMES): 
            break

    cv2.destroyAllWindows()
    return state['pts']

def consolidate_and_shift_keypoints(kpts, confs, bbox):
    """Consolidates face points, strips unneeded points, and shifts coords."""
    x1, y1, x2, y2 = map(int, bbox)
    
    face_kpts = [kpts[i] for i in range(5) if confs[i] > 0.5]
    face_pt = np.mean(face_kpts, axis=0).tolist() if face_kpts else [0.0, 0.0]

    new_kpts = [face_pt]
    for i in range(5, 17):
        new_kpts.append(kpts[i].tolist() if confs[i] > 0.5 else [0.0, 0.0])

    for i in range(len(new_kpts)):
        if new_kpts[i] != [0.0, 0.0]:
            new_kpts[i][0] -= x1
            new_kpts[i][1] -= y1
            
    return new_kpts

def normalize_keypoints(kpts, crop_w, crop_h):
    """Normalizes keypoints using Torso length, centering at the hip.
    
    Convention: positive Y = up (head), negative Y = down (feet).
    This matches the training data from datapreprocess.py which flips the
    3D Y-axis via y = -Y_3d so the skeleton stands upright.
    
    In image/pixel space, Y increases downward (top=0, bottom=H).
    After centering at the hip, the head has negative Y (above hip)
    and feet have positive Y (below hip). We negate Y to flip this
    so the convention matches the training data.
    """
    pts = np.array(kpts, dtype=np.float32)
    
    # Calculate hip midpoint (indices 7 and 8)
    hip_mid = (pts[7] + pts[8]) / 2.0
    
    # Calculate shoulder midpoint (indices 1 and 2)
    shoulder_mid = (pts[1] + pts[2]) / 2.0
    
    # Shift so hip is at (0, 0)
    pts = pts - hip_mid
    
    # Torso length (shoulder to hip)
    # Since hip is now (0,0), the vector from hip to shoulder is just the new shoulder_mid
    shoulder_mid_shifted = shoulder_mid - hip_mid
    torso_length = np.linalg.norm(shoulder_mid_shifted)
    
    if torso_length < 1e-6:
        torso_length = 1.0
        
    pts = pts / torso_length
    
    # Flip Y-axis: image Y goes downward, but training data uses Y-up convention
    pts[:, 1] = -pts[:, 1]
    
    normalized = []
    for pt in pts:
        if np.isnan(pt).any() or np.isinf(pt).any():
            normalized.append([0.0, 0.0])
        else:
            normalized.append([round(float(pt[0]), 4), round(float(pt[1]), 4)])
            
    return normalized

def adjust_keypoints_gui(image, keypoints):
    """Interactive OpenCV window to drag and adjust keypoints."""
    window_name = "Adjust Keypoints - Drag points, Press ENTER to save"
    
    state = {'dragging_idx': -1, 'pts': keypoints.copy()}

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, pt in enumerate(state['pts']):
                if pt == [0.0, 0.0]: continue
                if np.sqrt((pt[0] - x)**2 + (pt[1] - y)**2) < 10:
                    state['dragging_idx'] = i
                    break
        elif event == cv2.EVENT_MOUSEMOVE:
            if state['dragging_idx'] != -1:
                state['pts'][state['dragging_idx']] = [float(x), float(y)]
        elif event == cv2.EVENT_LBUTTONUP:
            state['dragging_idx'] = -1

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- GUI OPENED ---")
    print("Drag the red dots to finely adjust keypoints if needed.")
    print("Press the ENTER key in the window when finished.")

    while True:
        display_img = image.copy()
        pts = state['pts']

        for (start, end) in NEW_SKELETON:
            if pts[start] != [0.0, 0.0] and pts[end] != [0.0, 0.0]:
                cv2.line(display_img, (int(pts[start][0]), int(pts[start][1])), 
                         (int(pts[end][0]), int(pts[end][1])), (255, 0, 0), 2)

        for i, pt in enumerate(pts):
            if pt != [0.0, 0.0]:
                cv2.circle(display_img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
                cv2.putText(display_img, str(i), (int(pt[0]) + 5, int(pt[1]) - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow(window_name, display_img)
        if cv2.waitKey(1) & 0xFF == 13: 
            break

    cv2.destroyAllWindows()
    return state['pts']


if __name__ == "__main__":
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image ({image_path}).")
        exit()

    print("Running detection...")
    det_results = det_model(image, conf=0.25, iou=0.45, classes=[0], device=device, verbose=False)
    
    shifted_kpts = []
    bbox = []

    # Check if a person was detected
    if not det_results or len(det_results[0].boxes) == 0:
        print("No person detected by YOLO.")
        manual_kpts = manual_annotation_gui(image)
        
        # Calculate a bounding box based on the user's manual points (with 50px padding)
        pts_array = np.array(manual_kpts)
        x_min, y_min = np.min(pts_array, axis=0)
        x_max, y_max = np.max(pts_array, axis=0)
        
        padding = 200
        h, w = image.shape[:2]
        x1 = max(0, int(x_min) - padding)
        y1 = max(0, int(y_min) - padding)
        x2 = min(w, int(x_max) + padding)
        y2 = min(h, int(y_max) + padding)
        bbox = [x1, y1, x2, y2]
        
        # Shift the manual points so they match the cropped image coordinates
        for pt in manual_kpts:
            shifted_kpts.append([pt[0] - x1, pt[1] - y1])

    else:
        # YOLO Detection successful
        bbox = det_results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, bbox)

        print("Running pose estimation...")
        pose_results = pose_model(image, conf=0.25, iou=0.45, classes=[0], device=device, verbose=False)
        
        if not pose_results or pose_results[0].keypoints is None:
            print("Poses detected, but keypoints missing. Exiting.")
            exit()

        kpts_xy = pose_results[0].keypoints.xy[0].cpu().numpy()
        kpts_conf = pose_results[0].keypoints.conf[0].cpu().numpy()
        shifted_kpts = consolidate_and_shift_keypoints(kpts_xy, kpts_conf, bbox)

    # Crop the image (works for both YOLO and Manual bounding boxes)
    cropped_img = image[y1:y2, x1:x2].copy()
    crop_h, crop_w = cropped_img.shape[:2]
    
    # Launch GUI for final adjustment (allows fixing YOLO mistakes or manual misclicks)
    adjusted_kpts = adjust_keypoints_gui(cropped_img, shifted_kpts)

    # Normalize adjusted keypoints
    normalized_kpts = normalize_keypoints(adjusted_kpts, crop_w, crop_h)

    # Save results
    cv2.imwrite(save_img_path, cropped_img)
    
    keypoint_data = {
        "format": "[x_normalized, y_normalized]",
        "order": KEYPOINT_NAMES,
        "keypoints": normalized_kpts
    }
    
    with open(save_kpts_path, 'w') as f:
        json.dump(keypoint_data, f, indent=4)

    print(f"\nSuccess! Cropped image saved to: {save_img_path}")
    print(f"Normalized keypoints saved to: {save_kpts_path}")