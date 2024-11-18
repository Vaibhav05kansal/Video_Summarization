import cv2

def generate_summarized_video(fused_scores, video_path, temp_video_path, threshold=0.05):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    selected_frames = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num < len(fused_scores) and fused_scores[frame_num] > threshold:
            out.write(frame)
            selected_frames.append(frame_num)
        frame_num += 1

    cap.release()
    out.release()
    return selected_frames, fps
