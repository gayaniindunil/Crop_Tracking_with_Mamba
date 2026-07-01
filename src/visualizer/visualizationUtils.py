import cv2
import numpy as np
from pathlib import Path

class VisualizationUtils:
    @staticmethod
    def _initialize_writer(output_path, fps, size):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        return cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    @staticmethod
    def generate_position_trajectory_video(track_sequence: list, output_video_path: Path | str, fps: int = 5, display_size: tuple = (512, 512)):
        """Draws a progressive trajectory line (motion trail) over the source images."""
        if not track_sequence: return
        writer = VisualizationUtils._initialize_writer(output_video_path, fps, display_size)
        center_points = []
        try:
            for frame_data in track_sequence:
                frame = cv2.imread(str(frame_data["image_path"]))
                if frame is None: continue
                frame = cv2.resize(frame, display_size)
                x, y, w, h = frame_data["bbox"]
                cx, cy = int(x + (w / 2.0)), int(y + (h / 2.0))
                center_points.append((cx, cy))
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                for i in range(1, len(center_points)):
                    cv2.line(frame, center_points[i-1], center_points[i], (255, 255, 0), 2, cv2.LINE_AA)
                writer.write(frame)
        finally: writer.release()

    @staticmethod
    def generate_bbox_only_black_bg_video(track_sequence: list, output_video_path: Path | str, fps: int = 5, display_size: tuple = (512, 512)):
        """Isolates the bounding box and label over a black background."""
        if not track_sequence: return
        writer = VisualizationUtils._initialize_writer(output_video_path, fps, display_size)
        try:
            for frame_data in track_sequence:
                frame = np.zeros((display_size[1], display_size[0], 3), dtype=np.uint8)
                x, y, w, h = frame_data["bbox"]
                cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
                cv2.putText(frame, f"Track: {frame_data.get('track_id', '?')}", (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(frame)
        finally: writer.release()

    @staticmethod
    def generate_track_video(track_sequence: list, output_video_path: Path | str, fps: int = 5, display_size: tuple = (512, 512)):
        """
        Compiles an .mp4 tracking video for a specific track ID using OpenCV.
        
        Args:
            track_sequence (list): Output list from dataset.get_images_and_boxes_for_track_id()
            output_video_path (Path | str): Absolute destination path for the compiled video.
            fps (int): Frames per second for output video playback rate.
            display_size (tuple): Width and height (W, H) to render the video frames.
        """
        if not track_sequence:
            print("Track sequence is empty. Cannot compile tracking video.")
            return

        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure OpenCV VideoWriter using standard MP4 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, display_size)

        print(f"Compiling tracking video to {output_video_path}...")

        try:
            for frame_data in track_sequence:
                img_path = Path(frame_data["image_path"])
                
                if not img_path.exists():
                    print(f"Warning: Image frame missing, skipping path: {img_path}")
                    continue

                # 1. Load image using standard OpenCV pipelines
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                # 2. Resize frame cleanly to target dimensions 
                frame = cv2.resize(frame, display_size)

                # 3. Extract and cast tracking bounding box components
                x, y, w, h = frame_data["bbox"]
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)

                # 4. Draw bounding box rectangle onto frame (BGR format: Lime Green = (0, 255, 0))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

                # 5. Overlay clear text labels onto the visual canvas
                label_text = f"Track: {frame_data.get('track_id', '?')} Class: {frame_data['category_id']}"
                
                # Background bounding box for label readability
                (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color=(0, 0, 0), thickness=-1)
                
                # Render tracking string metadata
                cv2.putText(
                    frame, 
                    label_text, 
                    (x1, y1 - 5), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=0.5, 
                    color=(255, 255, 255), 
                    thickness=1, 
                    lineType=cv2.LINE_AA
                )

                # 6. Push finalized frame processing structural layers into container matrix
                video_writer.write(frame)

        finally:
            # Enforce clean memory workspace clean-up operations
            video_writer.release()
            print(f"Video pipeline closed successfully. Total written updates stored at target path location.")
