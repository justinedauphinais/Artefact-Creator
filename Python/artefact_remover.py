import cv2
import numpy as np

def remove_artifacts(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

        out.write(denoised_frame)

        frame_number += 1
        if frame_number % 10 == 0:
            print(f"Processed {frame_number}/{frame_count} frames")

    cap.release()
    out.release()
    print(f"Artifact removal complete. Processed video saved to {output_video_path}")

if __name__ == "__main__":
    input_video = "artifact_video.mp4"
    output_video = "output_video.mp4"
    remove_artifacts(input_video, output_video)
