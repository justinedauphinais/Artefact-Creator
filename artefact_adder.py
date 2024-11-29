import cv2
import numpy as np

def add_artifacts(input_video_path, output_video_path, noise_level=10, compression_quality=20):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Width of the frames
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frames
    fps    = cap.get(cv2.CAP_PROP_FPS)                # Frames per second

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames to add artifacts...")

    # Read and process frames
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Add Gaussian noise
        noise = np.zeros_like(frame, dtype=np.uint8)
        cv2.randn(noise, (0, 0, 0), (noise_level, noise_level, noise_level))
        noisy_frame = cv2.add(frame, noise)

        # Introduce compression artifacts by encoding and decoding the frame with low quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality]
        result, encimg = cv2.imencode('.jpg', noisy_frame, encode_param)
        compressed_frame = cv2.imdecode(encimg, cv2.IMREAD_COLOR)

        # Write the frame to the output video file
        out.write(compressed_frame)

        frame_number += 1
        if frame_number % 10 == 0:
            print(f"Processed {frame_number}/{frame_count} frames")

    # Release everything if job is finished
    cap.release()
    out.release()
    print(f"Artifact addition complete. Processed video saved to {output_video_path}")

if __name__ == "__main__":
    input_video = "input_video.mp4"    # Replace with your input video file path
    output_video = "artifact_video.mp4"  # Replace with your desired output video file path
    add_artifacts(input_video, output_video)
