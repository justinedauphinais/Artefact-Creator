import cv2
import numpy as np

def remove_artifacts(input_video_path, output_video_path):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error opening video file: {input_video_path}")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float `width`
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`
    fps    = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Processing {frame_count} frames...")

    # Read and process frames
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply denoising algorithm
        # You can adjust the parameters h, hForColorComponents, templateWindowSize, and searchWindowSize
        denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

        # Write the frame to the output video file
        out.write(denoised_frame)

        frame_number += 1
        if frame_number % 10 == 0:
            print(f"Processed {frame_number}/{frame_count} frames")

    # Release everything if job is finished
    cap.release()
    out.release()
    print(f"Artifact removal complete. Processed video saved to {output_video_path}")

if __name__ == "__main__":
    input_video = "artefact_video.mp4"    # Replace with your input video file path
    output_video = "output_video.mp4"  # Replace with your desired output video file path
    remove_artifacts(input_video, output_video)
