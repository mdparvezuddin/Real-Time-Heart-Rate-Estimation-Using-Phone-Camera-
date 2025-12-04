import cv2

# Use the exact same path you've been using
video_path = r"C:\Users\shiva\Videos\VID20251104144712~2.mp4"

print(f"Attempting to open video at: {video_path}")

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print("\n--- ERROR ---")
    print("cv2.VideoCapture() could not open the video file.")
    print("This is the root of the problem.")
else:
    print("\nVideo file opened successfully!")
    
    # Get the FPS property
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Reported FPS: {fps}")
    
    if fps == 0.0:
        print("\n--- ROOT CAUSE CONFIRMED ---")
        print("The FPS is 0.0. This is a deep codec or environment issue.")
    else:
        print("\n--- SUCCESS ---")
        print(f"This script read the FPS as {fps}.")
        print("This means your OpenCV install is working, but the project's code is failing.")

# Release the capture object
cap.release()