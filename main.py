import cv2
from ultralytics import YOLO

# Load the pretrained YOLOv8 model
model = YOLO('yolov8n.pt')  # Replace with your desired YOLOv8 model (e.g., yolov8s.pt)

# Open the video file
cap = cv2.VideoCapture('road.mp4')  

# Known real-world dimensions
real_width_car = 2  # Average width of a car in meters (you can adjust this)

# Camera's focal length (this is just an estimate, you can calibrate your camera for accuracy)
focal_length = 800  # Focal length in pixels (adjust this based on your camera specs)

while True:
    # Read each frame from the video
    ret, frame = cap.read()
    if not ret:
        break  # Break if the video ends

    # Perform object detection
    results = model(frame)
    
    # Annotate the frame with the detected objects
    annotated_frame = results[0].plot()  # Use plot() method to draw boxes
    

    for result in results[0].boxes:
      if result.cls == 2:  # Filter for car (class ID 2 for cars in COCO)
        # Extract coordinates as numpy arrays (or lists)
        xyxy = result.xyxy[0].cpu().numpy()  # Convert to numpy array
        x1, y1, x2, y2 = map(int, xyxy)  # Convert each coordinate to int

        object_width_pixels = x2 - x1  # Width of the detected object in pixels

        # Estimate the distance based on the bounding box width
        if object_width_pixels > 0:  # Avoid division by zero
            distance = (real_width_car * focal_length) / object_width_pixels

            # Display the distance on the frame if the car is within 50 meters
            if distance <= 50:
                cv2.putText(annotated_frame, f"Distance: {distance:.2f}m", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw bounding box for car
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

   
    # Show the output frame
    cv2.imshow("Object Detection", annotated_frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
