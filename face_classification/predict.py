import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path

if __name__ == '__main__':
    # 1. Load the pre-trained detection model (standard YOLOv8n to find people)
    detector = YOLO("yolov8n.pt")
    
    # 2. Load your custom trained classification model weights
    classifier_path = "/home/videk/Desktop/faks/rins/face_classification/runs/classify/train/weights/best.pt"
    classifier = YOLO(classifier_path)
    
    # 3. Define the path to an image you want to test
    image_path = "/home/videk/Desktop/faks/rins/face_classification/test.jpg"
    
    # Read the image using OpenCV
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        exit()
        
    # 4. Run detection to find all people in the image (classes=0 filters for 'person')
    print("Running person detection...")
    det_results = detector.predict(source=img, classes=[0])
        
    print(f"\n--- Prediction Results ---")
    
    # 5. Process each detected person
    for i, box in enumerate(det_results[0].boxes):
        # Get bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Crop the detected person from the image
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
            
        # 6. Run the classification on the cropped image
        cls_results = classifier.predict(source=crop, verbose=False)
        
        # Extract classification result
        top_class_id = cls_results[0].probs.top1
        top_class_name = cls_results[0].names[top_class_id]
        confidence = cls_results[0].probs.top1conf.item()
        
        print(f"Detection {i+1}: Found a person!")
        print(f"  -> Classified as: {top_class_name}")
        print(f"  -> Confidence: {confidence * 100:.2f}%\n")
        
        # Optional: Draw a box and label on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{top_class_name} ({confidence*100:.1f}%)"
        cv2.putText(img, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    # Save the final annotated image
    output_path = "/home/videk/Desktop/faks/rins/face_classification/test_annotated.jpg"
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to: {output_path}")
    
    # Display the image on screen
    # Convert BGR (OpenCV) to RGB (Matplotlib) so the colors look right
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide the graph axes
    plt.show()
