import cv2
import numpy as np
from ultralytics import YOLO
import time
import os
import torch

def check_system_info():
    """
    Check and display system information (GPU/CPU)
    """
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 40)
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {'✅ YES' if cuda_available else '❌ NO'}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"GPU Count: {gpu_count}")
        print(f"Current GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"Device Index: {current_device}")
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
            gpu_memory_reserved = torch.cuda.memory_reserved(current_device) / 1024**3
            print(f"GPU Memory Allocated: {gpu_memory_allocated:.2f} GB")
            print(f"GPU Memory Reserved: {gpu_memory_reserved:.2f} GB")
    else:
        print("Using CPU for inference")
        print("💡 For faster detection, install CUDA-compatible PyTorch")
    
    print("=" * 40)
    return cuda_available

def detect_soccer_players_from_camera():
    """
    Real-time soccer player detection using trained YOLOv11/YOLOv8 model
    """
    # 🔧 CHANGE THIS: Update path to your trained model
    model_path = 'C:\\Users\\DEATHSEC\\Desktop\\codes\\Soccer Player Detection\\best.pt'
    
    # Verify model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("Please download your best.pt from Google Drive and update the path!")
        return
    
    try:
        # Check system information first
        cuda_available = check_system_info()
        
        # Load the trained YOLO model
        print("🧠 Loading soccer player detection model...")
        yolo_model = YOLO(model_path)
        
        # Set device for model (GPU if available, else CPU)
        device = 'cuda:0' if cuda_available else 'cpu'
        yolo_model.to(device)
        print(f"✅ Model loaded on: {device.upper()}")
        
        # Warm up the model for better performance
        print("🔥 Warming up model...")
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _ = yolo_model(dummy_frame, verbose=False)
        print("✅ Model warmed up!")
        
        # Open the default camera (usually the first one)
        video_capture = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not video_capture.isOpened():
            print("❌ Error: Could not open camera.")
            return
        
        # Set camera properties for better performance
        video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        video_capture.set(cv2.CAP_PROP_FPS, 30)
        
        print("🚀 Starting soccer player detection...")
        print("Press 'q' to quit, 's' to save screenshot, 'i' for system info")
        
        # Performance tracking
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()
            if not ret:
                print("❌ Error: Failed to capture frame from the camera.")
                break
            
            frame_count += 1
            
            # Measure inference time
            inference_start = time.time()
            
            # Detect soccer players in the frame
            results = yolo_model(frame, conf=0.4, iou=0.5, verbose=False)  # Added verbose=False
            
            inference_time = time.time() - inference_start
            
            # Process detections
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    classes = result.names
                    cls = result.boxes.cls
                    conf = result.boxes.conf
                    detections = result.boxes.xyxy
                    
                    # Draw bounding boxes and labels
                    for pos, detection in enumerate(detections):
                        confidence = conf[pos].item()
                        
                        if confidence >= 0.3:  # Lower threshold for soccer players
                            # Get coordinates
                            xmin, ymin, xmax, ymax = detection
                            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                            
                            # Get class name and create label
                            class_id = int(cls[pos])
                            class_name = classes[class_id]
                            label = f"{class_name}: {confidence:.2f}"
                            
                            # Color coding (you can customize these)
                            if class_name.lower() in ['person', 'player', 'soccer_player']:
                                color = (0, 255, 0)  # Green for soccer players
                            else:
                                color = (255, 0, 0)  # Blue for other detections
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                            
                            # Calculate label background size
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            
                            # Draw label background
                            cv2.rectangle(frame, (xmin, ymin - label_size[1] - 10), 
                                        (xmin + label_size[0], ymin), color, -1)
                            
                            # Draw label text
                            cv2.putText(frame, label, (xmin, ymin - 5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Calculate and display FPS
            if frame_count % 30 == 0:  # Update every 30 frames
                end_time = time.time()
                fps = 30 / (end_time - start_time)
                start_time = end_time
                
            # Add comprehensive info to frame
            detection_count = len(result.boxes) if result.boxes is not None else 0
            device_info = "GPU" if cuda_available else "CPU"
            
            # Main info line
            info_text = f"FPS: {fps:.1f} | Players: {detection_count} | Device: {device_info}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Inference time info
            inference_info = f"Inference: {inference_time*1000:.1f}ms"
            cv2.putText(frame, inference_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # GPU memory info (if available)
            if cuda_available and frame_count % 60 == 0:  # Update every 60 frames
                gpu_memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
                memory_info = f"GPU Memory: {gpu_memory_used:.0f}MB"
                cv2.putText(frame, memory_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save, 'i' for info", (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the resulting frame
            cv2.imshow('Soccer Player Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("🚪 Quitting soccer player detection...")
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"soccer_detection_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('i'):
                # Display system info in console
                print("\n" + "="*50)
                check_system_info()
                print("="*50)
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
    
    finally:
        # Release the camera and close all OpenCV windows
        video_capture.release()
        cv2.destroyAllWindows()
        print("✅ Camera released and windows closed.")

def detect_soccer_players_from_image(image_path):
    """
    Detect soccer players in a single image
    """
    # 🔧 CHANGE THIS: Update path to your trained model  
    model_path = 'C:\\Users\\DEATHSEC\\Desktop\\codes\\Soccer Player Detection\\best.pt'  # Update this path
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return
        
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at {image_path}")
        return
    
    try:
        # Check system information
        cuda_available = check_system_info()
        
        # Load model and image
        yolo_model = YOLO(model_path)
        
        # Set device
        device = 'cuda:0' if cuda_available else 'cpu'
        yolo_model.to(device)
        print(f"✅ Model loaded on: {device.upper()}")
        
        image = cv2.imread(image_path)
        
        if image is None:
            print("❌ Error: Could not load image.")
            return
        
        print(f"🔍 Detecting soccer players in: {image_path}")
        
        # Measure inference time
        inference_start = time.time()
        
        # Run detection
        results = yolo_model(image, conf=0.3, verbose=False)
        
        inference_time = time.time() - inference_start
        print(f"⚡ Inference time: {inference_time*1000:.1f}ms on {device.upper()}")
        
        # Process and display results
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                classes = result.names
                cls = result.boxes.cls
                conf = result.boxes.conf
                detections = result.boxes.xyxy
                
                print(f"✅ Found {len(detections)} detections!")
                
                # Draw results
                for pos, detection in enumerate(detections):
                    confidence = conf[pos].item()
                    xmin, ymin, xmax, ymax = map(int, detection)
                    
                    class_id = int(cls[pos])
                    class_name = classes[class_id]
                    label = f"{class_name}: {confidence:.2f}"
                    
                    # Draw bounding box and label
                    color = (0, 255, 0)
                    cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
                    cv2.putText(image, label, (xmin, ymin - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display result
        cv2.imshow('Soccer Player Detection - Image', image)
        print("Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        # Save result
        output_path = image_path.replace('.', '_detected.')
        cv2.imwrite(output_path, image)
        print(f"💾 Result saved: {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """
    Main function - choose detection mode
    """
    print("⚽ Soccer Player Detection System")
    print("=" * 40)
    
    # Show system info at startup
    check_system_info()
    
    print("\n🎯 DETECTION MODES:")
    print("1. Real-time camera detection")
    print("2. Single image detection") 
    print("3. System benchmark test")
    print("4. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            detect_soccer_players_from_camera()
            break
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            detect_soccer_players_from_image(image_path)
            break
        elif choice == '3':
            run_benchmark_test()
            break
        elif choice == '4':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

def run_benchmark_test():
    """
    Run a benchmark test to compare GPU vs CPU performance
    """
    model_path = 'C:\\Users\\DEATHSEC\\Desktop\\codes\\Soccer Player Detection\\best.pt'  # Update this path
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        return
    
    print("🏁 Running Benchmark Test...")
    print("=" * 40)
    
    # Create test image
    test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    
    # Test CPU
    print("Testing CPU performance...")
    model_cpu = YOLO(model_path)
    model_cpu.to('cpu')
    
    cpu_times = []
    for i in range(10):
        start = time.time()
        _ = model_cpu(test_image, verbose=False)
        cpu_times.append(time.time() - start)
    
    avg_cpu_time = np.mean(cpu_times) * 1000
    print(f"CPU Average: {avg_cpu_time:.1f}ms")
    
    # Test GPU (if available)
    if torch.cuda.is_available():
        print("Testing GPU performance...")
        model_gpu = YOLO(model_path)
        model_gpu.to('cuda:0')
        
        # Warm up
        _ = model_gpu(test_image, verbose=False)
        
        gpu_times = []
        for i in range(10):
            start = time.time()
            _ = model_gpu(test_image, verbose=False)
            torch.cuda.synchronize()  # Wait for GPU to finish
            gpu_times.append(time.time() - start)
        
        avg_gpu_time = np.mean(gpu_times) * 1000
        print(f"GPU Average: {avg_gpu_time:.1f}ms")
        
        speedup = avg_cpu_time / avg_gpu_time
        print(f"🚀 GPU Speedup: {speedup:.1f}x faster")
    else:
        print("❌ GPU not available for testing")
    
    print("=" * 40)

if __name__ == "__main__":
    main()
    
    # Option 2: Direct camera detection
    # detect_soccer_players_from_camera()
    
    # Option 3: Direct image detection  
    # detect_soccer_players_from_image('path/to/your/image.jpg')