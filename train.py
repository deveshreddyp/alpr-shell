from ultralytics import YOLO
import os
import torch

def train_model():
    """
    This function trains a new YOLOv8 model on our license plate dataset.
    """
    print("Loading pre-trained YOLOv8 'nano' model...")
    # Load a pre-trained model. 'yolov8n.pt' is the smallest, fastest one.
    # 'n' stands for 'nano'. We use a pre-trained model so it
    # already knows basic shapes, lines, and objects.
    model = YOLO('yolov8n.pt')

    # Check if a CUDA-capable GPU is available and set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")

    print("Starting training... This will take a while.")
    
    # Train the model
    # 'data' points to our .yaml file describing the dataset.
    # 'epochs' is how many times to go through the data. 50 is a good start.
    # 'imgsz' is the image size to train at. 640 is standard.
    # 'device' will automatically use your GPU (cuda) if you have one.
    results = model.train(data='dataset.yaml', epochs=50, imgsz=640, device=device)

    print("\n----------------------------------")
    print("Training complete!")
    
    # Find the path to the best model
    # The 'results.save_dir' attribute holds the directory (e.g., 'runs/detect/train')
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    
    print(f"Results saved to: {results.save_dir}")
    print(f"Your new model is at: {best_model_path}")
    print("----------------------------------")

if __name__ == '__main__':
    # This block ensures the training only runs when you
    # execute this script directly (e.g., `python train.py`)
    train_model()