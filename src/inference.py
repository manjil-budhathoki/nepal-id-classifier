import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os


# Define the classes exactly as they were during training
class_names = ['back', 'front', 'not_id']

# Define the SquarePad again 
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top
        return transforms.functional.pad(image, (p_left, p_top, p_right, p_bottom), fill=0, padding_mode='constant')

# Define the transform 
inference_transform = transforms.Compose([
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- LOAD THE SAVED MODEL ---

def load_model(model_path='best_model.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Re-create the model architecture
    model = models.resnet18(weights=None) # No need to download weights, we have our own
    
    # Modify the last layer to match our 3 classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set to evaluation mode (Important!)
    
    return model, device

# --- PREDICT FUNCTION ---

def predict_image(image_path, model, device):
    if not os.path.exists(image_path):
        return "Error: Image not found"
        
    image = Image.open(image_path)
    
    # Preprocess
    image_tensor = inference_transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Calculate percentages (Softmax)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        
        # Get the winner
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds[0]]
    
    # Print detailed results
    print(f"\nImage: {os.path.basename(image_path)}")
    print(f"Prediction: >>> {predicted_class.upper()} <<<")
    print(f"Confidence:")
    print(f"  Front:  {probabilities[1]:.2f}%")
    print(f"  Back:   {probabilities[0]:.2f}%")
    print(f"  Not ID: {probabilities[2]:.2f}%")
    
    return predicted_class

# --- RUN IT ---

if __name__ == "__main__":
    # Load model once
    model, device = load_model('best_model.pth')

    test_image = "hello.jpeg"
    
    predict_image(test_image, model, device)