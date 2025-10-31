import torch
import torch.nn as nn
import timm
from PIL import Image
import torchvision.transforms as transforms
import os

def load_model_fixed(model_name, num_classes=3):
    try:
        if model_name == 'mobilenetv3_large_100':
            model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=num_classes)
        elif model_name == 'resnet50':
            model = timm.create_model('resnet50', pretrained=False, num_classes=num_classes)
        else:
            print(f"Unknown model: {model_name}")
            return None
            
        checkpoint_path = f"artifacts/models/best_{model_name}.pth"
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        print(f"✓ {model_name} loaded successfully")
        return model
        
    except Exception as e:
        print(f"✗ Model loading error {model_name}: {e}")
        return None

def predict_image(model, image_path, class_names, transform):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
            
        return class_names[predicted_class], confidence
    except Exception as e:
        return f"Error: {e}", 0.0

def main():
    print("TESTING MODEL PREDICTIONS")
    
    class_names = ['minivan', 'sedan', 'wagon']
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
         transforms.Normalize(self.DATA_MEAN, self.DATA_STD)
    ])
    
    test_images = [
        'data/raw/minivan/kisspng-2016-nissan-quest-car-2013-nissan-quest-2011-nissa-5ae78661c06196.926277781525122657788.jpg',
        'data/raw/sedan/d2e5f48218cc6572a9388998fa185769.jpg', 
        'data/raw/wagon/0d1901c8711742e1cb1769c805ad9e8b.jpg'
    ]
    
    print("Test Images:")
    for img in test_images:
        print(f"  {os.path.basename(img)}")
    print(f"Classes: {class_names}")
    
    models = {}
    for model_name in ['resnet50', 'mobilenetv3_large_100']:
        print(f"\nLoading: {model_name}")
        model = load_model_fixed(model_name, num_classes=len(class_names))
        if model:
            models[model_name] = model
    
    if not models:
        print("Couldn't load any models")
        return
    
    print("RESULTS:")
    
    for model_name, model in models.items():
        print(f"\n{model_name.upper()} PREDICTIONS:")
        
        for img_path in test_images:
            if os.path.exists(img_path):
                predicted_class, confidence = predict_image(model, img_path, class_names, transform)
                print(f"  {os.path.basename(img_path)}:")
                print(f"    Predicted: {predicted_class}")
                print(f"    Confidence: {confidence:.2%}")
            else:
                print(f"  {img_path}: File not found")
        
if __name__ == "__main__":
    main()