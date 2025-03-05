from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.uploadedfile import InMemoryUploadedFile
import numpy as np
from PIL import Image
import io
from .model_loader import load_model

# Load model (ensure this is done once at startup)
model = load_model()

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        try:
            # Validate input
            image_file = request.FILES.get('image')
            if not image_file or not isinstance(image_file, InMemoryUploadedFile):
                return JsonResponse({'error': 'Invalid or missing image file'}, status=400)

            # Process image
            image = Image.open(image_file).convert('RGB')
            image = image.resize((224, 224))  # Resize to match model input size
            image = np.array(image) / 255.0  # Normalize
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Make prediction
            predictions = model.predict(image)
            predicted_class = np.argmax(predictions, axis=1)[0]

            # Define disease names and remedies
            class_labels = [
                'Class0', 'Class1', 'Class2', 'Class3', 'Class4', 'Class5',
                'Class6', 'Class7', 'Class8', 'Class9', 'Class10', 'Class11',
                'Class12', 'Class13', 'Class14', 'Class15', 'Class16', 'Class17',
                'Class18', 'Class19', 'Class20', 'Class21', 'Class22', 'Class23',
                'Class24', 'Class25', 'Class26', 'Class27', 'Class28', 'Class29',
                'Class30', 'Class31', 'Class32', 'Class33', 'Class34', 'Class35',
                'Class36', 'Class37', 'Class38', 'Class39', 'Class40', 'Class41'
            ]

            # Define disease names corresponding to class labels
            disease_names = [
                'Healthy', 'Powdery Mildew', 'Leaf Spot', 'Blight', 'Rust', 'Mosaic Virus',
                'Anthracnose', 'Canker', 'Scab', 'Wilt', 'Root Rot', 'Leaf Curl',
                'Downy Mildew', 'Gray Mold', 'Black Spot', 'Yellowing', 'Bacterial Spot',
                'Fusarium Wilt', 'Verticillium Wilt', 'Leaf Blight', 'Leaf Scorch',
                'Leaf Rust', 'Leaf Miner', 'Leafhopper Damage', 'Spider Mite Damage',
                'Aphid Damage', 'Whitefly Damage', 'Thrip Damage', 'Mealybug Damage',
                'Scale Damage', 'Nematode Damage', 'Virus', 'Fungus', 'Bacteria',
                'Nutrient Deficiency', 'Overwatering', 'Underwatering', 'Heat Stress',
                'Cold Stress', 'Chemical Burn', 'Physical Damage', 'Unknown Disease'
            ]

            # Define remedies corresponding to class labels
            remedies = [
                'No action needed. The plant is healthy.',
                'Apply fungicide and ensure proper air circulation.',
                'Remove affected leaves and apply fungicide.',
                'Apply fungicide and avoid overhead watering.',
                'Apply fungicide and remove infected leaves.',
                'Remove infected plants and control aphids.',
                'Apply fungicide and remove infected plant parts.',
                'Prune affected branches and apply fungicide.',
                'Apply fungicide and ensure proper spacing.',
                'Improve drainage and avoid overwatering.',
                'Improve soil drainage and apply fungicide.',
                'Control whiteflies and apply insecticide.',
                'Apply fungicide and avoid overhead watering.',
                'Remove affected plant parts and apply fungicide.',
                'Apply fungicide and ensure proper air circulation.',
                'Check soil pH and nutrient levels.',
                'Apply copper-based fungicide.',
                'Remove infected plants and improve soil drainage.',
                'Remove infected plants and improve soil drainage.',
                'Apply fungicide and remove affected leaves.',
                'Ensure adequate watering and shade.',
                'Apply fungicide and remove infected leaves.',
                'Apply insecticide and remove affected leaves.',
                'Apply insecticide and remove affected leaves.',
                'Apply miticide and increase humidity.',
                'Apply insecticide and remove affected leaves.',
                'Apply insecticide and use yellow sticky traps.',
                'Apply insecticide and remove affected leaves.',
                'Apply insecticide and remove affected leaves.',
                'Apply insecticide and remove affected leaves.',
                'Remove infected plants and control nematodes.',
                'Remove infected plants and control vectors.',
                'Apply fungicide and improve air circulation.',
                'Apply bactericide and remove infected parts.',
                'Adjust fertilization and soil pH.',
                'Reduce watering frequency and improve drainage.',
                'Increase watering frequency and ensure proper drainage.',
                'Provide shade and increase watering.',
                'Protect plants from frost and cold winds.',
                'Flush soil with water and avoid over-fertilization.',
                'Remove damaged parts and protect plants.',
                'Consult an expert for diagnosis and treatment.'
            ]
            # Validate predicted class index
            if predicted_class >= len(class_labels):
                return JsonResponse({'error': 'Invalid prediction'}, status=500)

            # Return result
            return JsonResponse({
                'prediction': class_labels[predicted_class],
                'disease_name': disease_names[predicted_class],
                'remedy': remedies[predicted_class]
            })
        except Exception as e:
            return JsonResponse({'error': f'Internal server error: {str(e)}'}, status=500)
    return JsonResponse({'error': 'Invalid request method'}, status=400)

from django.shortcuts import render

def home(request):
    return render(request, 'index.html')






