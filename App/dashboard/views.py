import io
import json
from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import Prediction
from django.db.models import Count

from django.core.files.base import ContentFile
import os
import numpy as np
from django.conf import settings

# Load class names from JSON file
CLASS_NAMES_PATH = os.path.join(settings.BASE_DIR, 'model/class_names.json')
try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
except Exception:
    class_names = []

# Lazy model loading - only load when needed
_model = None

def get_model():
    global _model
    if _model is None:
        try:
            from tensorflow.keras.models import load_model
            MODEL_PATH = os.path.join(settings.BASE_DIR, 'model/ensemble_model.h5')
            _model = load_model(MODEL_PATH)
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            return None
    return _model

def preprocess_image(image_file):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    # Wrap the uploaded file in a BytesIO stream
    img_bytes = io.BytesIO(image_file.read())
    
    # Load the image from the stream
    img = load_img(img_bytes, target_size=(224, 224))
    img = img_to_array(img) / 255.0
    
    # Reset stream pointer for safety if you need to read it again
    image_file.seek(0) 
    
    return np.expand_dims(img, axis=0)

@login_required
def dashboard_view(request):
    return render(request, 'dashboard/dashboard.html')

@login_required
def predict_view(request):
    result = None
    image_url = None  # Initialize to None
    error_msg = None
    
    if request.method == 'POST':
        image_file = request.FILES.get('image_file')
        
        if image_file:
            model = get_model()
            if model is None:
                error_msg = "ML model is not available. Please upload ensemble_model.h5 to the model/ directory."
            else:
                # 1. Prediction Logic
                processed_img = preprocess_image(image_file)
                prediction_probs = model.predict(processed_img)
                predicted_idx = np.argmax(prediction_probs)
                
                # Clean up the class name for display
                raw_result = class_names[predicted_idx] if class_names else "Unknown"
                result = raw_result.replace("_", " ") 

                # 2. Save to Database
                # Django handles saving the file to MEDIA_ROOT/inputs/ here
                new_pred = Prediction(
                    user=request.user,
                    input_file=image_file,
                    predicted_class=raw_result,
                    confidence=float(np.max(prediction_probs))
                )
                new_pred.save()
                
                # 3. Capture the URL for the template
                image_url = new_pred.input_file.url

    # Ensure both result AND image_url are in the context
    context = {
        'result': result,
        'image_url': image_url,
        'error_msg': error_msg,
    }
    return render(request, 'dashboard/predict.html', context)

@login_required
def history_view(request):
    qs = Prediction.objects.filter(user=request.user)

    # Count per class
    class_counts = qs.values('predicted_class').annotate(count=Count('id'))

    labels = [item['predicted_class'] for item in class_counts]
    data = [item['count'] for item in class_counts]

    return render(request, 'dashboard/history.html', {'labels': labels,'data': data,})

@login_required
def profile_page(request):
    profile = request.user.profile
    return render(request, 'dashboard/profile.html', {'profile': profile})

@login_required
def my_predictions(request):
    predictions = Prediction.objects.filter(user=request.user)
    return render(request, 'dashboard/my_predictions.html', {'predictions': predictions})

