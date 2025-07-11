import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

def predict_image(model_path, image_path):
    """
    Make a prediction on a single image using a trained Alzheimer's classification model
    
    Parameters:
    model_path (str): Path to the trained model (.h5 file)
    image_path (str): Path to the MRI image to classify
    
    Returns:
    tuple: (predicted_class, confidence)
    """
    # Define class names
    class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    
    # Load the model
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, 0
        
    # Load and preprocess the image
    try:
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize to [0,1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, 0
    
    # Make prediction
    prediction = model.predict(img_array)
    pred_class_idx = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][pred_class_idx] * 100
    predicted_class = class_names[pred_class_idx]
    
    # Display results
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%", 
              fontsize=14)
    plt.axis('off')
    plt.show()
    
    # Print probabilities for all classes
    print("\nClass probabilities:")
    for i, (cls, prob) in enumerate(zip(class_names, prediction[0])):
        print(f"  {cls}: {prob*100:.2f}%")
    
    return predicted_class, confidence

if __name__ == "__main__":
    # Check if model exists
    model_path = "alzheimers_model_final.h5"
    if not os.path.exists(model_path):
        model_path = "alzheimers_model_improved.h5"
        if not os.path.exists(model_path):
            print("Error: Trained model not found.")
            print("Please train the model first by running 'python new.py'")
            exit(1)
    
    # Ask for image path
    print("Enter the path to the MRI image to classify:")
    image_path = input()
    
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at {image_path}")
        exit(1)
        
    # Make prediction
    predicted_class, confidence = predict_image(model_path, image_path)
    
    if predicted_class:
        print(f"\nFinal prediction: {predicted_class} with {confidence:.2f}% confidence") 