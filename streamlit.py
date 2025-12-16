# Streamlit UI for image upload

import streamlit as st
from PIL import Image

import torch
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import os


# Load model (run once)
@st.cache_resource
def load_model():
	weights = EfficientNet_B0_Weights.IMAGENET1K_V1
	model = efficientnet_b0(weights=weights)
	in_features = model.classifier[1].in_features
	model.classifier[1] = torch.nn.Linear(in_features, 2)
	model_path = os.path.join(os.path.dirname(__file__), 'efficientnet_b0_stroke_best.pth')
	model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
	model.eval()
	return model

model = load_model()

st.title("Stroke Detection from Facial Images")
st.write("Upload a facial image to detect stroke.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
	image = Image.open(uploaded_file).convert('RGB')
	st.image(image, caption="Uploaded Image", use_container_width=True)

	# Preprocess image

	weights = EfficientNet_B0_Weights.IMAGENET1K_V1
	preprocess = weights.transforms()
	input_tensor = preprocess(image).unsqueeze(0)

	# Inference
	with torch.no_grad():
		outputs = model(input_tensor)
		_, preds = torch.max(outputs, 1)
		prob = torch.nn.functional.softmax(outputs, dim=1)[0][preds.item()].item()

	# Map prediction to label
	class_names = ['No Stroke', 'Stroke']
	result = class_names[preds.item()]
	if(result == 'Stroke' and prob > 0.4):
		result = 'Stroke detected.'
	elif(result == 'Stroke'):
		result = 'Uncertain - Further Analysis Needed'
	elif(result == 'No Stroke' and prob < 0.6):
		result = 'Uncertain - Further Analysis Needed'
	st.write(f"**Prediction:** {result}")
	st.write("Please consult a medical professional for an accurate diagnosis.")