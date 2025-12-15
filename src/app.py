import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2  # Needed for the heatmap colors

# --- 1. CONFIGURATION ---
# IMPORTANT: Must match folder order: 0=back, 1=front, 2=not_id
CLASS_NAMES = ['back', 'front', 'not_id'] 
MODEL_PATH = 'best_model.pth'
CONFIDENCE_THRESHOLD = 60.0

# --- 2. GRAD-CAM CLASS ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_index):
        output = self.model(input_tensor)
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(1, output.size()[-1]).zero_().to(input_tensor.device)
        if class_index is None:
            class_index = torch.argmax(output)
        one_hot_output[0][class_index] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations.detach()
        for i in range(activations.shape[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
            
        heatmap = torch.mean(activations, dim=1).squeeze()
        heatmap = np.maximum(heatmap.cpu(), 0)
        heatmap /= torch.max(heatmap)
        return heatmap.numpy(), output

# --- 3. HELPER FUNCTIONS ---
class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = max(w, h)
        p_left = (max_wh - w) // 2
        p_top = (max_wh - h) // 2
        p_right = max_wh - w - p_left
        p_bottom = max_wh - h - p_top
        return transforms.functional.pad(image, (p_left, p_top, p_right, p_bottom), fill=0, padding_mode='constant')

@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 3)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except FileNotFoundError:
        return None, None

def process_image_for_model(image, device):
    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def overlay_heatmap(heatmap, original_image):
    img_cv = np.array(original_image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    heatmap = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img_cv * 0.6
    return Image.fromarray(np.uint8(superimposed_img)[:, :, ::-1])

# --- 4. STREAMLIT APP ---
st.set_page_config(page_title="ID Explainability", page_icon="🔍")
st.title("🇳🇵 Smart ID Scanner (with AI Vision)")

model, device = load_trained_model()

if model is None:
    st.error("Model file not found. Please train first.")
    st.stop()

grad_cam = GradCAM(model, model.layer4[-1])

uploaded_file = st.file_uploader("Upload ID Card", type=["jpg", "png", "jpeg"])

if uploaded_file:
    input_image = Image.open(uploaded_file).convert('RGB')
    img_tensor = process_image_for_model(input_image, device)
    
    # Run Prediction & Heatmap
    heatmap_raw, output = grad_cam.generate_heatmap(img_tensor, class_index=None)
    
    # Calculate Probabilities
    probs = torch.nn.functional.softmax(output, dim=1)[0] * 100
    max_prob, preds = torch.max(probs, 0)
    pred_idx = preds.item()
    
    # Threshold Logic
    if max_prob < CONFIDENCE_THRESHOLD:
        final_class = "not_id"
        confidence_msg = f"⚠️ Low Confidence ({max_prob:.1f}%)"
    else:
        final_class = CLASS_NAMES[pred_idx]
        confidence_msg = f"✅ Confident ({max_prob:.1f}%)"

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(input_image, use_container_width=True)
    with col2:
        st.subheader("AI Vision")
        overlay_img = overlay_heatmap(heatmap_raw, input_image)
        st.image(overlay_img, caption="Red = Important Features", use_container_width=True)

    # --- MAIN RESULT ---
    st.markdown(f"### Result: **{final_class.upper()}**")
    st.caption(confidence_msg)

    # --- DETAILED CONFIDENCE BARS (NEW SECTION) ---
    st.divider()
    st.subheader("📊 Confidence Breakdown")
    st.write("This shows how confused the model was:")

    # 1. Back Probability
    prob_back = probs[0].item()
    st.write(f"**Back:** {prob_back:.1f}%")
    st.progress(int(prob_back))

    # 2. Front Probability
    prob_front = probs[1].item()
    st.write(f"**Front:** {prob_front:.1f}%")
    st.progress(int(prob_front))

    # 3. Not ID Probability
    prob_not = probs[2].item()
    st.write(f"**Not ID:** {prob_not:.1f}%")
    st.progress(int(prob_not))
    
    # Advice Logic
    st.info("💡 Analysis Note:")
    if prob_back > 20 and prob_front > 20:
        st.write("- The model is confused between **Front and Back**. Check if the Back image has text that looks like a Front header.")
    elif prob_not > 20 and (prob_front > 20 or prob_back > 20):
        st.write("- The model sees the ID but thinks it might just be **random paper (Not ID)**. This usually happens with white backgrounds.")
    else:
        st.write("- The model is very sure of its decision.")