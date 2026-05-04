import streamlit as st
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import random
import os
import matplotlib.pyplot as plt

########################################
# CONFIG
########################################
MODEL_DIR = "./models"
DATA_DIR = "./data"
NUM_CLASSES = 150
device = torch.device("cpu")

########################################
# 클래스 이름 로드
########################################
@st.cache_resource
def load_class_names():
    dataset = datasets.ImageFolder(DATA_DIR)
    return dataset.classes

########################################
# 모델 목록
########################################
def get_model_list():
    return [f.replace(".pth", "") for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]

########################################
# 모델 로드
########################################
@st.cache_resource
def load_model(model_name):
    model = torchvision.models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    model.load_state_dict(
        torch.load(f"{MODEL_DIR}/{model_name}.pth", map_location=device)
    )
    model.eval()
    return model

########################################
# 전처리
########################################
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

########################################
# 예측 함수 (Top-5)
########################################
def predict_top5(model, image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)

        top_probs, top_idxs = torch.topk(probs, 5)

    return top_probs[0].tolist(), top_idxs[0].tolist()

########################################
# 그래프
########################################
def plot_probs(class_names, probs, indices):
    labels = [class_names[i] for i in indices]

    fig, ax = plt.subplots()
    ax.barh(labels, probs)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_title("Top-5 Predictions")

    return fig

########################################
# UI
########################################

def get_representative_image(class_name):
    class_path = os.path.join(DATA_DIR, class_name)

    if not os.path.isdir(class_path):
        return None

    for file in os.listdir(class_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg")):
            try:
                return Image.open(os.path.join(class_path, file))
            except:
                continue

    return None

st.set_page_config(page_title="Image Classifier", layout="centered")

st.title("🧠 Image Classification Demo")
st.write("모델을 선택하고 이미지를 업로드하면 Top-5 예측 결과를 확인할 수 있습니다.")

########################################
# 모델 선택
########################################
model_list = get_model_list()

if not model_list:
    st.warning("❗ 모델이 없습니다. 먼저 학습을 실행하세요.")
    st.stop()

selected_model = st.selectbox("모델 선택", model_list)

model = load_model(selected_model)
class_names = load_class_names()

########################################
# 이미지 업로드
########################################
uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="입력 이미지", use_container_width=True)

    ########################################
    # 예측
    ########################################
    with st.spinner("🔍 예측 중..."):
        probs, indices = predict_top5(model, image)

    ########################################
    # 결과 출력 (Top-5)
    ########################################
    st.subheader("📊 Top-5 예측 결과")

    for i in range(5):
        class_name = class_names[indices[i]]
        prob = probs[i]

        col1, col2 = st.columns([1, 2])

        with col1:
            rep_img = get_representative_image(class_name)
            if rep_img:
                st.image(rep_img, use_container_width=True)
            else:
                st.write("No Image")

        with col2:
            st.markdown(f"""
            **{i+1}. {class_name}**  
            확률: **{prob*100:.2f}%**
            """)

    ########################################
    # 그래프
    ########################################
    fig = plot_probs(class_names, probs, indices)
    st.pyplot(fig)