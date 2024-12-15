import streamlit as st
from utils.utils import apply_transforms, add_noise, get_device, load_model, mse_loss
import matplotlib.pyplot as plt

# Page config
st.set_page_config(
    page_title="Image Denoising Demo",
    page_icon="🖼️",
)

device = get_device()
model = load_model(device)

st.title('Denoising Autoencoder')
# slider widget of values between 0 and 1
noise = st.slider('Select noise factor', min_value=0.0, max_value=1.0, step=0.01, value=0.5)

# select an image from 0 to 10
image = st.selectbox('Select an image', list(range(10)))

st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### Original Image")
    st.image(f'data/{image}.png', width=200)

with col2:
    st.markdown("#### Noisy Image")
    img = plt.imread(f'data/{image}.png')
    img = apply_transforms(img).unsqueeze(0).to(device)
    noisy_img = add_noise(img, noise_factor=noise, device=device)
    st.image(noisy_img.squeeze(0).permute(1,2,0).cpu().numpy(), width=200)

with col3:
    st.markdown("#### Denoised Image")
    denoised_img = model(noisy_img)
    st.image(denoised_img.squeeze(0).permute(1,2,0).cpu().detach().numpy(), width=200)

st.divider()

st.metric(label="MSE Loss", value=mse_loss(denoised_img, img))
