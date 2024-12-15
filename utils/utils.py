import torch
from torchvision import transforms
from torch import nn
import streamlit as st

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class AutoencoderV2(nn.Module):
    def __init__(self):
        super(AutoencoderV2, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(x.shape)
        x = self.decoder(x)
        return x

@st.cache_resource
def get_device():
    if torch.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'
    
def apply_transforms(img):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
    return transform(img)

def add_noise(img, noise_factor=0.7, device='cpu'):
    noisy_img = img + noise_factor * torch.randn(*img.shape).to(device)
    noisy_img = torch.clip(noisy_img, 0., 1.)
    return noisy_img

@st.cache_resource
def load_model(device):
    model = AutoencoderV2().to(device)
    try:
        # model.load_state_dict(torch.load('models/autoencoder_0.pth'))
        # Load state dict with proper device mapping
        state_dict = torch.load('models/autoencoder_0.pth', 
                              map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        st.error('Model not found. Please train the model first.')
    model.eval()
    return model


def mse_loss(x, y):
    return torch.nn.functional.mse_loss(x, y)
