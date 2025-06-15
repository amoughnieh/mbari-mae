import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms

# Quick dataset class
class SpectrogramDataset(Dataset):
    def __init__(self, folder):
        self.images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((128, 128)),  # resize to manageable size
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx])
        return self.transform(img)

# Simple autoencoder
class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Train
dataset = SpectrogramDataset('final_project/data/Test set - Dense spectrograms')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
model = SimpleAutoencoder()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(10):
    for data in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'final_project/marine_autoencoder_test_set.pth')

# Generate and save sample reconstructions
model.eval()
with torch.no_grad():
    sample_batch = next(iter(dataloader))
    reconstructed = model(sample_batch)
    
    # Save original vs reconstructed
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(4):
        # Original
        axes[0, i].imshow(sample_batch[i].squeeze(), cmap='viridis')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[i].squeeze(), cmap='viridis')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('final_project/data/marine_autoencoder_test_set_results.png', dpi=150)
    plt.show() 