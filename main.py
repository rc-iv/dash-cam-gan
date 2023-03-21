import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from generator import Generator
from discriminator import Discriminator


# Transform the images to tensors and normalize them
image_transforms = {
    'train': transforms.Compose([
        transforms.Resize((1080, 1920)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((1080, 1920)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}


# Load training images
train_data = ImageFolder('../images/training_data/train', transform=image_transforms['train'])
val_data = ImageFolder('../images/training_data/val', transform=image_transforms['val'])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Create instances of the generator and discriminator networks

nz = 100  # Size of the latent vector
ngf = 64  # Number of generator filters
ndf = 64  # Number of discriminator filters
nc = 3  # Number of image channels (3 for RGB, 1 for grayscale)

generator = Generator(nz, ngf, nc)
discriminator = Discriminator(ndf, nc)

# Define loss function and optimizer
criterion = nn.BCELoss()

lr = 0.0002  # Learning rate
beta1 = 0.5  # Beta1 hyperparameter for Adam optimizer

optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Number of epochs
num_epochs = 50

# Establish convention for real and fake labels during training
real_label = 1
fake_label = 0

# Move models to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# Train the model
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        # Update the discriminator network
        discriminator.zero_grad()
        real_images = images.to(device)
        b_size = real_images.size(0)

        label = torch.full((b_size, 1), real_label, device=device)
        output = discriminator(real_images).view(-1, 1)
        errD_real = criterion(output, label)
        errD_real.backward()

        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = generator(noise)
        label.fill_(fake_label)
        output = discriminator(fake_images.detach()).view(-1, 1)
        errD_fake = criterion(output, label)
        errD_fake.backward()

        errD = errD_real + errD_fake
        optimizerD.step()

        # Update the generator network
        generator.zero_grad()
        label.fill_(real_label)
        output = discriminator(fake_images).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()
        # Print training progress
        if i % 50 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (
            epoch, num_epochs, i, len(train_loader), errD.item(), errG.item()))

    # Save checkpoints (optional)
    if (epoch + 1) % 10 == 0:
        torch.save(generator.state_dict(), f'generator_epoch_{epoch + 1}.pth')
        torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch + 1}.pth')
