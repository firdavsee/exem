gan.py


import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from model import Generator, Discriminator  # Импортируем модели

# Подготовка данных MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Создание генератора и дискриминатора
generator = Generator()
discriminator = Discriminator()

# Функция потерь и оптимизаторы
criterion = torch.nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002)

epochs = 10  # Количество эпох

# Обучение GAN
for epoch in range(epochs):
    for real_images, _ in tqdm(dataloader):
        real_labels = torch.ones(real_images.size(0), 1)
        fake_labels = torch.zeros(real_images.size(0), 1)

        # Обучение дискриминатора
        optimizer_D.zero_grad()
        outputs = discriminator(real_images)
        loss_real = criterion(outputs, real_labels)

        noise = torch.randn(real_images.size(0), 100)
        fake_images = generator(noise)
        outputs = discriminator(fake_images.detach())
        loss_fake = criterion(outputs, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizer_D.step()

        # Обучение генератора
        optimizer_G.zero_grad()
        outputs = discriminator(fake_images)
        loss_G = criterion(outputs, real_labels)
        loss_G.backward()
        optimizer_G.step()

    print(f"Эпоха [{epoch+1}/{epochs}] | Потери генератора: {loss_G.item()} | Потери дискриминатора: {loss_D.item()}")

    # Сохранение примера изображения
    with torch.no_grad():
        noise = torch.randn(16, 100)
        fake_images = generator(noise)
        fake_images = fake_images.view(16, 28, 28).cpu().numpy()

        fig, axes = plt.subplots(2, 8, figsize=(10, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow(fake_images[i], cmap="gray")
            ax.axis("off")
        plt.show()
