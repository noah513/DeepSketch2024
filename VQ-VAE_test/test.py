import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score

# GPU 설정 (0번 GPU를 사용하도록 지정)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 데이터 디렉토리 설정
data_dir = './data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# 데이터 다운로드 여부를 확인
download = not os.path.exists(os.path.join(data_dir, 'MNIST/raw'))

# MNIST 데이터 로드 및 전처리
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root=data_dir, train=True, download=download, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=download, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

train_data = next(iter(train_loader))
test_data = next(iter(test_loader))

x_train, y_train = train_data[0].numpy(), train_data[1].numpy()
x_test, y_test = test_data[0].numpy(), test_data[1].numpy()

# 0-9의 각 숫자 이미지에서 1000개씩 샘플링하여 사용
num_samples = 1000
x_train_samples = []
y_train_samples = []

for i in range(10):
    idx = np.where(y_train == i)[0][:num_samples]
    x_train_samples.append(x_train[idx])
    y_train_samples.append(y_train[idx])

x_train_samples = np.concatenate(x_train_samples)
y_train_samples = np.concatenate(y_train_samples)

# 0-99 이미지 생성
def create_combined_images(x_samples, num_samples_per_class_train=200, num_samples_per_class_test=20):
    combined_images_train = []
    combined_images_test = []
    for i in range(100):
        for _ in range(num_samples_per_class_train):
            num1 = i // 10
            num2 = i % 10
            img1 = x_samples[num1 * num_samples + np.random.randint(num_samples)]
            img2 = x_samples[num2 * num_samples + np.random.randint(num_samples)]
            combined_img = np.hstack((img1[0], img2[0]))
            combined_images_train.append(combined_img)
        for _ in range(num_samples_per_class_test):
            num1 = i // 10
            num2 = i % 10
            img1 = x_samples[num1 * num_samples + np.random.randint(num_samples)]
            img2 = x_samples[num2 * num_samples + np.random.randint(num_samples)]
            combined_img = np.hstack((img1[0], img2[0]))
            combined_images_test.append(combined_img)
    return (np.array(combined_images_train), np.array(combined_images_test))

# 20,000개의 training dataset 및 2,000개의 testing dataset 생성
combined_images_train, combined_images_test = create_combined_images(x_train_samples, num_samples_per_class_train=200, num_samples_per_class_test=20)

# 이미지 평탄화
combined_images_train_flat = combined_images_train.reshape(combined_images_train.shape[0], -1)
combined_images_test_flat = combined_images_test.reshape(combined_images_test.shape[0], -1)

# 데이터셋 크기 출력
print(f'Training data shape: {combined_images_train_flat.shape}')
print(f'Test data shape: {combined_images_test_flat.shape}')

# 데이터셋 샘플 확인
plt.figure(figsize=(10, 2))
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.imshow(combined_images_train[i].reshape(28, 56), cmap='gray')
    plt.axis('off')
plt.show()

# 데이터셋을 PyTorch Tensor로 변환 및 DataLoader 생성
X_train_tensor = torch.tensor(combined_images_train_flat, dtype=torch.float32)
train_dataset = torch.utils.data.TensorDataset(X_train_tensor)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

X_test_tensor = torch.tensor(combined_images_test_flat, dtype=torch.float32)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

# VQ-VAE 모델 정의
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, x):
        flatten = x.view(-1, self.embedding_dim)
        distances = (torch.sum(flatten ** 2, dim=1, keepdim=True) 
                     + torch.sum(self.embeddings.weight ** 2, dim=1)
                     - 2 * torch.matmul(flatten, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embeddings(encoding_indices).view_as(x)

        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()
        avg_probs = torch.mean(torch.bincount(encoding_indices.flatten()) / encoding_indices.numel())
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices

class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_latents, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.num_latents = num_latents
        self.embedding_dim = embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_latents * embedding_dim)
        )
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = nn.Sequential(
            nn.Linear(num_latents * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = z_e.view(-1, self.num_latents, self.embedding_dim)
        quantized_list = []
        vq_loss = 0
        perplexity = 0
        encoding_indices_list = []
        for i in range(self.num_latents):
            quantized, loss, p, encoding_indices = self.quantizer(z_e[:, i, :])
            quantized_list.append(quantized)
            vq_loss += loss
            perplexity += p
            encoding_indices_list.append(encoding_indices)
        z_q = torch.stack(quantized_list, dim=1).view(-1, self.num_latents * self.embedding_dim)
        encoding_indices = torch.stack(encoding_indices_list, dim=1)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss / self.num_latents, perplexity / self.num_latents, encoding_indices

# 모델 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hidden_dim = 512
input_dim = 1568
num_latents = 64
embedding_dim = 16
num_embeddings = 512
commitment_cost = 0.25
vqvae = VQVAE(input_dim, hidden_dim, num_latents, embedding_dim, num_embeddings, commitment_cost).to(device)

# VQ-VAE 모델 학습
vqvae_model_path = f'model/VQVAE_latents_{num_latents}_embed_{embedding_dim}.pth'
os.makedirs('model', exist_ok=True)

vqvae_loaded = False
if os.path.exists(vqvae_model_path):
    choice = input("학습된 VQ-VAE 모델이 있습니다. 로드하시겠습니까? (Y/N): ").strip().upper()
else:
    choice = "N"

if choice == "Y":
    vqvae.load_state_dict(torch.load(vqvae_model_path))
    vqvae_loaded = True
    print("VQ-VAE 모델이 로드되었습니다.")
else:
    optimizer = optim.Adam(vqvae.parameters(), lr=4.5e-4)
    num_epochs = 20

    vqvae_losses = []
    for epoch in range(num_epochs):
        vqvae.train()
        train_loss = 0
        for images, in train_loader:
            images = images.to(device)

            optimizer.zero_grad()
            recon_images, vq_loss, _, _ = vqvae(images)
            recon_loss = F.mse_loss(recon_images, images)
            loss = recon_loss + vq_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        vqvae_losses.append(train_loss / len(train_loader))
        print(f'Epoch [{epoch+1}/{num_epochs}], VQ-VAE Loss: {train_loss/len(train_loader):.4f}')

    # VQ-VAE 모델 저장
    torch.save(vqvae.state_dict(), vqvae_model_path)

# VQ-VAE를 이용해 얻은 임베딩으로 클러스터링 수행
vqvae.eval()
train_indices = []
test_indices = []

with torch.no_grad():
    for images, in train_loader:
        images = images.to(device)
        _, _, _, encoding_indices = vqvae(images)
        train_indices.append(encoding_indices.cpu().numpy())

    for images, in test_loader:
        images = images.to(device)
        _, _, _, encoding_indices = vqvae(images)
        test_indices.append(encoding_indices.cpu().numpy())

train_indices = np.concatenate(train_indices)
test_indices = np.concatenate(test_indices)

# 각 이미지가 속한 클러스터의 빈도를 계산하여 feature로 사용
train_features = np.zeros((train_indices.shape[0], num_embeddings))
test_features = np.zeros((test_indices.shape[0], num_embeddings))

for i in range(train_indices.shape[0]):
    train_features[i] = np.bincount(train_indices[i].flatten(), minlength=num_embeddings)

for i in range(test_indices.shape[0]):
    test_features[i] = np.bincount(test_indices[i].flatten(), minlength=num_embeddings)

# 클러스터링을 수행하고 정확도를 평가
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

kmeans = KMeans(n_clusters=100, random_state=0).fit(train_features)
train_clusters = kmeans.predict(train_features)
test_clusters = kmeans.predict(test_features)

# ARI (Adjusted Rand Index)로 클러스터링 성능 평가
train_ari = adjusted_rand_score(y_train_samples[:20000], train_clusters)
test_ari = adjusted_rand_score(y_train_samples[20000:], test_clusters)

print(f'Train ARI: {train_ari:.4f}')
print(f'Test ARI: {test_ari:.4f}')

# 예측된 클러스터 결과 저장
results_df = pd.DataFrame({
    'True Label': y_train_samples,
    'Cluster': np.concatenate((train_clusters, test_clusters))
})

os.makedirs('clustering_results', exist_ok=True)
results_df.to_csv('clustering_results/clustering_results.csv', index=False)

# 그래프 저장을 위한 디렉토리 생성
os.makedirs('VQ_Logistic_graphs', exist_ok=True)

# VQ-VAE 손실 그래프 그리기
if not vqvae_loaded:
    plt.figure()
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, vqvae_losses, label='VQ-VAE Loss', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('VQ-VAE Loss', color='blue')
    plt.tick_params(axis='y', labelcolor='blue')
    plt.title('VQ-VAE Training Loss')
    plt.legend(loc='upper right')
    plt.savefig(os.path.join('VQ_Logistic_graphs', 'vqvae_loss.png'))
    plt.close()

print("클러스터링 및 성능 평가 완료")