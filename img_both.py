import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

class ImagePredictionDataset(Dataset):
    def __init__(self, num_processes=4, is_test=False):
        init_imgs_list, actions_list, final_imgs_list = [], [], []
        # This is related to how I store the data (Explained in README.md)
        suffix = "_test" if is_test else ""

        for i in range(num_processes):
            init_imgs_list.append(torch.load(f"dataset/initial_imgs{suffix}_{i}.pt"))
            actions_list.append(torch.load(f"dataset/actions{suffix}_{i}.pt"))
            final_imgs_list.append(torch.load(f"dataset/final_imgs{suffix}_{i}.pt"))

        # Normalizing data helps the model converge faster
        self.init_imgs = torch.cat(init_imgs_list, dim=0).float() / 255.0
        self.final_imgs = torch.cat(final_imgs_list, dim=0).float() / 255.0
        self.actions = torch.cat(actions_list, dim=0).long()

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.init_imgs[idx], self.actions[idx], self.final_imgs[idx]
    
class ImgMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(49156, 1024)  # 49,152 from pixels + 4 from actions
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, 49152)  # predict image

    def forward(self, img, action):
        batch_size = img.size(0)
        img_flat = img.view(batch_size, -1)
        action_one_hot = F.one_hot(action.long(), num_classes=4).float()  # e.g. Action 0: 1 0 0 0

        x = torch.cat((img_flat, action_one_hot), dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        x = torch.sigmoid(x)  # To stay in [0, 1]

        output_img = x.view(batch_size, 3, 128, 128)
        return output_img
    
class ImgCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # (kernel_size=3, padding=1) combination keeps the image in the same size
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=3, padding=1)

    def forward(self, img, action):
        batch_size = img.size(0)
        action_one_hot = F.one_hot(action.long(), num_classes=4).float()
        action_spatial = action_one_hot.view(batch_size, 4, 1, 1).expand(-1, -1, 128, 128)

        x = torch.cat((img, action_spatial), dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)

        output_img = torch.sigmoid(x)
        return output_img
    
def train(model, train_loader, val_loader, epochs=15, lr=1e-3, device="cpu"):
    model = model.to(device)
    J = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_history, val_history = [], []

    print(f"Training Image Model on {device}")
    for epoch in range(epochs):
        # Train
        model.train()
        running_train_loss = 0.0
        for init_imgs, actions, final_imgs in train_loader:
            init_imgs, actions, final_imgs = init_imgs.to(device), actions.to(device), final_imgs.to(device)

            optimizer.zero_grad()
            predictions = model(init_imgs, actions)
            loss = J(predictions, final_imgs)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * init_imgs.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_history.append(avg_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for init_imgs, actions, final_imgs in val_loader:
                init_imgs, actions, final_imgs = init_imgs.to(device), actions.to(device), final_imgs.to(device)
                predictions = model(init_imgs, actions)
                loss = J(predictions, final_imgs)
                running_val_loss += loss.item() * init_imgs.size(0)

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        val_history.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}")

    save_path = f"{model.__class__.__name__}_weights.pth"
    torch.save(model.state_dict(), save_path)

    return model, train_history, val_history

def plot_learning_curves(train_history, val_history, title="Learning Curve", ylabel="MSE Loss"):
    epochs = range(1, len(train_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_history, label='Training', color='blue', linewidth=2)
    plt.plot(epochs, val_history, label='Validation', color='orange', linewidth=2, linestyle='--')

    plt.title(title, fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()

def evaluate_img_test_set(model, test_loader, device="cpu"):
    model.eval()
    J = nn.MSELoss()
    running_loss = 0.0
    with torch.no_grad():
        for init_imgs, actions, final_imgs in test_loader:
            init_imgs, actions, final_imgs = init_imgs.to(device), actions.to(device), final_imgs.to(device)
            predictions = model(init_imgs, actions)
            loss = J(predictions, final_imgs)
            running_loss += loss.item() * init_imgs.size(0)
    return running_loss / len(test_loader.dataset)

def visualize_test_predictions(img_dataset, cnn_model, mlp_model, device, num_samples=5):
    # 5 random indices from the test set
    indices = np.random.choice(len(img_dataset), num_samples, replace=False)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    cnn_model.eval()
    mlp_model.eval()

    with torch.no_grad():
        for row_idx, data_idx in enumerate(indices):
            init_img, action, true_final_img = img_dataset[data_idx]

            # .unsqueeze() adds some kind of a fake batch_size dimension
            # which is needed to pass data to NN
            init_batch = init_img.unsqueeze(0).to(device)
            action_batch = action.unsqueeze(0).to(device)

            cnn_pred = cnn_model(init_batch, action_batch).squeeze(0).cpu()
            mlp_pred = mlp_model(init_batch, action_batch).squeeze(0).cpu()

            # Initial image
            ax = axes[row_idx, 0]
            ax.imshow(init_img.permute(1, 2, 0).numpy())
            ax.axis('off')
            ax.text(5, 15, f"Action: {action.item()}", color='white',
                    backgroundcolor='black', fontsize=12, fontweight='bold')
            if row_idx == 0: ax.set_title("Initial Image & Action", fontsize=14)

            # True final image
            ax = axes[row_idx, 1]
            ax.imshow(true_final_img.permute(1, 2, 0).numpy())
            ax.axis('off')
            if row_idx == 0: ax.set_title("True Final Image", fontsize=14)

            # ImgCNN prediction
            ax = axes[row_idx, 2]
            ax.imshow(cnn_pred.permute(1, 2, 0).numpy())
            ax.axis('off')
            if row_idx == 0: ax.set_title("ImgCNN Prediction", fontsize=14)

            # ImgMLP prediction
            ax = axes[row_idx, 3]
            ax.imshow(mlp_pred.permute(1, 2, 0).numpy())
            ax.axis('off')
            if row_idx == 0: ax.set_title("ImgMLP Prediction", fontsize=14)

    plt.tight_layout()
    plt.show()

def test(trained_img_mlp, trained_img_cnn, device="cpu"):
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)

    print("\nLoading Image Test Data")
    img_test_dataset = ImagePredictionDataset(num_processes=4, is_test=True)
    img_test_loader = DataLoader(img_test_dataset, batch_size=64, shuffle=False)
    print(f"Loaded {len(img_test_dataset)} test samples.")

    img_mlp_test_loss = evaluate_img_test_set(trained_img_mlp, img_test_loader, device)
    img_cnn_test_loss = evaluate_img_test_set(trained_img_cnn, img_test_loader, device)

    print(f"ImgMLP Test MSE: {img_mlp_test_loss:.5f}")
    print(f"ImgCNN Test MSE: {img_cnn_test_loss:.5f}")

    print("5 TEST SET VISUALIZATIONS\n")
    visualize_test_predictions(img_test_dataset, trained_img_cnn, trained_img_mlp, device, num_samples=5)
    
    return img_mlp_test_loss, img_cnn_test_loss

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("IMAGE GENERATION")
    img_dataset = ImagePredictionDataset(num_processes=4)

    total_size = len(img_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    img_train_set, img_val_set = random_split(img_dataset, [train_size, val_size])

    img_train_loader = DataLoader(img_train_set, batch_size=64, shuffle=True)
    img_val_loader = DataLoader(img_val_set, batch_size=64, shuffle=False)

    print(f"Split {total_size} images -> {train_size} Train | {val_size} Val\n")

    # Train ImgMLP
    trained_img_mlp, img_mlp_train_loss, img_mlp_val_loss = train(
        ImgMLP(), img_train_loader, img_val_loader, epochs=5, lr=0.001, device=device
    )
    plot_learning_curves(img_mlp_train_loss, img_mlp_val_loss, title="ImgMLP MSE Loss")

    # Train ImgCNN
    trained_img_cnn, img_cnn_train_loss, img_cnn_val_loss = train(
        ImgCNN(), img_train_loader, img_val_loader, epochs=5, lr=0.001, device=device
    )
    plot_learning_curves(img_cnn_train_loss, img_cnn_val_loss, title="ImgCNN MSE Loss")


    # Visualizing using the validation set
    test_init_img, test_action, test_final_img = img_val_set[0]
    test_init_img_batch = test_init_img.unsqueeze(0).to(device)
    test_action_batch = test_action.unsqueeze(0).to(device)

    trained_img_cnn.eval()
    trained_img_mlp.eval()
    with torch.no_grad():
        cnn_pred = trained_img_cnn(test_init_img_batch, test_action_batch).squeeze(0).cpu()
        mlp_pred = trained_img_mlp(test_init_img_batch, test_action_batch).squeeze(0).cpu()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(test_init_img.permute(1, 2, 0).numpy())
    axes[0].set_title(f"Initial Image (Action: {test_action.item()})")
    axes[1].imshow(test_final_img.permute(1, 2, 0).numpy())
    axes[1].set_title("True Final Image")
    axes[2].imshow(cnn_pred.permute(1, 2, 0).numpy())
    axes[2].set_title("ImgCNN Prediction")
    axes[3].imshow(mlp_pred.permute(1, 2, 0).numpy())
    axes[3].set_title("ImgMLP Prediction")
    for ax in axes: ax.axis('off')
    plt.show()

    test(trained_img_mlp, trained_img_cnn, device=device)