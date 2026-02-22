import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

class PositionPredictionDataset(Dataset):
    def __init__(self, num_processes=4, is_test=False, train_min=None, train_max=None):
        init_imgs_list, actions_list, final_obj_positions_list = [], [], []
        suffix = "_test" if is_test else ""

        for i in range(num_processes):
            init_imgs_list.append(torch.load(f"dataset/initial_imgs{suffix}_{i}.pt"))
            actions_list.append(torch.load(f"dataset/actions{suffix}_{i}.pt"))
            final_obj_positions_list.append(torch.load(f"dataset/final_obj_positions{suffix}_{i}.pt"))

        self.init_imgs = torch.cat(init_imgs_list, dim=0).float() / 255.0
        self.final_obj_positions = torch.cat(final_obj_positions_list, dim=0).float()
        self.actions = torch.cat(actions_list, dim=0).long()

        # This condition is needed to normalize the test data with the same values
        if train_min is not None and train_max is not None:
            self.pos_min = train_min
            self.pos_max = train_max
        else:
            self.pos_min = self.final_obj_positions.min(dim=0)[0]
            self.pos_max = self.final_obj_positions.max(dim=0)[0]

        self.final_obj_positions = (self.final_obj_positions - self.pos_min) / (self.pos_max - self.pos_min)

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return self.init_imgs[idx], self.actions[idx], self.final_obj_positions[idx]
    
class PositionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # (kernel_size=3, padding=1) combination keeps the image in the same size
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2) # 128x128 -> 64x64

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2) # 64x64 -> 32x32

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2) # 32x32 -> 16x16

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, 2) # 16x16 -> 8x8

        self.fc1 = nn.Linear(8192 + 4, 256)
        self.ln1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, img, action):
        batch_size = img.size(0)

        x = self.pool1(F.relu(self.conv1(img)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))

        x = x.view(batch_size, -1)

        action_one_hot = F.one_hot(action.long(), num_classes=4).float()
        x = torch.cat((x, action_one_hot), dim=1)

        x = F.relu(self.ln1(self.fc1(x)))

        out_coords = torch.sigmoid(self.fc2(x))

        return out_coords
    
def train(model, train_loader, val_loader, dataset, epochs=15, lr=1e-3, device="cpu"):
    model = model.to(device)
    J = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loss_history, val_loss_history = [], []
    val_distance_history = []  # In addition to MSE, I wanted to track the distance between true and pred

    pos_min = dataset.pos_min.to(device)
    pos_max = dataset.pos_max.to(device)

    print(f"Training Position Model on {device}")
    for epoch in range(epochs):
        # Train
        model.train()
        running_train_loss = 0.0
        for init_imgs, actions, final_pos in train_loader:
            init_imgs, actions, final_pos = init_imgs.to(device), actions.to(device), final_pos.to(device)

            optimizer.zero_grad()
            predictions = model(init_imgs, actions)
            loss = J(predictions, final_pos)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * init_imgs.size(0)

        avg_train_loss = running_train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)

        # Validation
        model.eval()
        running_val_loss = 0.0
        total_val_distance = 0.0
        with torch.no_grad():
            for init_imgs, actions, true_norm_pos in val_loader:
                init_imgs, actions, true_norm_pos = init_imgs.to(device), actions.to(device), true_norm_pos.to(device)

                pred_norm_pos = model(init_imgs, actions)
                loss = J(pred_norm_pos, true_norm_pos)
                running_val_loss += loss.item() * init_imgs.size(0)

                # Calculate physical distance
                pred_real_pos = (pred_norm_pos * (pos_max - pos_min)) + pos_min
                true_real_pos = (true_norm_pos * (pos_max - pos_min)) + pos_min
                distances = torch.norm(pred_real_pos - true_real_pos, dim=1)
                total_val_distance += distances.sum().item()

        avg_val_loss = running_val_loss / len(val_loader.dataset)
        avg_val_distance = total_val_distance / len(val_loader.dataset)

        val_loss_history.append(avg_val_loss)
        val_distance_history.append(avg_val_distance)

        print(f"Epoch [{epoch+1}/{epochs}] | Train MSE: {avg_train_loss:.5f} | Val MSE: {avg_val_loss:.5f} | Val Dist: {avg_val_distance:.4f}")

    save_path = "PositionCNN_weights.pth"
    torch.save(model.state_dict(), save_path)

    return model, train_loss_history, val_loss_history, val_distance_history

def evaluate_distance_error(model, dataloader, dataset, device="cpu"):
    model.eval()
    total_distance_error = 0.0

    # Normalization factors
    pos_min = dataset.pos_min.to(device)
    pos_max = dataset.pos_max.to(device)

    with torch.no_grad():
        for init_imgs, actions, true_norm_pos in dataloader:
            init_imgs = init_imgs.to(device)
            actions = actions.to(device)
            true_norm_pos = true_norm_pos.to(device)

            pred_norm_pos = model(init_imgs, actions)

            # Unnormalize
            pred_real_pos = (pred_norm_pos * (pos_max - pos_min)) + pos_min
            true_real_pos = (true_norm_pos * (pos_max - pos_min)) + pos_min
            distances = torch.norm(pred_real_pos - true_real_pos, dim=1)

            total_distance_error += distances.sum().item()

    avg_distance = total_distance_error / len(dataloader.dataset)
    return avg_distance

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

def test(model, train_dataset, device="cpu"):
    print("\n" + "=" * 50)
    print("TEST SET EVALUATION")
    print("=" * 50)

    print("\nLoading Position Test Data")
    pos_test_dataset = PositionPredictionDataset(
        num_processes=4,
        is_test=True,
        train_min=train_dataset.pos_min,
        train_max=train_dataset.pos_max
    )
    pos_test_loader = DataLoader(pos_test_dataset, batch_size=64, shuffle=False)
    print(f"Loaded {len(pos_test_dataset)} test samples.")

    test_dist = evaluate_distance_error(model, pos_test_loader, pos_test_dataset, device)

    print(f"PositionCNN Test Distance Error: {test_dist:.4f} units")
    
    return test_dist


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("POSITION PREDICTION")
    pos_dataset = PositionPredictionDataset(num_processes=4)

    total_size = len(pos_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size

    pos_train_set, pos_val_set = random_split(pos_dataset, [train_size, val_size])
    pos_train_loader = DataLoader(pos_train_set, batch_size=64, shuffle=True)
    pos_val_loader = DataLoader(pos_val_set, batch_size=64, shuffle=False)

    print(f"Workspace Bounds (Normalization Factors) -> X: [{pos_dataset.pos_min[0]:.3f}, {pos_dataset.pos_max[0]:.3f}], Y: [{pos_dataset.pos_min[1]:.3f}, {pos_dataset.pos_max[1]:.3f}]\n")

    # Train PositionCNN
    trained_pos_cnn, pos_cnn_train_mse, pos_cnn_val_mse, pos_cnn_val_dist = train(
        PositionCNN(), pos_train_loader, pos_val_loader, pos_dataset, epochs=5, lr=0.001, device=device
    )
    cnn_error = evaluate_distance_error(trained_pos_cnn, pos_val_loader, pos_dataset, device=device)
    print(f"PositionCNN Validation Distance Error: {cnn_error:.4f} units\n")

    test(trained_pos_cnn, pos_dataset, device=device)

    plot_learning_curves(pos_cnn_train_mse, pos_cnn_val_mse, title="PositionCNN (MSE Loss)")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(pos_cnn_val_dist) + 1), pos_cnn_val_dist, color='purple', linewidth=2, label="CNN Distance")
    plt.title("PositionCNN Validation Distance Error", fontsize=14)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Distance Error (units)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()