import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# =====================
# 1. Load Dataset
# =====================

with open("data/demos/gotolocal_seq.pkl", "rb") as f:
    data = pickle.load(f)


# =====================
# 2. Group by Instruction
# =====================

grouped = defaultdict(list)

for d in data:

    instr = d["instruction"]

    for obs, act in zip(d["obs_seq"], d["act_seq"]):

        grouped[instr].append((obs, act))


print("Total Instructions:", len(grouped))

for k in grouped:
    print(k, "→", len(grouped[k]))


# =====================
# 3. Dataset
# =====================

class PolicyDataset(Dataset):

    def __init__(self, pairs):

        self.X = [p[0] for p in pairs]
        self.Y = [p[1] for p in pairs]

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        obs = torch.tensor(
            self.X[idx],
            dtype=torch.float32
        ) / 255.0

        act = torch.tensor(
            self.Y[idx],
            dtype=torch.long
        )

        return obs, act


# =====================
# 4. Improved CNN
# =====================

class PolicyNet(nn.Module):

    def __init__(self, n_actions=7):

        super().__init__()

        self.conv = nn.Sequential(

            # 7x7 -> 7x7
            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            # 7x7 -> 7x7
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            # IMPORTANT:
            # no pooling to preserve spatial info
        )

        self.fc = nn.Sequential(

            nn.Linear(
                64 * 7 * 7,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                n_actions
            )
        )

    def forward(self, x):

        # (B,7,7,3) -> (B,3,7,7)
        x = x.permute(0, 3, 1, 2)

        x = self.conv(x)

        x = x.reshape(x.size(0), -1)

        logits = self.fc(x)

        return logits


# =====================
# 5. Training Function
# =====================

def train_model(pairs, instruction, device):

    dataset = PolicyDataset(pairs)

    # 90 / 10 split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False
    )

    model = PolicyNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    loss_fn = nn.CrossEntropyLoss()

    # Early stopping
    max_epochs = 50
    patience = 5

    best_val_loss = float("inf")
    best_model_state = None

    patience_counter = 0

    for epoch in range(max_epochs):

        # =====================
        # TRAIN
        # =====================

        model.train()

        train_loss = 0

        train_correct = 0
        train_total = 0

        for obs, act in train_loader:

            obs = obs.to(device)
            act = act.to(device)

            logits = model(obs)

            loss = loss_fn(logits, act)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            train_correct += (preds == act).sum().item()

            train_total += act.size(0)

        train_acc = train_correct / train_total

        # =====================
        # VALIDATION
        # =====================

        model.eval()

        val_loss = 0

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for obs, act in val_loader:

                obs = obs.to(device)
                act = act.to(device)

                logits = model(obs)

                loss = loss_fn(logits, act)

                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)

                val_correct += (preds == act).sum().item()

                val_total += act.size(0)

        val_acc = val_correct / val_total

        print(
            f"{instruction} | "
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}%"
        )

        # =====================
        # EARLY STOPPING
        # =====================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_model_state = model.state_dict()

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print(f"Early stopping triggered for {instruction}")

            break

    # restore best model
    model.load_state_dict(best_model_state)

    return model


# =====================
# 6. Train All Models
# =====================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("\nUsing device:", device)

os.makedirs("data/models", exist_ok=True)


for instr in grouped:

    print("\n========================")
    print("Training:", instr)
    print("========================")

    pairs = grouped[instr]

    model = train_model(
        pairs,
        instr,
        device
    )

    fname = instr.replace(" ", "_") + ".pt"

    save_path = os.path.join(
        "data/models",
        fname
    )

    torch.save(
        model.state_dict(),
        save_path
    )

    print("Saved:", save_path)


print("\nAll models trained successfully.")