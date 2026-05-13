import os
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# =====================================================
# 1. LOAD DATASET
# =====================================================

with open("data/demos/gotolocal_seq.pkl", "rb") as f:
    data = pickle.load(f)

print("Total trajectories:", len(data))


# =====================================================
# 2. GROUP BY INSTRUCTION
# =====================================================

grouped = defaultdict(list)

for traj in data:

    grouped[traj["instruction"]].append(traj)

print("Total Instructions:", len(grouped))

for k in grouped:
    print(k, "→", len(grouped[k]))


# =====================================================
# 3. BUILD VOCAB
# =====================================================

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1
}

vocab = dict(SPECIAL_TOKENS)

for traj in data:

    tokens = traj["instruction"].lower().split()

    for token in tokens:

        if token not in vocab:

            vocab[token] = len(vocab)

print("Vocab Size:", len(vocab))


# =====================================================
# 4. TOKENIZER
# =====================================================

MAX_INSTR_LEN = 10


def tokenize_instruction(instr):

    tokens = instr.lower().split()

    ids = []

    for tok in tokens:

        ids.append(
            vocab.get(tok, vocab["<unk>"])
        )

    while len(ids) < MAX_INSTR_LEN:

        ids.append(vocab["<pad>"])

    return ids[:MAX_INSTR_LEN]


# =====================================================
# 5. TRAJECTORY DATASET
# =====================================================

class TrajectoryDataset(Dataset):

    def __init__(self, trajectories):

        self.trajectories = trajectories

    def __len__(self):

        return len(self.trajectories)

    def __getitem__(self, idx):

        traj = self.trajectories[idx]

        instr = tokenize_instruction(
            traj["instruction"]
        )

        obs_seq = torch.tensor(
            traj["obs_seq"],
            dtype=torch.float32
        ) / 255.0

        act_seq = torch.tensor(
            traj["act_seq"],
            dtype=torch.long
        )

        instr = torch.tensor(
            instr,
            dtype=torch.long
        )

        return instr, obs_seq, act_seq


# =====================================================
# 6. COLLATE FUNCTION
# =====================================================

def collate_fn(batch):

    instrs = []
    obs_seqs = []
    act_seqs = []
    lengths = []

    for instr, obs_seq, act_seq in batch:

        instrs.append(instr)

        obs_seqs.append(obs_seq)

        act_seqs.append(act_seq)

        lengths.append(len(obs_seq))

    max_len = max(lengths)

    padded_obs = []
    padded_act = []

    for obs_seq, act_seq in zip(obs_seqs, act_seqs):

        pad_len = max_len - len(obs_seq)

        if pad_len > 0:

            obs_pad = torch.zeros(
                pad_len,
                7,
                7,
                3
            )

            act_pad = torch.zeros(
                pad_len,
                dtype=torch.long
            )

            obs_seq = torch.cat(
                [obs_seq, obs_pad],
                dim=0
            )

            act_seq = torch.cat(
                [act_seq, act_pad],
                dim=0
            )

        padded_obs.append(obs_seq)
        padded_act.append(act_seq)

    return (
        torch.stack(instrs),
        torch.stack(padded_obs),
        torch.stack(padded_act),
        lengths
    )


# =====================================================
# 7. FiLM BLOCK
# =====================================================

class FiLMBlock(nn.Module):

    def __init__(self, instr_dim, channels):

        super().__init__()

        self.gamma = nn.Linear(
            instr_dim,
            channels
        )

        self.beta = nn.Linear(
            instr_dim,
            channels
        )

    def forward(self, x, instr_embedding):

        gamma = self.gamma(instr_embedding)
        beta = self.beta(instr_embedding)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * x + beta


# =====================================================
# 8. BABYAI POLICY MODEL
# =====================================================

class BabyAIModel(nn.Module):

    def __init__(self,
                 vocab_size,
                 n_actions=7):

        super().__init__()

        # =====================================
        # WORD EMBEDDING
        # =====================================

        self.word_embedding = nn.Embedding(
            vocab_size,
            128
        )

        # =====================================
        # GRU
        # =====================================

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            batch_first=True
        )

        # =====================================
        # CNN
        # =====================================

        self.conv1 = nn.Sequential(

            nn.Conv2d(
                3,
                128,
                kernel_size=2,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2, 2)
        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=1,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(),

            nn.MaxPool2d(2, 2)
        )

        # =====================================
        # FiLM
        # =====================================

        self.film1 = FiLMBlock(
            128,
            128
        )

        self.film2 = FiLMBlock(
            128,
            128
        )

        # =====================================
        # LSTM
        # =====================================

        self.lstm = nn.LSTMCell(
            input_size=128 * 2 * 2,
            hidden_size=128
        )

        # =====================================
        # POLICY HEAD
        # =====================================

        self.policy = nn.Sequential(

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                n_actions
            )
        )

    def encode_instruction(self, instr):

        emb = self.word_embedding(instr)

        _, hidden = self.gru(emb)

        return hidden.squeeze(0)

    def encode_obs(self,
                   obs,
                   instr_embedding):

        obs = obs.permute(0, 3, 1, 2)

        x = self.conv1(obs)

        x = self.film1(
            x,
            instr_embedding
        )

        x = self.conv2(x)

        x = self.film2(
            x,
            instr_embedding
        )

        x = x.reshape(x.size(0), -1)

        return x

    def forward(self,
                instr,
                obs_seq,
                lengths):

        batch_size = obs_seq.size(0)
        seq_len = obs_seq.size(1)

        instr_embedding = self.encode_instruction(
            instr
        )

        hx = torch.zeros(
            batch_size,
            128,
            device=obs_seq.device
        )

        cx = torch.zeros(
            batch_size,
            128,
            device=obs_seq.device
        )

        outputs = []

        for t in range(seq_len):

            obs_t = obs_seq[:, t]

            x = self.encode_obs(
                obs_t,
                instr_embedding
            )

            hx, cx = self.lstm(
                x,
                (hx, cx)
            )

            logits = self.policy(hx)

            outputs.append(logits)

        outputs = torch.stack(
            outputs,
            dim=1
        )

        return outputs


# =====================================================
# 9. TRAIN FUNCTION
# =====================================================

def train_model(instruction,
                trajectories,
                device):

    print("\n========================")
    print("Training:", instruction)
    print("========================")

    dataset = TrajectoryDataset(
        trajectories
    )

    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = BabyAIModel(
        vocab_size=len(vocab)
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    patience = 5
    patience_counter = 0

    max_epochs = 50

    for epoch in range(max_epochs):

        # =====================================
        # TRAIN
        # =====================================

        model.train()

        train_loss = 0

        train_correct = 0
        train_total = 0

        for instr, obs_seq, act_seq, lengths in train_loader:

            instr = instr.to(device)
            obs_seq = obs_seq.to(device)
            act_seq = act_seq.to(device)

            logits = model(
                instr,
                obs_seq,
                lengths
            )

            logits = logits.reshape(
                -1,
                7
            )

            targets = act_seq.reshape(-1)

            loss = loss_fn(
                logits,
                targets
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

            preds = torch.argmax(
                logits,
                dim=1
            )

            train_correct += (
                preds == targets
            ).sum().item()

            train_total += targets.size(0)

        train_acc = train_correct / train_total

        # =====================================
        # VALIDATION
        # =====================================

        model.eval()

        val_loss = 0

        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for instr, obs_seq, act_seq, lengths in val_loader:

                instr = instr.to(device)
                obs_seq = obs_seq.to(device)
                act_seq = act_seq.to(device)

                logits = model(
                    instr,
                    obs_seq,
                    lengths
                )

                logits = logits.reshape(
                    -1,
                    7
                )

                targets = act_seq.reshape(-1)

                loss = loss_fn(
                    logits,
                    targets
                )

                val_loss += loss.item()

                preds = torch.argmax(
                    logits,
                    dim=1
                )

                val_correct += (
                    preds == targets
                ).sum().item()

                val_total += targets.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}%"
        )

        # =====================================
        # EARLY STOPPING
        # =====================================

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_name = (
                instruction
                .replace(" ", "_")
                + ".pt"
            )

            save_path = os.path.join(
                "data/models",
                save_name
            )

            torch.save(
                model.state_dict(),
                save_path
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= patience:

            print("Early stopping triggered")

            break


# =====================================================
# 10. TRAIN ALL 18 MODELS
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:", device)

os.makedirs(
    "data/models",
    exist_ok=True
)

for instruction in grouped:

    trajectories = grouped[instruction]

    train_model(
        instruction,
        trajectories,
        device
    )

print("\nAll 18 models trained.")