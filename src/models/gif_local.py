import os
import re
import warnings

warnings.filterwarnings("ignore")

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import imageio

import gymnasium as gym
import minigrid

import torch
import torch.nn as nn


# =====================================================
# 1. NORMALIZE INSTRUCTION
# =====================================================

def normalize_instruction(instr):

    instr = instr.lower()

    instr = re.sub(
        r"\bgo to a (\w+ \w+)\b",
        r"go to the \1",
        instr
    )

    return instr.strip()


# =====================================================
# 2. BUILD VOCAB
# =====================================================

SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1
}

vocab = dict(SPECIAL_TOKENS)

instructions = [
    "go to the blue ball",
    "go to the blue box",
    "go to the blue key",

    "go to the green ball",
    "go to the green box",
    "go to the green key",

    "go to the grey ball",
    "go to the grey box",
    "go to the grey key",

    "go to the purple ball",
    "go to the purple box",
    "go to the purple key",

    "go to the red ball",
    "go to the red box",
    "go to the red key",

    "go to the yellow ball",
    "go to the yellow box",
    "go to the yellow key"
]

for instr in instructions:

    for token in instr.split():

        if token not in vocab:

            vocab[token] = len(vocab)


# =====================================================
# 3. TOKENIZER
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
# 4. FiLM BLOCK
# =====================================================

class FiLMBlock(nn.Module):

    def __init__(self,
                 instr_dim,
                 channels):

        super().__init__()

        self.gamma = nn.Linear(
            instr_dim,
            channels
        )

        self.beta = nn.Linear(
            instr_dim,
            channels
        )

    def forward(self,
                x,
                instr_embedding):

        gamma = self.gamma(instr_embedding)
        beta = self.beta(instr_embedding)

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        return gamma * x + beta


# =====================================================
# 5. BABYAI MODEL
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
        # CNN BLOCK 1
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

        # =====================================
        # CNN BLOCK 2
        # =====================================

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

    def forward_step(self,
                     instr_embedding,
                     obs,
                     hx,
                     cx):

        x = self.encode_obs(
            obs,
            instr_embedding
        )

        hx, cx = self.lstm(
            x,
            (hx, cx)
        )

        logits = self.policy(hx)

        return logits, hx, cx


# =====================================================
# 6. DEVICE
# =====================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# =====================================================
# 7. ENVIRONMENT
# =====================================================

env = gym.make(
    "BabyAI-GoToLocal-v0",
    render_mode="rgb_array"
)

env = env.unwrapped


# =====================================================
# 8. MODEL DIRECTORY
# =====================================================

model_dir = "data/models"

model_files = sorted([
    f for f in os.listdir(model_dir)
    if f.endswith(".pt")
])

print("\nFound models:", len(model_files))


# =====================================================
# 9. GENERATE GIF
# =====================================================

all_frames = []

for model_file in model_files:

    # =====================================
    # RECOVER INSTRUCTION
    # =====================================

    instruction = model_file.replace(".pt", "")

    instruction = instruction.replace("_", " ")

    instruction = normalize_instruction(
        instruction
    )

    print("\n====================")
    print("Instruction:", instruction)
    print("====================")

    # =====================================
    # LOAD MODEL
    # =====================================

    model_path = os.path.join(
        model_dir,
        model_file
    )

    model = BabyAIModel(
        vocab_size=len(vocab)
    ).to(device)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model.eval()

    # =====================================
    # ENCODE INSTRUCTION
    # =====================================

    instr_tensor = torch.tensor(
        [tokenize_instruction(instruction)],
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        instr_embedding = model.encode_instruction(
            instr_tensor
        )

    # =====================================
    # RESET UNTIL MATCHING MISSION
    # =====================================

    while True:

        obs, _ = env.reset()

        mission = normalize_instruction(
            env.mission
        )

        if mission == instruction:
            break

    print("Environment Mission:", mission)

    # =====================================
    # INITIAL LSTM STATE
    # =====================================

    hx = torch.zeros(
        1,
        128,
        device=device
    )

    cx = torch.zeros(
        1,
        128,
        device=device
    )

    # =====================================
    # ROLLOUT
    # =====================================

    done = False

    reward = 0

    steps = 0

    max_steps = 150

    while not done and steps < max_steps:

        # ---------------------------------
        # SAVE FRAME
        # ---------------------------------

        frame = env.get_frame(
            highlight=True,
            tile_size=32
        )

        all_frames.append(frame)

        # ---------------------------------
        # MODEL ACTION
        # ---------------------------------

        img = torch.tensor(
            obs["image"],
            dtype=torch.float32
        ).unsqueeze(0) / 255.0

        img = img.to(device)

        with torch.no_grad():

            logits, hx, cx = model.forward_step(
                instr_embedding,
                img,
                hx,
                cx
            )

            action = torch.argmax(
                logits,
                dim=1
            ).item()

        # ---------------------------------
        # STEP ENVIRONMENT
        # ---------------------------------

        obs, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        steps += 1

    # =====================================
    # FINAL FRAME
    # =====================================

    frame = env.get_frame(
        highlight=True,
        tile_size=32
    )

    all_frames.append(frame)

    print("Reward:", reward)

    print("Success:", reward > 0)

    print("Steps:", steps)

    # =====================================
    # PAUSE BETWEEN EPISODES
    # =====================================

    for _ in range(10):

        all_frames.append(frame)


# =====================================================
# 10. SAVE GIF
# =====================================================

output_path = "all_policies.gif"

imageio.mimsave(
    output_path,
    all_frames,
    fps=5,
    loop=0
)

print("\n================================")
print("Saved GIF:", output_path)
print("================================")