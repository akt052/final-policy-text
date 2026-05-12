import pickle
import os
import gymnasium as gym
import numpy as np
import hashlib 
import re
from minigrid.utils.baby_ai_bot import BabyAIBot


def normalize_instruction(instr):
    return re.sub(r"\bgo to a (\w+ \w+)\b", r"go to the \1", instr)


def get_traj_hash(obs_seq, act_seq):
    obs_bytes = np.array(obs_seq, dtype=np.uint8).tobytes()
    act_bytes = np.array(act_seq, dtype=np.uint8).tobytes() 
    return hashlib.sha1(obs_bytes + act_bytes).hexdigest()


def generate_dataset(
    env_name="BabyAI-GoToLocal-v0",
    target_per_mission=550,
    save_path="data/demos/gotolocal_seq.pkl"
):
    env = gym.make(env_name).unwrapped

    demos = []
    mission_counts = {}
    seen = set()

    while True:
        obs, _ = env.reset()
        instruction = normalize_instruction(env.mission)

        if instruction not in mission_counts:
            mission_counts[instruction] = 0

        if mission_counts[instruction] >= target_per_mission:
            continue

        bot = BabyAIBot(env)
        
        obs_seq = []
        act_seq = []

        done = False
        reward = 0

        while not done:
            try: 
                action = bot.replan(obs)
            except AssertionError:
                break
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs_seq.append(np.array(obs['image'], dtype=np.uint8))
            act_seq.append(int(action))

            obs = next_obs

            if len(obs_seq) > 100:
                break

        if done and reward > 0:
            traj_hash = get_traj_hash(obs_seq, act_seq)

            if traj_hash in seen:
                continue

            seen.add(traj_hash)  

            demos.append({
                "instruction": instruction,
                "obs_seq": obs_seq,
                "act_seq": act_seq 
            })

            mission_counts[instruction] += 1

            total = sum(mission_counts.values())
            print(f"{total} | {instruction}: {mission_counts[instruction]}")

            if all(c >= target_per_mission for c in mission_counts.values()) and len(mission_counts) >= 18:
                break

    with open(save_path, "wb") as f:
        pickle.dump(demos, f)

    print("\nSaved dataset:", save_path)