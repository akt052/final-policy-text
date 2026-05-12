import pickle
from collections import defaultdict

with open("data/demos/gotolocal_seq.pkl", "rb") as f:
    data = pickle.load(f)

flat_data = []

for d in data:
    instr = d["instruction"]
    for obs, act in zip(d["obs_seq"], d["act_seq"]):
        flat_data.append({
            "instruction": instr,
            "obs": obs,
            "action": act
        })

print("Total samples:", len(flat_data))


grouped = defaultdict(list)

for d in data:   
    instr = d["instruction"]
    
    for obs, act in zip(d["obs_seq"], d["act_seq"]):
        grouped[instr].append((obs, act))

obs, act = grouped["go to the green key"][2954]
print(act)

# check
for k in grouped:
    print(k, "→", len(grouped[k]))