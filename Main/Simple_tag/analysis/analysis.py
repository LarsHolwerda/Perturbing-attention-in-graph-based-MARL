import torch

data = torch.load("simple_tag__GAPPO__40__1761630900.pt")
for key, value in data.items():
    print(key, len(value))
print(len(data["observations"]))
i = 0  # pick an index

obs = data["observations"][i]
acts = data["actions"][i]
adj = data["adjacency"][i]
done = data["done"][i]

print("obs shape:", obs.shape)
print("acts shape:", acts.shape)
print("adj shape:", adj.shape)
print("done shape:", done.shape)

print("First obs shape:", data["observations"][0].shape)
print("Last obs shape:", data["observations"][-1].shape)

# Check approximate total number of steps stored
num_batches = len(data["observations"])
steps_per_batch = data["observations"][0].shape[1]
num_envs = data["observations"][0].shape[0]
total_steps = num_batches * num_envs * steps_per_batch
print(f"Approx total steps stored: {total_steps:,}")
print(data["observations"][0].shape)
print(len(data["observations"]))

for i, obs in enumerate(data["observations"]):
    print(f"Batch {i}: {obs.shape}")