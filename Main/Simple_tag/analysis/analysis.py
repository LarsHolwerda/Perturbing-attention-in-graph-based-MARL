import torch

data = torch.load("simple_tag__GAPPO__0__1760450999.pt")
for key, value in data.items():
    print(key, len(value))

i = 0  # pick an index

obs = data["observations"][i]
acts = data["actions"][i]
adj = data["adjacency"][i]
done = data["done"][i]

print("obs shape:", obs.shape)
print("acts shape:", acts.shape)
print("adj shape:", adj.shape)
print("done shape:", done.shape)

