import torch
import numpy as np
import matplotlib.pyplot as plt

records = torch.load("academy_3_vs_1_with_keeper__IPPO__0__1766332820.pt")

# Collect all sequences
all_sequences = set()
for r in records:
    all_sequences.update(r["sequence_percentages"].keys())

# Sort records chronologically
records = sorted(records, key=lambda r: (r["iteration"], r["batch"]))

# Build heatmap data
heatmap = {seq: [] for seq in all_sequences}
for r in records:
    for seq in all_sequences:
        heatmap[seq].append(r["sequence_percentages"].get(seq, 0.0))

# Compute total mass per sequence
sequence_total_mass = {
    seq: sum(freqs) for seq, freqs in heatmap.items()
}

# Filter out sequences that appear only once
filtered_sequences = [
    seq for seq, freqs in heatmap.items()
    if sum(f > 0 for f in freqs) > 3
]

# Sort sequences by importance
filtered_sequences = sorted(
    filtered_sequences,
    key=lambda seq: sequence_total_mass[seq],
    reverse=True
)

# Build matrix
matrix = np.array([heatmap[seq] for seq in filtered_sequences])

# Detect iteration boundaries
iteration_boundaries = []
last_iter = records[0]["iteration"]
for i, r in enumerate(records):
    if r["iteration"] != last_iter:
        iteration_boundaries.append(i)
        last_iter = r["iteration"]

# Plot
plt.figure(figsize=(14, max(4, len(filtered_sequences) * 0.4)))

im = plt.imshow(
    matrix,
    aspect="auto",
    interpolation="nearest",
    cmap="viridis",
    vmin=0.0,
    vmax=matrix.max()
)

for b in iteration_boundaries:
    plt.axvline(b - 0.5, color="white", linewidth=1)

plt.colorbar(im, label="Goal sequence frequency (%)")

plt.yticks(
    ticks=np.arange(len(filtered_sequences)),
    labels=[str(seq) for seq in filtered_sequences]
)

plt.xlabel("Training batches (chronological)")
plt.ylabel("Passing sequence (ranked)")
plt.title("Dominant passing-sequence discovery over training")

plt.tight_layout()
plt.show()
