import torch

records = torch.load("analysis/...pt")
all_sequences = set()
for r in records:
    all_sequences.update(r["sequence_percentages"].keys())

records = sorted(records, key=lambda r: (r["iteration"], r["batch"]))

heatmap = {
    seq: [] for seq in all_sequences
}

for r in records:
    for seq in all_sequences:
        heatmap[seq].append(
            r["sequence_percentages"].get(seq, 0.0)
        )