import os
import re

import matplotlib.pyplot as plt
import numpy as np


# Constant definitions
P = [2**i for i in range(6)]
C = ["K", "Vi", "E", "C", "Vr"]
# C = ["K", "Vi", "E", "Z", "C", "Vr"]


# Extract statistics
savgs = {}
wavgs = {}
pattern = re.compile(r"((\d+\r?\n){7})")
for p in P:
    filepath = f"s{p}"
    if os.path.exists(filepath):
        with open(filepath) as f:
            text = f.read()
            groups = [x for x,_ in re.findall(pattern, text)]
            data = np.array([[int(i) for i in x.strip().splitlines()] for x in groups])
            savgs[p] = np.sum(data, axis=0) / len(data)
    else:
        print(f"Could not find {filepath}")

    filepath = f"w{p}"
    if os.path.exists(filepath):
        with open(filepath) as f:
            text = f.read()
            groups = re.findall(pattern, text)
            data = np.array([[int(i) for i in x[0].strip().splitlines()] for x in groups])
            wavgs[p] = np.sum(data, axis=0) / len(data)
    else:
        print(f"Could not find {filepath}")


# Filter Z (runtime is negligible)
for k, v in savgs.items():
    savgs[k] = np.delete(v, 4)
for k, v in wavgs.items():
    wavgs[k] = np.delete(v, 4)


# Create graphs
plt.title("Strong Scaling Runtime")
plt.plot(P, [savgs[p][0] for p in P], label="Actual")
plt.plot(P, [savgs[P[0]][0] / p for p in P], label="Theoretical")
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xlabel("p")
plt.ylabel("Runtime (ms)")
plt.legend()
plt.savefig("strong_runtime.png")
plt.clf()

plt.title("Strong Scaling Speedup")
plt.plot(P, [savgs[P[0]][0] / savgs[p][0] for p in P], label="Actual")
plt.plot(P, P, label="Theoretical")
plt.xlabel("p")
plt.ylabel("Speedup")
plt.legend()
plt.savefig("strong_speedup.png")
plt.clf()

plt.title("Strong Scaling Performance Breakdown")
x = np.arange(len(P))
plt.xticks(x, P)
bars = []
for i,c in enumerate(C):
    heights = [savgs[p][i+1] for p in P]
    bars.append(plt.bar(x, heights, label=c, width=0.5, edgecolor="black"))
for p in range(len(bars[0].patches)):
    patches = [bars[i].patches[p] for i in range(len(bars))]
    patches.sort(key=lambda x: -x.get_height())
    for i, patch in enumerate(patches):
        patch.set_zorder(i)
plt.yscale("log")
plt.xlabel("Ranks")
plt.ylabel("Runtime (ms)")
plt.legend()
plt.savefig("strong_breakdown.png")
plt.clf()

plt.title("Weak Scaling Runtime")
plt.plot(P, [wavgs[p][0] for p in P], label="Actual")
plt.plot(P, [wavgs[P[0]][0] for _ in P], label="Theoretical")
plt.xscale("log", base=2)
plt.yscale("log", base=2)
plt.xlabel("p")
plt.ylabel("Runtime (ms)")
plt.ylim(2**13, 2**15)
plt.legend()
plt.savefig("weak_runtime.png")
plt.clf()

