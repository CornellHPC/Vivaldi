import os
import re

import matplotlib.pyplot as plt
import numpy as np


P = [2**i for i in range(6)]
C = ["K", "Vi", "E", "Z", "C", "Vr"]


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


plt.title("Strong Scaling Runtime")
plt.plot(P, [savgs[p][0] for p in P], label="Actual")
plt.plot(P, [savgs[P[0]][0] / p for p in P], label="Theoretical")
plt.xlabel("p")
plt.ylabel("Runtime (ms)")
plt.legend()
plt.savefig("strong_runtime.png")
plt.clf()

plt.title("Strong Scaling Speedup")
plt.plot(P, [savgs[P[0]][0] / savgs[p][0] for p in P], label="Actual")
plt.plot(P, P, label="Theoretical")
plt.xlabel("p")
plt.ylabel("Runtime (ms)")
plt.legend()
plt.savefig("strong_speedup.png")
plt.clf()

plt.title("Strong Scaling Performance Breakdown")
x = np.arange(len(C))
plt.xticks(x, C)
c_width = 0.5
bar_width = 0.1
shift = (bar_width*(len(C)-1)) / 2
for i,p in enumerate(P):
    plt.bar(x+bar_width*i-shift, savgs[p][1:], width=bar_width, label=f"p={p}")
plt.yscale("log")
plt.xlabel("Operation")
plt.ylabel("Runtime (ms)")
plt.legend()
plt.savefig("strong_breakdown.png")
plt.clf()

