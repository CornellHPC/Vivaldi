import os
import re

import matplotlib.pyplot as plt
import numpy as np


# Constants
SCALING_HIGHEST_POWER = 6
N_TRIALS = 5


# Constant definitions
P = [2**i for i in range(SCALING_HIGHEST_POWER+1)]
C = ["K", "VI", "E", "Z", "C MPI", "C Computation", "VR MPI", "VR Computation"]


def get_scaling_data(scaling_type):
    # ``scaling_type`` is one of "s" (strong) or "w" (weak)
    M = np.zeros((len(P), N_TRIALS))
    for i, p in enumerate(P):
        for trial in range(N_TRIALS):
            trial_in_fname = trial + 1
            fpath = f"../basic_time/{scaling_type}{p}_{trial_in_fname}"
            if not os.path.exists(fpath):
                print(f"Could not find {fpath}")
                M[i, trial] = np.nan
                continue
                # exit(1)
            with open(fpath) as f:
                scaling = f.readline()
                M[i, trial] = float(scaling)
    # take median and mean and discard NaN values
    median_over_trials = np.nanmedian(M, axis=1)
    averages_over_trials = np.nanmean(M, axis=1)
    return median_over_trials, averages_over_trials


def get_breakdown_data(scaling_type):
    # ``scaling_type`` is one of "s" (strong) or "w" (weak)
    M = np.zeros((len(P), len(C), N_TRIALS))
    for i, p in enumerate(P):
        for trial in range(N_TRIALS):
            trial_in_fname = trial + 1
            fpath = f"../breakdown_time/{scaling_type}{p}_{trial_in_fname}"
            if not os.path.exists(fpath):
                print(f"Could not find {fpath}")
                for j in range(len(C)):
                    M[i, j, trial] = np.nan
                continue
                # exit(1)
            with open(fpath) as f:
                for line in f.read().split("\n"):
                    for j, c in enumerate(C):
                        if line.startswith(c + ": "):
                            M[i, j, trial] = int(line[len(c + ": ") :])
    # take median and mean and discard NaN values
    median_over_trials = np.nanmedian(M, axis=2)
    averages_over_trials = np.nanmean(M, axis=2)
    return median_over_trials, averages_over_trials


# Create basic strong scaling graph
def create_strong_runtime():
    median_over_trials, averages_over_trials = get_scaling_data("s")
    theoretical_scaling = averages_over_trials[0] / P * P[0]
    plt.title(f"Strong Scaling Average Runtime Over {N_TRIALS} Trials")
    plt.plot(P, averages_over_trials, label="Actual", marker="o")
    plt.plot(P, theoretical_scaling, label="Theoretical", marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("GPU count")
    plt.ylabel("Runtime (ms)")
    plt.legend()
    plt.savefig("strong_avg_runtime.png")
    print("Generate strong_avg_runtime.png")
    plt.clf()

    theoretical_scaling = median_over_trials[0] / P * P[0]
    plt.title(f"Strong Scaling Median Runtime Over {N_TRIALS} Trials")
    plt.plot(P, median_over_trials, label="Actual", marker="o")
    plt.plot(P, theoretical_scaling, label="Theoretical", marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("GPU count")
    plt.ylabel("Runtime (ms)")
    plt.legend()
    plt.savefig("strong_med_runtime.png")
    print("Generate strong_med_runtime.png")
    plt.clf()


def create_strong_speedup():
    median_over_trials, averages_over_trials = get_scaling_data("s")
    actual_strong_speedup = averages_over_trials[0] / averages_over_trials
    theoretical_speedup = [p / P[0] for p in P]
    plt.title(f"Strong Scaling Average Speedup Over {N_TRIALS} Trials")
    plt.plot(P, actual_strong_speedup, label="Actual", marker="o")
    plt.plot(P, theoretical_speedup, label="Theoretical", marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("GPU count")
    plt.ylabel("Speedup")
    plt.legend()
    plt.savefig("strong_avg_speedup.png")
    print("Generate strong_avg_speedup.png")
    plt.clf()

    actual_strong_speedup = median_over_trials[0] / median_over_trials
    plt.title(f"Strong Scaling Median Speedup Over {N_TRIALS} Trials")
    plt.plot(P, actual_strong_speedup, label="Actual", marker="o")
    plt.plot(P, theoretical_speedup, label="Theoretical", marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("GPU count")
    plt.ylabel("Speedup")
    plt.legend
    plt.savefig("strong_med_speedup.png")
    print("Generate strong_med_speedup.png")
    plt.clf()


def create_weak_runtime():
    median_over_trials, averages_over_trials = get_scaling_data("w")
    theoretical_scaling = np.array([averages_over_trials[0] for _ in P])
    plt.title(f"Weak Scaling Average Runtime Over {N_TRIALS} Trials")
    plt.plot(P, averages_over_trials, label="Actual", marker="o")
    plt.plot(P, theoretical_scaling, label="Theoretical", marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("GPU count")
    plt.ylabel("Runtime (ms)")
    plt.ylim(2**11, 2**14)
    plt.legend()
    plt.savefig("weak_avg_runtime.png")
    print("Generate weak_avg_runtime.png")
    plt.clf()

    theoretical_scaling = np.array([median_over_trials[0] for _ in P])
    plt.title(f"Weak Scaling Median Runtime Over {N_TRIALS} Trials")
    plt.plot(P, median_over_trials, label="Actual", marker="o")
    plt.plot(P, theoretical_scaling, label="Theoretical", marker="o")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("GPU count")
    plt.ylabel("Runtime (ms)")
    plt.ylim(2**11, 2**14)
    plt.legend()
    plt.savefig("weak_med_runtime.png")
    print("Generate weak_med_runtime.png")
    plt.clf()


# Create strong scaling breakdown graph
def create_strong_breakdown():
    median_over_trials, averages_over_trials = get_breakdown_data("s")
    plt.title(f"Strong Scaling Breakdown (Average Runtime Over {N_TRIALS} Trials)")
    x = np.arange(len(P))
    plt.xticks(x, P)
    bars = []
    for i, c in enumerate(C):
        heights = averages_over_trials[:, i]
        bars.append(plt.bar(x, heights, label=c, width=0.5, edgecolor="black"))
    for p in range(len(bars[0].patches)):
        patches = [bars[i].patches[p] for i in range(len(bars))]
        patches.sort(key=lambda x: -x.get_height())
        for i, patch in enumerate(patches):
            patch.set_zorder(i)
    plt.xlabel("GPU count")
    plt.ylabel("Runtime (ms)")
    plt.legend()
    plt.savefig("strong_avg_breakdown.png")
    print("Generate strong_avg_breakdown.png")
    plt.clf()

    plt.title(f"Strong Scaling Breakdown (Median Runtime Over {N_TRIALS} Trials)")
    x = np.arange(len(P))
    plt.xticks(x, P)
    bars = []
    for i, c in enumerate(C):
        heights = median_over_trials[:, i]
        bars.append(plt.bar(x, heights, label=c, width=0.5, edgecolor="black"))
    for p in range(len(bars[0].patches)):
        patches = [bars[i].patches[p] for i in range(len(bars))]
        patches.sort(key=lambda x: -x.get_height())
        for i, patch in enumerate(patches):
            patch.set_zorder(i)
    plt.xlabel("GPU count")
    plt.ylabel("Runtime (ms)")
    plt.legend()
    plt.savefig("strong_med_breakdown.png")
    print("Generate strong_med_breakdown.png")
    plt.clf()


# Create strong scaling breakdown graph
def create_weak_breakdown():
    median_over_trials, averages_over_trials = get_breakdown_data("w")
    plt.title(f"Weak Scaling Breakdown (Average Runtime Over {N_TRIALS} Trials)")
    x = np.arange(len(P))
    plt.xticks(x, P)
    bars = []
    for i, c in enumerate(C):
        heights = averages_over_trials[:, i]
        bars.append(plt.bar(x, heights, label=c, width=0.5, edgecolor="black"))
    for p in range(len(bars[0].patches)):
        patches = [bars[i].patches[p] for i in range(len(bars))]
        patches.sort(key=lambda x: -x.get_height())
        for i, patch in enumerate(patches):
            patch.set_zorder(i)
    plt.xlabel("GPU count")
    plt.ylabel("Runtime (ms)")
    plt.legend()
    plt.savefig("weak_avg_breakdown.png")
    print("Generate weak_avg_breakdown.png")
    plt.clf()

    plt.title(f"Weak Scaling Breakdown (Median Runtime Over {N_TRIALS} Trials)")
    x = np.arange(len(P))
    plt.xticks(x, P)
    bars = []
    for i, c in enumerate(C):
        heights = median_over_trials[:, i]
        bars.append(plt.bar(x, heights, label=c, width=0.5, edgecolor="black"))
    for p in range(len(bars[0].patches)):
        patches = [bars[i].patches[p] for i in range(len(bars))]
        patches.sort(key=lambda x: -x.get_height())
        for i, patch in enumerate(patches):
            patch.set_zorder(i)
    plt.xlabel("GPU count")
    plt.ylabel("Runtime (ms)")
    plt.legend()
    plt.savefig("weak_med_breakdown.png")
    print("Generate weak_med_breakdown.png")
    plt.clf()


if __name__ == "__main__":
    create_strong_runtime()
    create_strong_speedup()
    create_strong_breakdown()
    create_weak_runtime()
    create_weak_breakdown()
    print("Graphs generated successfully")
