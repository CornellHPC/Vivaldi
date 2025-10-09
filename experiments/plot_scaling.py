#!/usr/bin/env python3
import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import glob


# Data directory
DATA_DIR = "/global/homes/j/jbellav/m4646/hoon/cpop_results_final/"
#DATA_DIR = "/pscratch/sd/j/jbellav/cpop_results_kdd_mnist/"
#DATA_DIR = "./results"

def parse_data(scaling_type):
    """Parse data files and extract timing information."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    # Get all rank0 time files for this scaling type
    pattern = os.path.join(DATA_DIR, f"{scaling_type}_*_time_*_rank0")
    files = glob.glob(pattern)

    for filepath in files:
        filename = os.path.basename(filepath)

        # Parse filename: {scaling_type}_{gpu_count}_{npoints}_{nfeatures}_{nclusters}_{niters}_1_1_1_2_0_0_{dataset}_{algorithm}_time_{trial}_rank{p}
        parts = filename.split('_')

        gpu_count = int(parts[1])
        nclusters = int(parts[4])
        dataset = parts[12]
        algorithm = parts[13]
        #trial = int(parts[15])
        trial = 2

        # Extract elapsed time from file
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('Elapsed:'):
                    elapsed = float(line.split(':')[1].strip())
                    data[nclusters][dataset][algorithm].append((gpu_count, trial, elapsed))
                    break

    return data

def parse_breakdown_data(scaling_type):
    """Parse timing breakdown data from all ranks."""
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))))

    # Get all time files for this scaling type
    pattern = os.path.join(DATA_DIR, f"{scaling_type}_*_time_*_rank*")
    files = glob.glob(pattern)

    print(f"Total of {len(files)}")

    i = 0
    for filepath in files:
        if i%100==0:
            print(f"Parsing file {i}")
        i += 1
        filename = os.path.basename(filepath)

        # Parse filename
        parts = filename.split('_')

        gpu_count = int(parts[1])
        nclusters = int(parts[4])
        dataset = parts[12]
        algorithm = parts[13]
        #trial = int(parts[15])
        trial = 2
        rank = int(parts[16].replace('rank', ''))


        # Extract timing breakdown from file
        timings = {}
        e_reduce = 0.0
        e_gather = 0.0
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('K:'):
                    timings['K'] = float(line.split(':')[1].strip())
                elif line.startswith('K Redist:'):
                    timings['K Redist'] = float(line.split(':')[1].strip())
                elif line.startswith('E MPI:'):
                    timings['E MPI'] = float(line.split(':')[1].strip())
                elif line.startswith('E Reduce:') and algorithm == '15d':
                    e_reduce = float(line.split(':')[1].strip().split(" ")[0].strip())
                elif line.startswith('E Gather:') and algorithm == '15d':
                    e_gather = float(line.split(':')[1].strip().split(" ")[0].strip())
                elif line.startswith('E SpMM:'):
                    timings['E SpMM'] = float(line.split(':')[1].strip())
                elif line.startswith('C:'):
                    timings['C'] = float(line.split(':')[1].strip())
                elif line.startswith('VR Computation:'):
                    timings['VR Computation'] = float(line.split(':')[1].strip())

        # For 15D algorithm, compute E MPI from E Reduce + E Gather
        if algorithm == '15d':
            timings['E MPI'] = e_reduce + e_gather

        # Store timings for this configuration
        key = (gpu_count, trial)
        if key not in data[nclusters][dataset][algorithm]:
            data[nclusters][dataset][algorithm][key] = defaultdict(list)

        # Append timings from this rank
        for phase, time in timings.items():
            data[nclusters][dataset][algorithm][key][phase].append(time)

    return data

def average_breakdown_trials(data):
    """Average breakdown data across trials, taking max across ranks for each phase."""
    averaged_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))

    for nclusters in data:
        for dataset in data[nclusters]:
            for algorithm in data[nclusters][dataset]:
                # Group by gpu_count
                gpu_groups = defaultdict(lambda: defaultdict(list))

                for (gpu_count, trial), phase_ranks in data[nclusters][dataset][algorithm].items():
                    if trial > 1:  # Skip trial 1
                        # Take max across ranks for each phase
                        for phase, rank_times in phase_ranks.items():
                            max_time = max(rank_times)
                            gpu_groups[gpu_count][phase].append(max_time)

                # Average across trials for each gpu_count and phase
                for gpu_count, phases in gpu_groups.items():
                    for phase, times in phases.items():
                        averaged_data[nclusters][dataset][algorithm][gpu_count][phase] = np.mean(times)

    return averaged_data

def average_trials(data):
    """Average trials for each configuration, skipping trial 1."""
    averaged_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for nclusters in data:
        for dataset in data[nclusters]:
            for algorithm in data[nclusters][dataset]:
                # Group by gpu_count
                gpu_groups = defaultdict(list)
                for gpu_count, trial, elapsed in data[nclusters][dataset][algorithm]:
                    if trial > 1:  # Skip trial 1
                        gpu_groups[gpu_count].append(elapsed)

                # Average for each gpu_count
                for gpu_count, times in gpu_groups.items():
                    averaged_data[nclusters][dataset][algorithm][gpu_count] = np.mean(times)

    return averaged_data

def plot_strong_scaling(output_dir):
    """Generate strong scaling plots."""
    print("Generating strong scaling plots...")

    data = parse_data("strong")
    averaged_data = average_trials(data)

    # Define nicer colors
    #colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749']
    colors = ['purple', 'teal', 'forestgreen', 'darkred']
    names = { '1d': '1D',
              '1dr': 'Hybrid 1D',
              '15d': '1.5D',
              '2d': '2D'
             }

    for nclusters in averaged_data:
        for dataset in averaged_data[nclusters]:
            # Create dataset-specific directory structure
            dataset_dir = os.path.join(output_dir, dataset, 'strong_scaling')
            os.makedirs(dataset_dir, exist_ok=True)

            plt.figure(figsize=(4, 4))

            # Plot each algorithm
            for i, algorithm in enumerate(sorted(averaged_data[nclusters][dataset].keys())):
                gpu_counts = sorted(averaged_data[nclusters][dataset][algorithm].keys())
                times = [averaged_data[nclusters][dataset][algorithm][gc] for gc in gpu_counts]


                plt.plot(gpu_counts, times, marker='o', label=names[algorithm],
                         color=colors[i % len(colors)], linewidth=2, markersize=8,
                         markeredgecolor='black', markerfacecolor="white")

            plt.xscale('log', base=2)
            plt.yscale('log', base=2)
            plt.xticks([4, 16, 64, 256], labels=['4', '16', '64', '256'], fontsize=12)
            plt.xlabel('Number of GPUs', fontsize=14)
            plt.ylabel('Time (ms)', fontsize=14)
            plt.title(f'Strong Scaling - {dataset} - {nclusters} clusters', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)

            # Save plot
            output_filename = os.path.join(dataset_dir, f'strong_{dataset}_nclusters{nclusters}.png')
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {output_filename}")

def plot_weak_scaling(output_dir):
    """Generate weak scaling plots."""
    print("Generating weak scaling plots...")

    data = parse_data("weak")
    averaged_data = average_trials(data)

    # Define nicer colors
    #colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4749']
    colors = ['purple', 'teal', 'forestgreen', 'darkred']
    names = { '1d': '1D',
              '1dr': 'Hybrid 1D',
              '15d': '1.5D',
              '2d': '2D'
             }

    for nclusters in averaged_data:
        for dataset in averaged_data[nclusters]:
            # Create dataset-specific directory structure
            dataset_dir = os.path.join(output_dir, dataset, 'weak_scaling')
            os.makedirs(dataset_dir, exist_ok=True)

            plt.figure(figsize=(4, 4))

            # Plot each algorithm
            for i, algorithm in enumerate(sorted(averaged_data[nclusters][dataset].keys())):
                gpu_counts = sorted(averaged_data[nclusters][dataset][algorithm].keys())
                times = [averaged_data[nclusters][dataset][algorithm][gc] for gc in gpu_counts]

                plt.plot(gpu_counts, times, marker='o', label=names[algorithm],
                         color=colors[i % len(colors)], linewidth=2, markersize=8,
                         markeredgecolor='black', markerfacecolor="white")

            plt.xscale('log', base=2)
            plt.yscale('log', base=2)
            plt.xticks([4, 16, 64, 256], labels=['4', '16', '64', '256'], fontsize=12)
            plt.xlabel('Number of GPUs', fontsize=14)
            plt.ylabel('Time (ms)', fontsize=14)
            plt.title(f'Weak Scaling - {dataset} - {nclusters} clusters', fontsize=16)
            plt.legend(fontsize=12)
            plt.grid(True, alpha=0.3)

            # Save plot
            output_filename = os.path.join(dataset_dir, f'weak_{dataset}_nclusters{nclusters}.png')
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {output_filename}")

def plot_breakdown_bars(output_dir, scaling_type):
    """Generate runtime breakdown bar plots."""
    print(f"Generating {scaling_type} scaling breakdown plots...")

    data = parse_breakdown_data(scaling_type)
    averaged_data = average_breakdown_trials(data)

    # Define phase order and hatches
    phases = ['K', 'E MPI', 'E SpMM', 'C', 'VR Computation']
    phase_hatches = {
        'K': '',
        'E MPI': '\\\\\\',
        'E SpMM': '---',
        'C': 'xxx',
        'VR Computation': '...'
    }
    names = { '1d': '1D',
              '1dr': 'Hybrid 1D',
              '15d': '1.5D',
              '2d': '2D'
             }

    # Define colors for different algorithms
    algorithm_colors = ['cyan', 'orangered', 'lime', 'violet']

    for nclusters in averaged_data:
        for dataset in averaged_data[nclusters]:
            # Create dataset-specific directory structure
            dataset_dir = os.path.join(output_dir, dataset, f'{scaling_type}_scaling_breakdown')
            os.makedirs(dataset_dir, exist_ok=True)

            algorithms = sorted(averaged_data[nclusters][dataset].keys())

            # Get all GPU counts (assuming all algorithms have the same GPU counts)
            gpu_counts = sorted(list(averaged_data[nclusters][dataset][algorithms[0]].keys()))

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))

            # Bar width and positions
            bar_width = 0.2
            num_algorithms = len(algorithms)

            # For each phase, create bars for all algorithms
            for alg_idx, algorithm in enumerate(algorithms):
                # Calculate positions for this algorithm's bars
                positions = [i + alg_idx * bar_width for i in range(len(gpu_counts))]

                # Prepare data for stacked bar plot for this algorithm
                phase_data = {phase: [] for phase in phases}
                for gc in gpu_counts:
                    for phase in phases:
                        phase_data[phase].append(
                            averaged_data[nclusters][dataset][algorithm][gc].get(phase, 0)
                        )

                # Create stacked bars for this algorithm
                bottom = np.zeros(len(gpu_counts))
                for phase_idx, phase in enumerate(phases):
                    # Only add label for first phase to avoid duplicate legend entries
                    label = names[algorithm] if phase_idx == 0 else None
                    ax.bar(positions, phase_data[phase],
                          width=bar_width,
                          label=label,
                          bottom=bottom,
                          color=algorithm_colors[alg_idx % len(algorithm_colors)],
                          hatch=phase_hatches[phase],
                          edgecolor='black',
                          linewidth=0.5)
                    bottom += np.array(phase_data[phase])

            # Add phase labels as a second legend
            from matplotlib.patches import Patch
            phase_handles = [Patch(facecolor='white', edgecolor='black',
                                  hatch=phase_hatches[phase], label=phase)
                            for phase in phases]

            # Customize plot
            ax.set_xticks([i + bar_width * (num_algorithms - 1) / 2 for i in range(len(gpu_counts))])
            ax.set_xticklabels(gpu_counts, fontsize=12)
            ax.set_xlabel('Number of GPUs', fontsize=14)
            ax.set_ylabel('Time (ms)', fontsize=14)
            ax.set_title(f'{scaling_type.capitalize()} Scaling Breakdown - {dataset} - {nclusters} clusters', fontsize=16)

            # Create two legends: one for algorithms, one for phases
            algorithm_legend = ax.legend(loc='upper left', title='Algorithms', fontsize=12)
            ax.add_artist(algorithm_legend)
            ax.legend(handles=phase_handles, loc='upper right', title='Phases', fontsize=12)

            ax.grid(True, alpha=0.3, axis='y')

            # Save plot
            output_filename = os.path.join(
                dataset_dir,
                f'{scaling_type}_breakdown_{dataset}_nclusters{nclusters}.png'
            )
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {output_filename}")

def plot_15d_speedup(output_dir):
    """Generate speedup plots for 1.5D algorithm relative to cpop_data_window baseline."""
    print("Generating 1.5D speedup plots...")

    import pandas as pd

    # Directory containing baseline data
    baseline_dir = os.path.expanduser("~/m4646/hoon/cpop_data_window")

    # Parse 1.5D strong scaling data
    data_15d = parse_data("strong")
    averaged_15d = average_trials(data_15d)

    # Dataset mapping between CSV filenames and result filenames
    dataset_map = {
        'higgs_dataset.csv': 'higgs',
        'kdda_dataset.csv': 'kdd',
        'mnist8m_dataset.csv': 'mnist8m'
    }

    # K values to plot
    k_values = [16, 32, 64]

    # Colors for different k values
    colors = ['teal', 'purple', 'turquoise']

    for csv_file, dataset_name in dataset_map.items():
        csv_path = os.path.join(baseline_dir, csv_file)

        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found, skipping...")
            continue

        # Read baseline data
        df = pd.read_csv(csv_path)

        # Filter for largest block size and desired k values
        max_block_size = df['block_size'].max()
        df_filtered = df[(df['block_size'] == max_block_size) &
                        (df['n_clusters'].isin(k_values))]

        # Create dictionary of baseline times: {k: time_in_seconds}
        baseline_times = {}
        for _, row in df_filtered.iterrows():
            k = int(row['n_clusters'])
            time_sec = row['time (sec.)']
            baseline_times[k] = time_sec

        # Collect speedup data for all k values
        speedup_data = {}  # {k: {gpu_count: speedup}}
        all_gpu_counts = set()

        for k_idx, k in enumerate(k_values):
            # Check if we have data for this k value
            if k not in averaged_15d:
                print(f"  Warning: No data for k={k} in {dataset_name}, skipping...")
                continue

            if dataset_name not in averaged_15d[k]:
                print(f"  Warning: No data for dataset {dataset_name} with k={k}, skipping...")
                continue

            if '15d' not in averaged_15d[k][dataset_name]:
                print(f"  Warning: No 1.5D data for {dataset_name} with k={k}, skipping...")
                continue

            if k not in baseline_times:
                print(f"  Warning: No baseline time for k={k} in {dataset_name}, skipping...")
                continue

            # Get 1.5D times for different GPU counts
            gpu_counts = sorted(averaged_15d[k][dataset_name]['15d'].keys())
            times_15d_ms = [averaged_15d[k][dataset_name]['15d'][gc] for gc in gpu_counts]

            # Convert from milliseconds to seconds
            times_15d_sec = [t / 1000.0 for t in times_15d_ms]

            # Calculate speedup relative to baseline
            baseline_time = baseline_times[k]
            speedups = [baseline_time / t for t in times_15d_sec]

            # Store speedup data
            speedup_data[k] = dict(zip(gpu_counts, speedups))
            all_gpu_counts.update(gpu_counts)

        # Create bar plot
        if speedup_data:
            fig, ax = plt.subplots(figsize=(10, 6))

            gpu_counts_sorted = sorted(all_gpu_counts)
            num_k_values = len(speedup_data)
            bar_width = 0.25

            # Plot bars for each k value
            for k_idx, k in enumerate(sorted(speedup_data.keys())):
                positions = [i + k_idx * bar_width for i in range(len(gpu_counts_sorted))]
                speedups = [speedup_data[k].get(gc, 0) for gc in gpu_counts_sorted]

                bars = ax.bar(positions, speedups, bar_width,
                             label=f'k={k}',
                             color=colors[k_idx % len(colors)],
                             edgecolor='black',
                             linewidth=1)

                # Annotate bars with speedup values
                for bar, speedup in zip(bars, speedups):
                    if speedup > 0:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{speedup:.1f}x',
                               ha='center', va='bottom', fontsize=8)

            # Format plot
            ax.set_xticks([i + bar_width * (num_k_values - 1) / 2 for i in range(len(gpu_counts_sorted))])
            ax.set_xticklabels(gpu_counts_sorted, fontsize=12)
            ax.set_xlabel('Number of GPUs', fontsize=14)
            ax.set_ylabel('Speedup vs. Baseline', fontsize=14)
            ax.set_title(f'1.5D Algorithm Speedup - {dataset_name}', fontsize=16)
            ax.legend(fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            # Create dataset-specific directory
            dataset_dir = os.path.join(output_dir, dataset_name, 'speedup')
            os.makedirs(dataset_dir, exist_ok=True)

            # Save plot
            output_filename = os.path.join(dataset_dir, f'speedup_15d_{dataset_name}.png')
            plt.savefig(output_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved {output_filename}")
        else:
            print(f"  No data to plot for {dataset_name}")

def main():
    parser = argparse.ArgumentParser(description='Generate scaling plots from CPOP results')
    parser.add_argument('--scaling-type', choices=['strong', 'weak', 'both'], required=True,
                        help='Type of scaling plot to generate')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save plots')
    parser.add_argument('--breakdown', action='store_true',
                        help='Generate runtime breakdown plots')
    parser.add_argument('--speedup', action='store_true',
                        help='Generate 1.5D speedup plots vs. cpop_data_window baseline')

    args = parser.parse_args()

    if args.speedup:
        # Generate 1.5D speedup plots
        plot_15d_speedup(args.output_dir)
    elif args.breakdown:
        # Generate breakdown plots
        if args.scaling_type in ['strong', 'both']:
            plot_breakdown_bars(args.output_dir, 'strong')

        if args.scaling_type in ['weak', 'both']:
            plot_breakdown_bars(args.output_dir, 'weak')
    else:
        # Generate regular scaling plots
        if args.scaling_type in ['strong', 'both']:
            plot_strong_scaling(args.output_dir)

        if args.scaling_type in ['weak', 'both']:
            plot_weak_scaling(args.output_dir)

    print("All plots generated successfully!")

if __name__ == '__main__':
    main()
