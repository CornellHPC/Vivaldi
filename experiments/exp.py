import urllib.request, bz2, lzma, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker


DATASETS = [
    {
        "bin_fname": "data/poker.t.bin",
        "txt_fname": "data/poker.t.txt",
        "zip_fname": "data/poker.t.bz2",
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/poker.t.bz2",
        "name": "poker",
        "m": "1000000",
        "d": "10",
        "k": "10",
    },
    {
        "bin_fname": "data/HIGGS.bin",
        "txt_fname": "data/HIGGS.txt",
        "zip_fname": "data/HIGGS.xz",
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.xz",
        "name": "higgs",
        "m": "11000000",
        "d": "28",
        "k": "2",
    },
    {
        "bin_fname": "data/mnist8m.scale.bin",
        "txt_fname": "data/mnist8m.scale.txt",
        "zip_fname": "data/mnist8m.scale.xz",
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.scale.xz",
        "name": "mnist8m",
        "m": "8100000",
        "d": "784",
        "k": "10",
    },
]

RANDOM_DATASET = {
    "bin_fname": "data/rand.bin",
    "name": "rand",
    "m": "1024000",
    "d": "1024",
}

MAX_NUM_POINTS = 1200000 ## one million points limit for basically everything

SCALING_HIGHEST_POWER = 6 ## for graph generation
N_TRIALS = 5  ## number of trials for each experiment
P = [4, 8, 16, 32, 64, 128]  # number of GPUs (must be divisible by 4)
C = ["K", "VI", "E", "Z", "C MPI", "C Computation", "VR MPI", "VR Computation"]

def request_p_prefix(p, nodes, log_dir, s_name):
    if p >= 256:
        timestamp = "04:00:00"
    elif p >= 128:
        timestamp = "03:00:00"
    elif p >= 32:
        timestamp = "02:00:00"
    else:
        timestamp = "01:30:00"

    return f"""#!/bin/bash
#SBATCH --nodes={nodes}
#SBATCH --gpus={p}
#SBATCH --time={timestamp}
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output={log_dir}/{s_name}_out"""


test_counter = 0


def run_5_trials(
    f,
    input_dataset_path,
    results_dir,
    s_name,
    nodes,
    p,
    m,
    d,
    k,
    niter,
    sparse,
    gamma,
    c,
    r,
    convergence,
):
    global test_counter
    test_counter += 1
    print("Added test", test_counter)
    main_args = f"-i {input_dataset_path} -m {m} -n {d} --niter {niter} --sparse {sparse} --gamma {gamma} --c {c} --r {r} --convergence {convergence} -k {k}"
    log_args = (
        f"-o {results_dir}/{s_name}_assignments --benchmark {results_dir}/{s_name}_time_$i"
    )
    n_trials = 5
    f.write(f'echo "Running with args {main_args}"\n')
    f.write(f'echo ""\n')
    f.write(f"for i in $(seq 1 {n_trials}); do\n")
    f.write(f'  echo "Trial $i"\n')
    f.write(
        f"  srun -N {nodes} --ntasks-per-node {p//nodes} --cpus-per-task 32 --cpu-bind cores -G {p} $EXE_PATH {main_args} {log_args}\n"
    )
    f.write(f'  echo ""\n')
    f.write(f"done\n\n")
    


def create_file_text(
    p,
    unique_id,
    experiments_dir="$PWD",
):
    if p % 4 != 0:
        raise ValueError("Number of nodes must be divisible by 4")
    nodes = p // 4
    suffix = 0
    while os.path.exists(f"exp_{p}_{suffix}.sh"):
        suffix += 1
    s_name = f"exp_{unique_id}_{p}_{suffix}"
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    scripts_dir = "scripts"
    os.makedirs(scripts_dir, exist_ok=True)
    bash_file = os.path.join(scripts_dir, f"{s_name}.sh")

    with open(bash_file, "w") as f:
        f.write(request_p_prefix(p, nodes, log_dir, s_name) + "\n")
        f.write("export DVS_MAXNODES=1__\n")
        f.write(
            f'export EXE_PATH="{experiments_dir}/../build/device_wrapper {experiments_dir}/../build/main"\n'
        )
        f.write("module load cudatoolkit/12.2\n")
        gamma = 1  # gamma fixed at 1
        c = 1  # c fixed at 1
        r = 2  # r fixed at 2 (quadratic kernel)
        basic = True  # todo (all): this won't do breakdown
        for input_dataset in DATASETS:
            input_dataset_path = input_dataset["bin_fname"]
            input_dataset_name = input_dataset["name"]

            # strong scaling
            d = input_dataset["d"]
            niter = 100
            convergence = 0
            m = 128000  # experimentally decided that 128k points fit on 4 GPUs
            for k in [2, 5, 10, 50, 100]:
                # 32 based on experiments
                sparse = int(k > 32)
                run_5_trials(
                    f,
                    input_dataset_path,
                    results_dir,
                    f"{unique_id}_s_{p}_{m}_{d}_{k}_{niter}_{sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{input_dataset_name}",
                    nodes,
                    p,
                    m,
                    d,
                    k,
                    niter,
                    sparse,
                    gamma,
                    c,
                    r,
                    convergence,
                )
            # variant weak scaling
            d = input_dataset["d"]
            for k in [2, 5, 10, 50, 100]:
                # weak scaling number of points
                m = min(int(64000*np.sqrt(p)), int(input_dataset["m"]), MAX_NUM_POINTS)
                m -= m % p
                # 32 based on experiments
                sparse = int(k > 32)
                niter = 100
                convergence = 0
                run_5_trials(
                    f,
                    input_dataset_path,
                    results_dir,
                    f"{unique_id}_w_{p}_{m}_{d}_{k}_{niter}_{sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{input_dataset_name}",
                    nodes,
                    p,
                    m,
                    d,
                    k,
                    niter,
                    sparse,
                    gamma,
                    c,
                    r,
                    convergence,
                )
                niter = 1000
                convergence = 1
                run_5_trials(
                    f,
                    input_dataset_path,
                    results_dir,
                    f"{unique_id}_wc_{p}_{m}_{d}_{k}_{niter}_{sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{input_dataset_name}",
                    nodes,
                    p,
                    m,
                    d,
                    k,
                    niter,
                    sparse,
                    gamma,
                    c,
                    r,
                    convergence,
                )
                convergence = 2
                run_5_trials(
                    f,
                    input_dataset_path,
                    results_dir,
                    f"{unique_id}_wce_{p}_{m}_{d}_{k}_{niter}_{sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{input_dataset_name}",
                    nodes,
                    p,
                    m,
                    d,
                    k,
                    niter,
                    sparse,
                    gamma,
                    c,
                    r,
                    convergence,
                )
            # combblas (todo)
        # proper weak scaling
        input_dataset = RANDOM_DATASET
        input_dataset_path = input_dataset["bin_fname"]
        input_dataset_name = input_dataset["name"]
        d = 4*p
        niter = 100
        convergence = 0
        for k in [2, 5, 10, 50, 100]:
            # weak scaling number of points
            m = min(int(64000*np.sqrt(p)), int(input_dataset["m"]), MAX_NUM_POINTS)
            m -= m % p
            # 32 based on experiments
            sparse = int(k > 32)
            run_5_trials(
                f,
                input_dataset_path,
                results_dir,
                f"{unique_id}_wp_{p}_{m}_{d}_{k}_{niter}_{sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{input_dataset_name}",
                nodes,
                p,
                m,
                d,
                k,
                niter,
                sparse,
                gamma,
                c,
                r,
                convergence,
            )
        # mode test
        if p == 4:
            for m in {16000, 32000, 64000}:
                for k in [10, 20, 30, 40, 50, 60]:
                    sparse = 1
                    run_5_trials(
                        f,
                        input_dataset_path,
                        results_dir,
                        f"{unique_id}_m_1_{m}_{d}_{k}_{niter}_{sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{input_dataset_name}",
                        1,
                        1,
                        m,
                        d,
                        k,
                        niter,
                        sparse,
                        gamma,
                        c,
                        r,
                        convergence,
                    )
                    sparse = 0
                    run_5_trials(
                        f,
                        input_dataset_path,
                        results_dir,
                        f"{unique_id}_m_1_{m}_{d}_{k}_{niter}_{sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{input_dataset_name}",
                        1,
                        1,
                        m,
                        d,
                        k,
                        niter,
                        sparse,
                        gamma,
                        c,
                        r,
                        convergence,
                    )
        f.write("echo 'Done!'\n")


def create_random(low=0, high=100):
    filepath = RANDOM_DATASET["bin_fname"]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    m = int(RANDOM_DATASET["m"])
    d = int(RANDOM_DATASET["d"])
    data = np.random.uniform(low, high, m*d).astype(np.float32)
    data.tofile(filepath)


# Functions related to downloading, extracting, and preparing the datasets
def progress_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r[%s] %d%%" % ('#' * (percent // 2), percent))
    sys.stdout.flush()

def print_file_size(url):
  req = urllib.request.Request(url, method='HEAD')
  with urllib.request.urlopen(req) as response:
    file_size = response.getheader('Content-Length')
    if file_size:
      file_size = int(file_size)
      print(f"File size: {file_size} bytes")
    else:
      print("Could not retrieve file size")

def download(dataset):
    url = dataset["url"]
    name = dataset["name"]
    tar_file = dataset["zip_fname"]
    if os.path.exists(tar_file):
        print(f"File {tar_file} already exists. Skipping download.")
        return
    print(f"Downloading dataset {name} from {url} to {tar_file}...")
    print_file_size(url)
    os.makedirs(os.path.dirname(tar_file), exist_ok=True)
    urllib.request.urlretrieve(url, tar_file, reporthook=progress_hook)
    print()
    print("Download done!")
    
def extract(dataset):
    name = dataset["name"]
    tar_file = dataset["zip_fname"]
    txt_out = dataset["txt_fname"]
    if os.path.exists(txt_out):
        print(f"File {txt_out} already exists. Skipping extraction.")
        return
    print(f"Extracting dataset {name} from {tar_file} to {txt_out}...")
    _, extension = os.path.splitext(tar_file)
    if extension == ".bz2":
        with open(tar_file, "rb") as f:
            data = bz2.decompress(f.read())
        with open(txt_out, "wb+") as f:
            f.write(data)
    elif extension == ".xz":
        with open(tar_file, "rb") as f:
            data = lzma.decompress(f.read())
        with open(txt_out, "wb+") as f:
            f.write(data)
    print(f"Extraction done! Deleting tar file {tar_file}...")
    os.remove(tar_file)
    print(f"Deleted tar file {tar_file}...")

def prepare(dataset):
    name = dataset["name"]
    txt_in = dataset["txt_fname"]
    bin_out = dataset["bin_fname"]
    d = int(dataset["d"]) # number of features
    if os.path.exists(bin_out):
        print(f"Binary file {bin_out} already exists. Skipping preparation.")
        return
    
    print(f"Preparing dataset {name} from {txt_in} to {bin_out}...")
    nbytes = os.path.getsize(txt_in)
    print(f"Byte count of {txt_in} is {nbytes} bytes...")
    lines_read = 0
    with open(bin_out, "wb+") as out:
        with open(txt_in, 'r') as file:
            while True:
                line = file.readline()
                lines_read += 1
                if not line:
                    break
                progress_hook(file.tell(), 1, nbytes)
                features = line.rstrip().split(' ')
                features.pop(0) # label is unused
                features_vector = np.zeros(d, dtype=np.float32)
                for feature in features:
                    index, value = feature.split(':')
                    index = int(index) - 1
                    value = np.float32(value)
                    if index < 0 or index >= d:
                        print(f"Index {index} out of bounds for d={d}")
                        continue
                features_vector[index] = value
                out.write(features_vector.tobytes())
                if lines_read >= MAX_NUM_POINTS:
                    print(f"Reached maximum number of points: {MAX_NUM_POINTS}...")
                    break
    print()
    print(f"Wrote binary data to {bin_out}...")

def read_data(formatted_txt_file) -> np.ndarray:
  path = formatted_txt_file
  data = []
  labels = []

  # this currently assumes no missing data, 
  # all formatted with exactly one space between features
  with open(path, 'r') as file:
    rows = [line.rstrip() for line in file]
    for features in rows:
      features = features.split(' ')

      # extract label, which is unused
      label = features.pop(0)
      labels.append(label)

      features = np.float32(
          list(map(lambda feature: float(feature.split(':')[1]), features)))
      data.append(features)

  return np.array(data)


def get_scaling_data(unique_id, scaling_type, p, m, d, k, niter, sparse, gamma, c, r, convergence, basic, input_dataset_name):
    # ``scaling_type`` is one of "s" (strong) or "w" (weak)
    scaling_data = {}
    for trial in range(N_TRIALS):
        trial_in_fname = trial + 1
        fpath = f"results/{unique_id}_{scaling_type}_{p}_{m}_{d}_{k}_{niter}_{sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{input_dataset_name}_time_{trial_in_fname}"
        # print(f"Looking for {fpath}...")
        if not os.path.exists(fpath):
            print(f"Could not find {fpath}")
            continue
        with open(fpath, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                k_, v_ = line.split(": ")
                if not k_ in scaling_data:
                    scaling_data[k_] = np.zeros(N_TRIALS)
                scaling_data[k_][trial] = int(v_)
    return scaling_data


def construct_graphs():
    os.makedirs("graphs", exist_ok=True)
    unique_id = ""
    
    # construct strong scaling graph
    fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    niter = 100
    gamma = 1
    c = 1
    r = 2
    basic = True
    for idx, input_dataset in enumerate(DATASETS):
        input_dataset_name = input_dataset["name"]
        ax = axs[idx]
        d = input_dataset["d"]
        convergence = 0
        m = 128000  # experimentally decided that 128k points fit on 4 GPUs
        for k_idx, k in enumerate([2, 5, 10, 50, 100]):
            color = plt.cm.viridis(k_idx / 4)  # Use a colormap for different k values
            dataset_symbol = ["o", "s", "D", "^", "v"][k_idx]  # Different marker for each k value
            y = []
            sparse = int(k > 32)
            for p in P:
                scaling_data = get_scaling_data(
                    unique_id,
                    "s",
                    p,
                    m,
                    d,
                    k,
                    niter,
                    sparse,
                    gamma,
                    c,
                    r,
                    convergence,
                    basic,
                    input_dataset_name
                )
                y.append(np.average(scaling_data["Elapsed"]))
            ax.plot(P, y, label=f"k={k}", marker=dataset_symbol, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{input_dataset_name.capitalize()} Dataset")
        ax.set_xlabel("Number of GPUs (P)")
        ax.set_xticks(P)
        ax.minorticks_off()
        ax.set_xticklabels(P)
        if idx == 0:
            ax.set_ylabel("Average Time (ms)")
        ax.legend(loc="upper right")
    # plt.suptitle("Strong Scaling Across Datasets")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"graphs/strong_scaling_subfigures.png")
    
    # construct strong scaling breakdown graph
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    niter = 100
    gamma = 1
    c = 1
    r = 2
    basic = True

    for scaling_type in ["s", "w"]:
        bar_width = 0.1
        x_positions = []
        x_labels = []
        bar_offset = 0
        ax = axs[0] if scaling_type == "s" else axs[1]
        scaling_type_name = "Strong" if scaling_type == "s" else "Weak"
        ax.set_title(f"MNIST8M {scaling_type_name} Scaling Breakdown")
        for dataset_idx, input_dataset in enumerate(DATASETS):
            input_dataset_name = input_dataset["name"]
            if input_dataset_name != "mnist8m":
                continue
            d = input_dataset["d"]
            convergence = 0
            strong_m = 128000  # experimentally decided that 128k points fit on 4 GPUs
            # for k_idx, k in enumerate([2, 5, 10, 50, 100]):
            for k_idx, k in enumerate([10, 100]):
                sparse = int(k > 32)
                for p_idx, p in enumerate(P):
                    weak_m = min(int(64000*np.sqrt(p)), int(input_dataset["m"]), MAX_NUM_POINTS)
                    m = strong_m if scaling_type == "s" else weak_m
                    strong_scaling_data = get_scaling_data(
                        unique_id,
                        scaling_type,
                        p,
                        m,
                        d,
                        k,
                        niter,
                        sparse,
                        gamma,
                        c,
                        r,
                        convergence,
                        basic,
                        input_dataset_name
                    )
                    # Prepare x-axis labels and positions
                    x_label = f"P={p}\nK={k}" if p_idx == 0 else f"P={p}"
                    x_positions.append(bar_offset)
                    x_labels.append(x_label)

                    # Prepare stacked bar data
                    bottom = 0
                    running_time = 0
                    for key, values in strong_scaling_data.items():
                        # if key not in C:
                        #     continue
                        # if (key not in ["K", "E", "VR MPI"]) and (np.average(values) / np.average(strong_scaling_data["Elapsed"]) > 0.02):
                        #     print("Step", key, "took", np.average(values) / np.average(strong_scaling_data["Elapsed"]), "of total time")
                        #     # if this doesn't print then only K, E, and VR MPI took more than 2% of the time
                        useful_keys = ["K", "E", "VR MPI"]
                        if key not in useful_keys:
                            continue
                        color = plt.cm.plasma(useful_keys.index(key) / len(useful_keys))  # Use a colormap for different routines
                        avg_time = np.average(values)
                        running_time += avg_time
                        label = key if bar_offset == 0 else ""
                        if label == "K":
                            label = "Distributed GeMM"
                        elif label == "E":
                            label = "Local SpMM / Local GeMM"
                        elif label == "VR MPI":
                            label = "Assignments Gathering"
                        ax.bar(bar_offset, avg_time, bar_width, bottom=bottom, label=label, color=color)
                        bottom += avg_time
                    bar_offset += bar_width * 1.2
                bar_offset += bar_width * 1.3
        ax.set_yscale("log")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.minorticks_off()
        if scaling_type == "s":
            ax.set_ylabel("Average Time (ms)")
            ax.legend(loc="upper left", title="Routines")
    plt.tight_layout()
    # plt.subplots_adjust(hspace=0, wspace=0, bottom=0.2, left=0)
    plt.savefig(f"graphs/scaling_breakdown.png")
    
    # construct weak scaling graph
    fig, axs = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    niter = 100
    gamma = 1
    c = 1
    r = 2
    basic = True
    for idx, input_dataset in enumerate(DATASETS):
        input_dataset_name = input_dataset["name"]
        ax = axs[idx]
        d = input_dataset["d"]
        convergence = 0
        for k_idx, k in enumerate([2, 5, 10, 50, 100]):
            color = plt.cm.viridis(k_idx / 4)  # Use a colormap for different k values
            dataset_symbol = ["o", "s", "D", "^", "v"][k_idx]  # Different marker for each k value
            y = []
            sparse = int(k > 32)
            for p in P:
                m = min(int(64000*np.sqrt(p)), int(input_dataset["m"]), MAX_NUM_POINTS)
                scaling_data = get_scaling_data(
                    unique_id,
                    "w",
                    p,
                    m,
                    d,
                    k,
                    niter,
                    sparse,
                    gamma,
                    c,
                    r,
                    convergence,
                    basic,
                    input_dataset_name
                )
                y.append(np.average(scaling_data["Elapsed"]))
            ax.plot(P, y, label=f"k={k}", marker=dataset_symbol, color=color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{input_dataset_name.capitalize()} Dataset")
        ax.set_xlabel("Number of GPUs (P)")
        ax.set_xticks(P)
        ax.minorticks_off()
        ax.set_xticklabels(P)
        if idx == 0:
            ax.set_ylabel("Average Time (ms)")
        ax.legend(loc="upper right")
    # plt.suptitle("Strong Scaling Across Datasets")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"graphs/weak_scaling_subfigures.png")


# remember that D is the number of features and has to be correct for each dataset otherwise
# the MPI file read will be messed up
# but we can vary N and K for each experiment
# "convergence", "sparse", and "basic" are just flags
#   "convergence" is a flag for process-exclusion-based-convergence checking
#   "sparse" is the flag to use sparse V matrix
#   "basic" is the flag to use basic trial, otherwise it is a benchmark trial (but
#     the benchmark trial requires building with BASIC=0 and is non-negligibly slower due to finer-grained timing)

# example:
# for p in [4, 8, 16, 32, 64, 128, 256]:
#     create_file_text(p, "")

if __name__ == "__main__":
    legal = ["create_scripts", "create_random", "download", "extract", "prepare", "graphs"]
    usage_legal = " | ".join(legal)
    if len(sys.argv) < 2:
        print(f"Usage: python exp.py [{usage_legal}]")
        sys.exit(1)
    action = sys.argv[1]
    if action not in legal:
        print(f"Invalid action. Must be one of {usage_legal}")
        sys.exit(1)
    if action == "create_scripts":
        for p in [4, 8, 16, 32, 64, 128, 256]:
            create_file_text(p, "")
        print("Generated scripts in experiments/scripts/ directory.")
    if action == "create_random":
        create_random()
        print(f"Created random dataset at {RANDOM_DATASET['bin_fname']}.")
    if action == "download":
        for dataset in DATASETS:
            download(dataset)
        print("Generated zipped datasets in experiments/data/ directory.")
    if action == "extract":
        for dataset in DATASETS:
            extract(dataset)
        print("Generated LIBSVM-formatted textfile datasets in experiments/data/ directory.")
    if action == "prepare":
        for dataset in DATASETS:
            prepare(dataset)
        print("Generated binary datasets in experiments/data/ directory... ready for use!")
    if action == "graphs":
        construct_graphs()
        print("Generated graphs in experiments/graphs/ directory.")
