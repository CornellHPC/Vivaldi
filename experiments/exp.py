import urllib.request, bz2, datetime, lzma, math, os, re, stat, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker

# Color map for graph
CMAP = plt.cm.viridis
# Markers for graph
MARKERS = ["o", "s", "D", "^", "v", "p"]

ALGS = ["1d", "1dr", "15d", "2d"]

DATASETS = [
    {
        "bin_fname": "data/susy.bin",
        "txt_fname": "data/susy.txt",
        "zip_fname": "data/susy.xz",
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/SUSY.xz",
        "name": "susy",
        "label": "Susy",
        "m": 1600000,
        "d": 18,
        "k": 2,
    },
    {
        "bin_fname": "data/HIGGS.bin",
        "txt_fname": "data/HIGGS.txt",
        "zip_fname": "data/HIGGS.xz",
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/HIGGS.xz",
        "name": "higgs",
        "label": "HIGGS",
        "m": 1600000,
        "d": 28,
        "k": 2,
    },
    {
        "bin_fname": "data/mnist8m.scale.bin",
        "txt_fname": "data/mnist8m.scale.txt",
        "zip_fname": "data/mnist8m.scale.xz",
        "url": "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.scale.xz",
        "name": "mnist8m",
        "label": "MNIST8m",
        "m": 1600000,
        "d": 784,
        "k": 10,
    },
]

# TODO: Use remaining datasetse
DATASETS = [DATASETS[-1]]

RANDOM_DATASET = {
    "bin_fname": "data/rand.bin",
    "name": "rand",
    "label": "Synthetic",
    "m": 1600000,
    "d": 1024,
}

P = [4, 16, 64, 256]  # number of GPUs (must be divisible by 4)
K = [16, 32, 64, 128]
C = ["IO", "K", "K Redist", "VI", "E", "E Transpose", "E MPI", "E Reduce", "E Gather", "E SpMM", "E other", "Z", "C", "C MPI", "C Computation", "VR Computation", "Elapsed"]


def create_random(low=0, high=100):
    filepath = RANDOM_DATASET["bin_fname"]
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    m = int(RANDOM_DATASET["m"])
    d = int(RANDOM_DATASET["d"])
    data = np.random.uniform(low, high, m * d).astype(np.float32)
    data.tofile(filepath)


# Functions related to downloading, extracting, and preparing the datasets
def progress_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r[%s] %d%%" % ("#" * (percent // 2), percent))
    sys.stdout.flush()


def print_file_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as response:
        file_size = response.getheader("Content-Length")
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
    d = int(dataset["d"])  # number of features
    if os.path.exists(bin_out):
        print(f"Binary file {bin_out} already exists. Skipping preparation.")
        return

    print(f"Preparing dataset {name} from {txt_in} to {bin_out}...")
    nbytes = os.path.getsize(txt_in)
    print(f"Byte count of {txt_in} is {nbytes} bytes...")
    lines_read = 0
    with open(bin_out, "wb+") as out:
        with open(txt_in, "r") as file:
            while True:
                line = file.readline()
                lines_read += 1
                if not line:
                    break
                progress_hook(file.tell(), 1, nbytes)
                features = line.rstrip().split(" ")
                features.pop(0)  # label is unused
                features_vector = np.zeros(d, dtype=np.float32)
                for feature in features:
                    index, value = feature.split(":")
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
    with open(path, "r") as file:
        rows = [line.rstrip() for line in file]
        for features in rows:
            features = features.split(" ")

            # extract label, which is unused
            label = features.pop(0)
            labels.append(label)

            features = np.float32(
                list(map(lambda feature: float(feature.split(":")[1]), features))
            )
            data.append(features)

    return np.array(data)


def create_scripts(account, path=os.getcwd(), n_trials=5, base_m=2**17, max_m=2**20):
    P = [2**i for i in range(2,9,2)]

    for p in P:
        # Strong scaling
        m = int(min(base_m, max_m))
        m -= m % p
        create_script(path, "strong", account, p, m, K, DATASETS, ALGS)

        # Weak scaling
        m = int(min(base_m*math.sqrt(p)/2, max_m))
        m -= m % p
        create_script(path, "weak", account, p, m, K, DATASETS, ALGS)

    # Strong scaling runner
    strong_path = os.path.join(path, "scripts", "strong.sh")
    with open(strong_path, "w") as f:
        f.write("#!/bin/bash\n")
        for p in P:
            f.write(f"sbatch --array=1-{n_trials} scripts/strong_{p}.sh\n")
    st = os.stat(strong_path)
    os.chmod(strong_path, st.st_mode | stat.S_IEXEC)

    # Weak scaling runner
    weak_path = os.path.join(path, "scripts", "weak.sh")
    with open(weak_path, "w") as f:
        f.write("#!/bin/bash\n")
        for p in P:
            f.write(f"sbatch --array=1-{n_trials} scripts/weak_{p}.sh\n")
    st = os.stat(weak_path)
    os.chmod(weak_path, st.st_mode | stat.S_IEXEC)

    # All runner
    all_path = os.path.join(path, "scripts", "all.sh")
    with open(all_path, "w") as f:
        f.write("#!/bin/bash\n")
        for test in ["strong", "weak"]:
            f.write(f". {path}/scripts/{test}.sh\n")
    st = os.stat(all_path)
    os.chmod(all_path, st.st_mode | stat.S_IEXEC)


def create_script(path, prefix, account, p, m, k, dataset, alg, d=None, sparse=None, niter=100, gamma=1, c=1, r=2, basic=False, convergence=False):
    """
    This function accepts either a single value or list for
    m, d, k, dataset, and alg. If a list is supplied, the
    script will perform one srun for each value.
    """

    basic = int(basic)
    convergence = int(convergence)
    nodes = math.ceil(p /4)

    log_dir = os.path.join(path, "logs")
    results_dir = os.path.join(path, "results")
    scripts_dir = os.path.join(path, "scripts")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    fname = f"{prefix}_{p}"
    script_fname = os.path.join(scripts_dir, fname+".sh")
    log_fname = os.path.join(log_dir, fname)

    # If individual values supplied, these must be
    # converted to a list for script generation.
    if not isinstance(m, list):
        m = [m]
    if not isinstance(d, list):
        d = [d]
    if not isinstance(k, list):
        k = [k]
    if not isinstance(dataset, list):
        dataset = [dataset]
    if not isinstance(alg, list):
        alg = [alg]

    # Max of 45 seconds for one trial estimated through experiments
    seconds_per_trial = 45
    total_time_seconds = seconds_per_trial*len(m)*len(d)*len(k)*len(dataset)*len(alg)
    timestamp = str(datetime.timedelta(seconds=total_time_seconds))

    with open(script_fname, "w") as f:
        f.writelines([
            "#!/bin/bash\n",
            f"#SBATCH --nodes={nodes}\n",
            f"#SBATCH --gpus={p}\n",
            f"#SBATCH --time={timestamp}\n",
            "#SBATCH --constraint=gpu&hbm40g\n",
            "#SBATCH --qos=regular\n",
            f"#SBATCH --account={account}\n",
            f"#SBATCH --output={log_fname}_%a\n",
            f"#SBATCH --error={log_fname}_%a_err\n",
            "export DVS_MAXNODES=1__\n",
            "module load cudatoolkit/12.9\n",
        ])

        for _m in m:
            for _d in d:
                for _k in k:
                    for _dataset in dataset:
                        for _alg in alg:
                            dataset_fname = _dataset["bin_fname"]
                            dataset_label = _dataset["label"].lower()
                            if d[0] is None:
                                _d = _dataset["d"]
                            _m = min(_m, _dataset["m"])
                            if sparse is None:
                                _sparse = int(_k > 32)
                            else:
                                _sparse = int(sparse)

                            name = f"{prefix}_{p}_{_m}_{_d}_{_k}_{niter}_{_sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{dataset_label}_{_alg}"
                            log_fname = os.path.join(log_dir, f"{name}_out")
                            result_fname = os.path.join(results_dir, f"{name}_assignments")
                            bench_fname = os.path.join(results_dir, f"{name}_time_${{SLURM_ARRAY_TASK_ID}}")

                            f.write(f"srun -N {nodes} --ntasks-per-node {p//nodes} --cpus-per-task 32 --cpu-bind cores -G {p} $PWD/../build/device_wrapper $PWD/../build/main ") # No newline so params on same line
                            f.write(f"-i {dataset_fname} -m {_m} -n {_d} --niter {niter} --sparse {_sparse} --gamma {gamma} --c {c} --r {r} --convergence {convergence} -k {_k} -o {result_fname} --benchmark {bench_fname} --alg {_alg} \n")


def get_scaling_data(scaling_type, p, m, d, k, niter, sparse, gamma, c, r, convergence, basic, input_dataset_name, alg, n_trials):
    # ``scaling_type`` is one of "strong" or "weak"
    scaling_data = {}
    for trial in range(n_trials):
        trial_in_fname = trial + 1
        fpath = f"results/{scaling_type}_{p}_{m}_{d}_{k}_{niter}_{sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{input_dataset_name}_{alg}_time_{trial_in_fname}_rank0"
        if not os.path.exists(fpath):
            print(f"Could not find {fpath}")
            continue
        with open(fpath, "r") as f:
            for label in C:
                line = f.readline()
                _label, _v = line.split(": ")
                if _label != label:
                    print(f"Got label '{_label}' but expected label '{label}' in file '{fpath}'. Skipping this file...")
                v = int(_v.split()[0])
                if not label in scaling_data:
                    scaling_data[label] = np.zeros(n_trials)
                scaling_data[label][trial] = v
    return scaling_data


def construct_graphs(path=os.getcwd(), n_trials=5, base_m=2**17):
    os.makedirs("graphs", exist_ok=True)

    niter = 100
    gamma = 1
    c = 1
    r = 2
    convergence = int(False)
    basic = int(False)

    for input_dataset in DATASETS:
        input_dataset_name = input_dataset["name"]
        d = input_dataset["d"]

        # Strong scaling data
        m = base_m
        k = 32
        for alg in ALGS:
            y = []
            for p in P:
                sparse = int(k > 32)
                scaling_data = get_scaling_data("strong", p, m, d, k, niter, sparse, gamma, c, r, convergence, basic, input_dataset_name, alg, n_trials)
                elapsed = scaling_data["Elapsed"].mean()
                y.append(elapsed)
            plt.plot(P, y, label=alg)
        plt.xscale("log", base=4)
        plt.yscale("log")
        plt.title("Strong Scaling for MNIST8M (k=32)")
        plt.xlabel("Number of Ranks (p)")
        plt.ylabel("Runtime (ms)")
        plt.xticks(P)
        plt.legend()
        plt.savefig("graphs/strong.png")


def compare(file1, file2):
    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"One of the files does not exist: {file1} or {file2}")
        return
    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        data1 = np.fromfile(f1, dtype=np.int32)
        data2 = np.fromfile(f2, dtype=np.int32)
        if data1.shape != data2.shape:
            print(f"Files have different sizes: {data1.shape[0]} vs {data2.shape[0]}")
            return
        n = data1.shape[0]
        diff = np.abs(data1 - data2)
        diff_points = np.count_nonzero(diff)
        print(f"Comparison of {file1} and {file2}:")
        print(f"Total points: {n}")
        print(f"Different points: {diff_points} ({(diff_points / n) * 100:.2f}%)")


# remember that D is the number of features and has to be correct for each dataset otherwise
# the MPI file read will be messed up
# but we can vary N and K for each experiment
# "convergence", "sparse", and "basic" are just flags
#   "convergence" is a flag for process-exclusion-based-convergence checking
#   "sparse" is the flag to use sparse V matrix
#   "basic" is the flag to use basic trial, otherwise it is a benchmark trial (but
#     the benchmark trial requires building with BASIC=0 and is non-negligibly slower due to finer-grained timing)

if __name__ == "__main__":
    legal = [
        "create_scripts",
        "create_random",
        "download",
        "extract",
        "prepare",
        "graphs",
        "compare",
    ]
    usage_legal = " | ".join(legal)
    if len(sys.argv) < 2:
        print(f"Usage: python exp.py [{usage_legal}]")
        sys.exit(1)
    action = sys.argv[1]
    if action not in legal:
        print(f"Invalid action. Must be one of {usage_legal}")
        sys.exit(1)
    if action == "create_scripts":
        create_scripts("m4341")
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
        print(
            "Generated LIBSVM-formatted textfile datasets in experiments/data/ directory."
        )
    if action == "prepare":
        for dataset in DATASETS:
            prepare(dataset)
        print(
            "Generated binary datasets in experiments/data/ directory... ready for use!"
        )
    if action == "graphs":
        construct_graphs()
        print("Generated graphs in experiments/graphs/ directory.")
    if action == "compare":
        if len(sys.argv) != 4:
            print("Usage: python exp.py compare <file1> <file2>")
            sys.exit(1)
        file1 = sys.argv[2]
        file2 = sys.argv[3]
        compare(file1, file2)
        print("Comparison complete")
