import urllib.request, bz2, lzma, os, sys
import numpy as np

from datetime import timedelta


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
        "m": "1000000", # "8100000",
        "d": "784",
        "k": "10",
    },
]

MAX_NUM_POINTS = 1200000 ## one million points limit for basically everything


def request_p_prefix(p, nodes, log_dir, s_name):
    timestamp = str(timedelta(minutes=30+15*np.sqrt(p)))

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
    log_dir,
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
        f"-o {log_dir}/{s_name}_assignments --benchmark {log_dir}/{s_name}_time_$i"
    )
    n_trials = 5
    f.write(f'echo "Running with args {main_args}"\n')
    f.write(f'echo ""\n')
    f.write(f"for i in $(seq 1 {n_trials}); do\n")
    f.write(f'  echo "Trial $i"\n')
    f.write(
        f"  srun -N {nodes} --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G {p} $EXE_PATH {main_args} {log_args}\n"
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
    s_name = f"exp_{unique_id}_{p}_{suffix}.sh"
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
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
        niter = 100  # max niter is always fixed at 100
        gamma = 1  # gamma fixed at 1
        c = 1  # c fixed at 1
        r = 2  # r fixed at 2 (quadratic kernel)
        basic = True  # todo (all): this won't do breakdown
        for input_dataset in DATASETS:
            input_dataset_path = input_dataset["bin_fname"]
            input_dataset_name = input_dataset["name"]

            # strong scaling
            d = input_dataset["d"]
            convergence = 0
            m = 140000  # experimentally decided that 140k points fit on 4 GPUs
            for k in [2, 5, 10, 50, 100]:
                # 32 based on experiments
                sparse = int(k > 32)
                run_5_trials(
                    f,
                    input_dataset_path,
                    log_dir,
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
            convergence = 0
            for k in [2, 5, 10, 50, 100]:
                # weak scaling number of points
                m = min(int(70000*np.sqrt(p)), int(input_dataset["m"]), MAX_NUM_POINTS)
                # 32 based on experiments
                sparse = int(k > 32)
                run_5_trials(
                    f,
                    input_dataset_path,
                    log_dir,
                    f"{unique_id}_{p}_{m}_{d}_{k}_{niter}_{sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{input_dataset_name}",
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
            # convergence (todo)
            # combblas (todo)
        f.write("echo 'Done!'\n")


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
    if len(sys.argv) < 2:
        print('Usage: python exp.py ["create_scripts" | "download" | "extract" | "prepare"]')
        sys.exit(1)
    action = sys.argv[1]
    if action not in ["create_scripts", "download", "extract", "prepare"]:
        print('Invalid action. Must be one of ["create_scripts", "download", "extract", "prepare"]')
        sys.exit(1)
    if action == "create_scripts":
        for p in [4, 8, 16, 32, 64, 128, 256]:
            create_file_text(p, "")
        print("Generated scripts in experiments/scripts/ directory.")
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
    