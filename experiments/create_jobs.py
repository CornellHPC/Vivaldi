import os
import numpy as np

from datetime import timedelta


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
        gamma = 1  # gamma fixed todo (all): good value for gamma?
        c = 1  # c fixed todo (all): good value for c?
        r = 2  # r fixed todo (all): good value for r?
        basic = True  # todo (all): this won't do breakdown
        for input_dataset in DATASETS:
            input_dataset_path = input_dataset["bin_fname"]
            input_dataset_name = input_dataset["name"]

            # strong scaling
            d = input_dataset["d"]
            convergence = 0
            m = 140000  # todo (matthew): based on how many points fit on 4 GPUs, fixed for strong scaling
            for k in [2, 5, 10, 20]:
                sparse = (k >= 10)  # todo (matthew): based on the results of your experiment
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
            for k in [2, 5, 10, 20]:
                m = 70000*np.sqrt(p) # todo (matthew): confirm this is correct formula for num. points in weak scaling, e.g. \sqrt{p*70k^2}
                sparse = (k >= 10)  # todo (matthew): based on the results of your experiment
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
            # convergence
            d = input_dataset["d"]
            convergence = 1
            for k in [2, 5, 10, 20]:
                m = 70000*np.sqrt(p)  # todo (matthew): confirm this is correct formula for num. points in weak scaling, e.g. \sqrt{p*70k^2}
                sparse = (k >= 10)  # todo (matthew): based on the results of your experiment
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
        f.write("echo 'Done!'\n")


DATASETS = [
    {
        "bin_fname": "data/poker.t.bin",
        "txt_fname": "data/poker.t.txt",
        "name": "poker",
        "m": "1000000",
        "d": "10",
        "k": "10",
    },
    {
        "bin_fname": "data/HIGGS.bin",
        "txt_fname": "data/HIGGS.txt",
        "name": "higgs",
        "m": "11000000",
        "d": "28",
        "k": "2",
    },
    {
        "bin_fname": "data/mnist.scale.bin",
        "txt_fname": "data/mnist.scale.txt",
        "name": "mnist",
        "m": "8100000",
        "d": "784",
        "k": "10",
    },
]


# remember that D is the number of features and has to be correct for each dataset otherwise
# the MPI file read will be messed up
# but we can vary N and K for each experiment
# "convergence", "sparse", and "basic" are just flags
#   "convergence" is a flag for process-exclusion-based-convergence checking
#   "sparse" is the flag to use sparse V matrix
#   "basic" is the flag to use basic trial, otherwise it is a benchmark trial (but
#     the benchmark trial requires building with BASIC=0 and is non-negligibly slower due to finer-grained timing)

# example:
for p in [4, 8, 16, 32, 64, 128, 256]:
    create_file_text(p, "")
