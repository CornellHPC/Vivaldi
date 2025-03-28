import os

def create_file_text(input_dataset_path_from_experiments_dir,
                     input_dataset_name,
                     p, m, d, k, niter, sparse, gamma, c, r, convergence, 
                     basic,
                     type_of_trial_letter="s", type_of_trial="Strong Basic Trial",
                     experiments_dir="$PWD"):
  if p % 4 != 0:
    raise ValueError("Number of nodes must be divisible by 4")
  nodes = p // 4
  s_name = f"{type_of_trial_letter}_{p}_{m}_{d}_{k}_{niter}_{sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{input_dataset_name}"
  log_dir = "logs"
  os.makedirs(log_dir, exist_ok=True)
  scripts_dir = "scripts"
  os.makedirs(scripts_dir, exist_ok=True)
  
  bash_file = os.path.join(scripts_dir, f"{s_name}.sh")
  with open(bash_file, "w") as f:
    f.write(f"""#!/bin/bash
#SBATCH --nodes={nodes}
#SBATCH --gpus={p}
#SBATCH --time=00:20:00
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --account=m4341
#SBATCH --output={log_dir}/{s_name}_out

export DVS_MAXNODES=1__
export EXE_PATH="{experiments_dir}/../build/device_wrapper {experiments_dir}/../build/main"
export DATA="-i {input_dataset_path_from_experiments_dir} -m {m} -n {d} --niter {niter} --sparse {sparse} --gamma {gamma} --c {c} --r {r} --convergence {convergence}"
export CLUSTERS="-k {k}"
export TRIALS=5

module load cudatoolkit/12.2

echo "Running {type_of_trial} test on {p} ranks!"
echo ""

for i in $(seq 1 $TRIALS); do
  echo "{type_of_trial} $i"
  srun -N {nodes} --ntasks-per-node 4 --cpus-per-task 32 --cpu-bind cores -G {p} $EXE_PATH $DATA $CLUSTERS -o {log_dir}/{s_name}_assignments --benchmark {log_dir}/{s_name}_time_$i
  echo ""
done

echo "Done!" """)

DATASETS = [
  {
    "bin_fname": "data/poker.t.bin",
    "txt_fname": "data/poker.t.txt",
    "m": "1000000",
    "d": "10",
    "k": "10"
  },
  {
    "bin_fname": "data/HIGGS.bin",
    "txt_fname": "data/HIGGS.txt",
    "m": "11000000",
    "d": "28",
    "k": "2"
  },
  {
    "bin_fname": "data/mnist.scale.bin",
    "txt_fname": "data/mnist.scale.txt",
    "m": "8100000",
    "d": "784",
    "k": "10"
  }
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
create_file_text(
  input_dataset_path_from_experiments_dir="data/poker.t.bin", 
  input_dataset_name="poker", 
  p=4, 
  m=1000000, 
  d=10, 
  k=10, 
  niter=100, 
  sparse=0, 
  gamma=2, 
  c=2, 
  r=2, 
  convergence=0, 
  basic=1, 
  type_of_trial_letter="s", 
  type_of_trial="Strong Basic Trial", 
  experiments_dir="$PWD")