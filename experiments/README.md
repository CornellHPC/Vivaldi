# Kettlecorn Experimentation

## Installation

See the README in the parent folder.

## Overview of Testing

Testing is split along three axes:

* The number of processes. The current configurations are:
  * 1 GPU on 1 node (e.g. *s1*)
  * 2 GPUs on 1 node (e.g. *s2*)
  * 4 GPUs on 1 node (e.g. *s4*)
  * 8 GPUs split amongst 2 nodes (e.g. *s8*)
  * 16 GPUs split amongst 4 nodes (e.g. *s16*)
  * 32 GPUs split amonst 8 nodes (e.g. *s32*)
  * 64 GPUs split amonst 16 nodes (e.g. *s64*)
  * 128 GPUs split amongst 32 nodes (e.g. *s128*)
  * 256 GPUs split amongst 64 nodes (e.g. *s256*)
* **Strong** and **weak** testing
  * In **strong** testing, the input size is fixed (70k points to be clustered into 128 clusters) and only the number of processes is increased.
  * In **weak** testing, the input size is also increased logarithmically with the number of processes.
* **Basic** and **breakdown** testing
  * In **basic** testing, there is no fine-grained timing. Because the act of timing introduces non-trivial delays, running in basic mode ensures that the algorithm runs as fast as possible. The only time benchmarked is the total algorithm runtime. This mode should be used to generate the overall strong and weak scaling plots.
  * In **breakdown** testing, there is fine-grained timing. In addition to benchmarking the total algorithm runtime, we benchmark the runtime of each individual routine (e.g. IO, SpMM, SpMV, etc) and naïvely benchmark MPI time. This mode should be used to generate the breakdown plots.


Important constants in testing are:

* For scaling tests, convergence detection is never used.
* Run for **100** iterations.
* Run with **64** features.
* Cluster in **128** clusters.
* For strong scaling, number of points is **70k**. For weak scaling, the initial number of points (i.e. for the *w1* configuration) is also **70k**.
* Timing includes only the relevant initialization and k-means loop. IO is not included in overall elapsed time.

## Running Tests

From the project folder build with:

```bash
make build
```


Once this completes, from the `experiments` folder run with:

```bash
pip install -r requirements.txt
python exp.py download
python exp.py extract
python exp.py prepare
python exp.py create_random
python exp.py create_scripts
```

This will do a number of things

* `download` and `extract` download and extract the dataset to the `experiments/data` directory. The data is stored in `txt` files that follow the LibSVM format
* `prepare` converts the dataset from LibSVM format to the format of a binary list of floats, which is the input to the algorithm. After running `python exp.py prepare`, the `txt` files in `experiments/data` can be deleted (since these take up a lot of space on disk and are not necessary anymore)
* `create_random` creates a random number dataset which is used in proper weak scaling tests (where both number of points and number of features are scaled in accordance with number of processes)
* `create_scripts` generates all the scripts. See `exp.py` for more. There are a couple of global variables at the top of the file that can be adjusted to only generate a subset of scripts (e.g. only generate strong scaling or weak scaling). The generated scripts go to the `experiments/scripts` folder. Launch these scripts FROM THE EXPERIMENTS DIRECTORY with sbatch, as in `sbatch scripts/exp__4_0.sh`. Note that the 32, 64, 128, and 256 GPU configurations demand a large number of resources and should be run with caution.


Scripts can be monitored with `squeue --me` or (to roughly see the place in the queue) `squeue | grep "gpu_ss11" | grep -n "<username>"`.


Once all scripts are done, run

```bash
python exp.py graphs
```


This will generate graphs in the `experiments/graphs` folder.

### Testing Output

The `logs` folder includes logging from tests that can usually be ignored.

The `results` folder contains the timing and assignments for basic tests. Each file has a fairly long name that indicates the arguments with which it was run. Each file starts with a prefix that looks like `_{w}_{p}_{m}_{d}_{k}_{niter}_{sparse}_{gamma}_{c}_{r}_{convergence}_{basic}_{input_dataset_name}`. Explanation:
* `w` is the type of test `s` for strong scaling, `w` for variant weak scaling, `wp` for proper weak scaling, `m` for cluster size mode testing, `wc`/`wce` for variant weak scaling with convergence (`e` is for process-exclusion)
* `p` is the number of processes/ranks
* `m` for number of points
* `d` for number of features
* `k` for number of clusters
* `niter`, `gamma`, `c`, `r` obvious
* `sparse` is `0` if the V matrix should be dense, else `1`
* `convergence` is `0` for no convergence, `1` for simple convergence with no process exclusion, `2` for convergence with process exclusion
* `basic` should always just be `True`
* `input_dataset_name` is one of `higgs`, `mnist`, or `poker`

These names are decided automatically by the `create_scripts` routine, so we don't have to manually adjust any of these parameters. The result files are suffixed by `_assignments` (for the actual point assignment data), and `_time_i` (for the timing breakdown). `_time_i` is the more useful of the two, and is used to generate graphs/tables for the paper.


## CombBLAS Implementation Testing
We also provide `exp_combblas.py` for generating strong scaling test scripts
for testing the alternative CombBLAS implementation. Note that this script
stores and fetches data from `$PSCRATCH` directory instead of `data` directory,
but `$PSCRATCH` can be replaced with `data` if you have already downloaded and prepared the data
from `exp.py`.

Running `python exp.py create_scripts` will generate three strong scaling scripts
for `p=16,64,256`, which can be run with `sbatch scripts/exp__16_0.sh` for example.