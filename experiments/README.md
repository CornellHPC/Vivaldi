# ClusterPop Experimentation

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

### Convergence Detection Testing

todo

### Correctness Testing

todo

## Running Tests

### Basic Testing

For basic testing, from the project folder build with:

```bash
make build BASIC=1
```

Once this completes, from the `experiments` folder run with:

```bash
sbatch basic/s1.sh
sbatch basic/s2.sh
sbatch basic/s4.sh
sbatch basic/s8.sh
sbatch basic/s16.sh
sbatch basic/s32.sh
sbatch basic/s64.sh
sbatch basic/s128.sh
sbatch basic/s256.sh
sbatch basic/w1.sh
sbatch basic/w2.sh
sbatch basic/w4.sh
sbatch basic/w8.sh
sbatch basic/w16.sh
sbatch basic/w32.sh
sbatch basic/w64.sh
sbatch basic/w128.sh
sbatch basic/w256.sh
```

Note that the 32, 64, 128, and 256 GPU configurations demand a large number of resources and should be run with caution.

### Breakdown Testing

For breakdown testing, from the project folder build with:

```bash
make build
```

Once this completes, from the `experiments` folder run with:

```bash
sbatch breakdown/s1.sh
sbatch breakdown/s2.sh
sbatch breakdown/s4.sh
sbatch breakdown/s8.sh
sbatch breakdown/s16.sh
sbatch breakdown/s32.sh
sbatch breakdown/s64.sh
sbatch breakdown/s128.sh
sbatch breakdown/s256.sh
sbatch breakdown/w1.sh
sbatch breakdown/w2.sh
sbatch breakdown/w4.sh
sbatch breakdown/w8.sh
sbatch breakdown/w16.sh
sbatch breakdown/w32.sh
sbatch breakdown/w64.sh
sbatch breakdown/w128.sh
sbatch breakdown/w256.sh
```

Note that the 32, 64, 128, and 256 GPU configurations demand a large number of resources and should be run with caution.

### Testing Output

* The `sl_out` folder includes logging from tests that can usually be ignored.
* The `assignments` folder contains the point assignments. This is only relevant for correctness testing.
* The `basic_time` folder contains the timing for basic tests. For example, `basic_time/s1` contains a single line with the total time of the algorithm.
* The `breakdown_time` folder contains the breakdown timing. For example, `breakdown_time/s1` contains lines for each routine’s time.

## Graphing

Graphing for the strong basic, strong breakdown, weak basic, and weak breakdown should be run with

```bash
python generate_graphs.py
```


