#include "slate/slate.hh"

#include "gpu_kernels.cuh"

using slate_matrix = std::unique_ptr<slate::Matrix<float>>;

namespace popcorn {

void extract_local_tiles(slate_matrix& K);

}