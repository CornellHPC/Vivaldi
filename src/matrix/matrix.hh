#ifndef DISTRIBUTED_POPCORN_MATRIX_H
#define DISTRIBUTED_POPCORN_MATRIX_H

#include <math.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../common.hh"

// Define CombBLAS sparse matrix format
template <typename UV> using UDER = combblas::SpCCols<int64_t, UV>;

namespace matrix {

void grid_size(int mpi_size, int *p_out, int *q_out);

slate::Options get_slate_opts();

void fill_slate_mat_with_buffer(slate::Matrix<DATA_TYPE> M, DATA_TYPE *buf);

void raise_slate_mat_to_power(slate::Matrix<DATA_TYPE> M, DATA_TYPE power);

void fill_slate_mat_with_scalar(slate::Matrix<DATA_TYPE> M, DATA_TYPE value);

DATA_TYPE get_slate_mat_value(slate::Matrix<DATA_TYPE> M, int64_t ii,
                              int64_t jj);

combblas::DnParMat<int64_t, DATA_TYPE>
slate_mat_to_combblas_dpm(slate::Matrix<DATA_TYPE> M);

combblas::SpParMat<int64_t, DATA_TYPE, UDER<DATA_TYPE>>
initialize_combblas_v_matrix(int m, int k);

} // namespace matrix

#endif // DISTRIBUTED_POPCORN_MATRIX_H
