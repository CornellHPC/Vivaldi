#ifndef DISTRIBUTED_POPCORN_KERNEL_MATRIX_H
#define DISTRIBUTED_POPCORN_KERNEL_MATRIX_H

#include "../common.hh"
#include "matrix.hh"

namespace matrix {

slate::Matrix<DATA_TYPE> slate_point_mat_to_polynomial_kernel_mat(
    slate::Matrix<DATA_TYPE> M, DATA_TYPE gamma, DATA_TYPE c, DATA_TYPE r);

}  // namespace matrix

#endif  // DISTRIBUTED_POPCORN_KERNEL_MATRIX_H
