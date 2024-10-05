#ifndef DISTRIBUTED_POPCORN_IO_H
#define DISTRIBUTED_POPCORN_IO_H

#include <mpi.h>

#include "../common.hh"
#include "matrix.hh"

namespace matrix {

// Loads matrix at filename with m rows and n columns
slate::Matrix<DATA_TYPE> load_slate_mat(const char *filename, int mpi_size,
                                        int64_t m, int64_t n);

}  // namespace matrix

#endif  // DISTRIBUTED_POPCORN_IO_H
