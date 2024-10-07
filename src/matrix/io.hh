#ifndef DISTRIBUTED_POPCORN_IO_H
#define DISTRIBUTED_POPCORN_IO_H

#include <mpi.h>

#include "../common.hh"
#include "matrix.hh"

namespace matrix {

// Loads matrix at filename with m rows and n columns
slate::Matrix<DATA_TYPE> load_slate_mat(const char *filename, int64_t m,
                                        int64_t n, int64_t mb, int64_t nb,
                                        MPI_Comm comm);

} // namespace matrix

#endif // DISTRIBUTED_POPCORN_IO_H
