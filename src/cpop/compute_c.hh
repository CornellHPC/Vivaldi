#ifndef CPOP_COMPUTE_C_HH
#define CPOP_COMPUTE_C_HH

#include "cusparse.h"
#include "mpi.h"

namespace cpop {

cusparseDnVecDescr_t compute_c(cusparseHandle_t& handle, cusparseSpMatDescr_t V,
                               cusparseDnMatDescr_t ET, MPI_Comm comm);

}  // namespace cpop

#endif  // CPOP_COMPUTE_C_HH
