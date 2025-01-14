#include "cusparse_v2.h"
#include "mpi.h"

namespace popcorn {

cusparseDnVecDescr_t compute_c(cusparseHandle_t& handle, cusparseSpMatDescr_t V,
                               cusparseDnMatDescr_t ET, MPI_Comm comm);

}
