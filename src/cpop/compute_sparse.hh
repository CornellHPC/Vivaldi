#ifndef CPOP_COMPUTE_SPARSE_HH
#define CPOP_COMPUTE_SPARSE_HH

#include "cusparse.h"

namespace cpop {

cusparseSpMatDescr_t initialize_v(cusparseHandle_t& handle, int m, int k);

cusparseDnMatDescr_t spmm(cusparseHandle_t& handle, cusparseSpMatDescr_t& V,
                          cusparseDnMatDescr_t& K);

}  // namespace cpop

#endif  // CPOP_COMPUTE_SPARSE_HH
