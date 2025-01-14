#include "cusparse_v2.h"
#include "slate/slate.hh"

#include "utils.hh"

namespace popcorn {

cusparseSpMatDescr_t initialize_v(cusparseHandle_t& handle, int m, int k);

cusparseDnMatDescr_t spmm(cusparseHandle_t& handle, cusparseSpMatDescr_t& V,
                          cusparseDnMatDescr_t& K);

}  // namespace popcorn
