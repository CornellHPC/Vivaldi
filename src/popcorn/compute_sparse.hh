#include "cusparse_v2.h"
#include "slate/slate.hh"
#include "utils.hh"

namespace popcorn {

void extract_kernel_tiles(float** tiles, slate::Matrix<float>& K, int col);

cusparseSpMatDescr_t initialize_v(cusparseHandle_t& handle, int m,
                                  int k);

cusparseDnMatDescr_t spmm(cusparseHandle_t& handle,
                          cusparseSpMatDescr_t V,
                          cusparseDnMatDescr_t K);

}  // namespace popcorn