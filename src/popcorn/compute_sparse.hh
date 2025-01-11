#include "cusparse_v2.h"
#include "slate/slate.hh"

#include "utils.hh"

using slate_matrix = std::unique_ptr<slate::Matrix<float>>;

namespace popcorn {

void extract_kernel_tiles(float* tiles, slate_matrix& K, int col);

cusparseSpMatDescr_t initialize_v(cusparseHandle_t& cusparse_handle, int m,
                                  int k);

cusparseDnMatDescr_t spmm(cusparseHandle_t& cusparse_handle,
                          cusparseSpMatDescr_t sparse,
                          cusparseDnMatDescr_t dense);

}  // namespace popcorn