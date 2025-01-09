#include "cusparse_v2.h"
#include "slate/slate.hh"

#include "utils.hh"

using slate_matrix = std::unique_ptr<slate::Matrix<float>>;

namespace popcorn {

void create_kernel_descriptors(cusparseDnMatDescr_t K1_desc,
                         cusparseDnMatDescr_t K2_desc, float* K1, float* K2,
                         slate_matrix& K, bool* multiple);

}