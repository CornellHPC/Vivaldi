#ifndef DISTRIBUTED_POPCORN_COMMON_H
#define DISTRIBUTED_POPCORN_COMMON_H

#include "CombBLAS/CombBLAS.h"

// Macro "Error" is defined in CombBLAS and SLATE. It is unused in
// CombBLAS, so we undefine it here to prevent name collisions.
#undef Error

#include "slate/slate.hh"

#include "const.hh"

using UDER = combblas::SpCCols<int64_t, DATA_TYPE>;
using SR = combblas::PlusTimesSRing<DATA_TYPE, DATA_TYPE>;

#endif // DISTRIBUTED_POPCORN_COMMON_H