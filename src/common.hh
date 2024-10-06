#ifndef DISTRIBUTED_POPCORN_COMMON_H
#define DISTRIBUTED_POPCORN_COMMON_H

#include "CombBLAS/CombBLAS.h"

// Macro "Error" is defined in CombBLAS and SLATE. It is unused in
// CombBLAS, so we undefine it here to prevent name collisions.
#undef Error

#include "slate/slate.hh"

#endif // DISTRIBUTED_POPCORN_COMMON_H
