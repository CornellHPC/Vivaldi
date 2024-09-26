#ifndef CUDA_H
#define CUDA_H

#ifdef CUDA
constexpr bool CUDA_AVAILABLE = true;
#else
constexpr bool CUDA_AVAILABLE = false;
#endif

#endif // CUDA_H
