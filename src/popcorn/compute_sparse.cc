#include "compute_sparse.hh"

void popcorn::extract_local_tiles(slate_matrix& K) {
  // tiles rank, _ and rank + size, _ are local
  for (int64_t j = 0; j < K->nt(); ++j) {
    for (int64_t i = 0; i < K->mt(); ++i) {
      if (K->tileIsLocal(i, j)) {
        slate::Tile<float> tile = K->at(i, j, K->tileDevice(i, j));
        float* data = tile.data();
      }
    }
  }
}