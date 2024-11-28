#include <cstdlib>
#include <ctime>
#include <fstream>

int main() {
  srand(time(0));

  float min = 1.0f;
  float max = 100.0f;

  std::ofstream out;
  out.open("rand", std::ios::out | std::ios::binary);
  for (int i = 0; i < 96000 * 64; ++i) {
    float p = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float val = (max - min) * p + min;
    out.write(reinterpret_cast<const char*>(&val), sizeof(float));
  }
  out.close();
  return 0;
}
