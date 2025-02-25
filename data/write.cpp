#include <fstream>

int main() {
  std::ofstream out;
  out.open("test", std::ios::out | std::ios::binary);
  for (float i = 0; i < 16384; ++i) {
    out.write(reinterpret_cast<const char*>(&i), sizeof(float));
  }
  out.close();
  return 0;
}
