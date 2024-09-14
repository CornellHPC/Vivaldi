#include "mpi.h"
#include "CombBLAS/CombBLAS.h"
#include <iostream>

int main(int argc, char *argv[]) {
  int rank, nprocs;

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  MPI_File fh;
  MPI_Status status;
  MPI_File_open(MPI_COMM_WORLD, "../DryBeanDataset/Dry_Bean_Dataset.arff",
                MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  constexpr int num_chars = 10;
  char *buf = (char *)malloc(num_chars * sizeof(char));
  MPI_File_read(fh, buf, num_chars, MPI_CHAR, &status);
  MPI_File_close(&fh);

  std::cout << "Rank " << rank << " received ";
  for (int i = 0; i < num_chars; ++i)
    std::cout << buf[i];
  std::cout << std::endl;

  MPI_Finalize();

  return 0;
}
