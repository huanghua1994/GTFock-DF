#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <assert.h>
#include <mpi.h>

#include "CMS.h"
#include "TinySCF.h"
#include "build_DF_tensor.h"

static void print_usage(char *exe_name)
{
	printf("Usage: mpirun -np <nprocs> %s <basis> <denfit_basis> <xyz> <nproc_row> <nproc_col> <niter>\n", exe_name);
}

int main(int argc, char **argv)
{
	int my_rank, comm_size, nproc_row, nproc_col, niter;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
	
	if (argc < 7)
	{
		if (my_rank == 0) print_usage(argv[0]);
		MPI_Finalize();
		return 255;
	}
	
	nproc_row = atoi(argv[4]);
	nproc_col = atoi(argv[5]);
	niter     = atoi(argv[6]);
	
	if (nproc_col * nproc_row != comm_size)
	{
		if (my_rank == 0) printf("FATAL: nproc_row * nproc_col != nprocs!!\n");
		MPI_Finalize();
		return 255;
	}
	
	TinySCF_t TinySCF;
	TinySCF = (TinySCF_t) malloc(sizeof(struct TinySCF_struct));
	assert(TinySCF != NULL);
	
	init_TinySCF(
		TinySCF, argv[1], argv[2], argv[3], 
		comm_size, my_rank, nproc_row, nproc_col, niter
	);

	TinySCF_init_batch_dgemm_arrays(TinySCF);
	
	TinySCF_compute_Hcore_Ovlp_mat(TinySCF);
	
	TinySCF_compute_sq_Schwarz_scrvals(TinySCF);
	
	TinySCF_get_initial_guess(TinySCF);
	
	TinySCF_build_DF_tensor(TinySCF);

	// TinySCF_do_SCF(TinySCF);

	TinySCF_free_batch_dgemm_arrays(TinySCF);
	
	free_TinySCF(TinySCF);
	
	MPI_Finalize();
	
	return 0;
}
