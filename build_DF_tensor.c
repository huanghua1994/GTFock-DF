#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <omp.h>

#include <mkl.h>

#include "utils.h"
#include "TinySCF.h"
#include "build_DF_tensor.h"

static void copy_3center_integral_results(
    int thread_npairs, int *thread_P_list, int thread_nints, double *thread_integrals, 
    int *df_shell_bf_sind, double *pqA, int nbf, int df_nbf,
    int startM, int endM, int startN, int endN, int dimN, int pqA_M_offset
)
{
    for (int ipair = 0; ipair < thread_npairs; ipair++)
    {
        int P = thread_P_list[ipair];
        int startP = df_shell_bf_sind[P];
        int dimP   = df_shell_bf_sind[P + 1] - startP;
        size_t row_mem_size = sizeof(double) * dimP;
        double *integrals = thread_integrals + thread_nints * ipair;
        
        for (int iM = startM; iM < endM; iM++)
        {
            int im = iM - startM;
            for (int iN = startN; iN < endN; iN++)
            {
                int in = iN - startN;
                int iM1 = iM - pqA_M_offset;
                double *eri_ptr = integrals + (im * dimN + in) * dimP;
                size_t pqA_offset0 = (size_t) (iM1 * nbf + iN) * (size_t) df_nbf + (size_t) startP;
                double *pqA_ptr0 = pqA + pqA_offset0;
                memcpy(pqA_ptr0, eri_ptr, row_mem_size);
            }
        }
    }
}

static void calc_DF_3center_integrals(TinySCF_t TinySCF, int *P_lists, int iMN_sind, int iMN_eind)
{
    double *pqA           = TinySCF->pqA;
    int nbf               = TinySCF->nbasfuncs;
    int df_nbf            = TinySCF->df_nbf;
    int nshell            = TinySCF->nshells;
    int *shell_bf_sind    = TinySCF->shell_bf_sind;
    int *df_shell_bf_sind = TinySCF->df_shell_bf_sind;
    Simint_t simint       = TinySCF->simint;
    int *uniq_sp_lid      = TinySCF->uniq_sp_lid;
    int *uniq_sp_rid      = TinySCF->uniq_sp_rid;
    int num_uniq_sp       = TinySCF->num_uniq_sp;
    double *sp_scrval     = TinySCF->sp_scrval;
    double *df_sp_scrval  = TinySCF->df_sp_scrval;
    double scrtol2        = TinySCF->shell_scrtol2;
    int pqA_M_offset      = shell_bf_sind[uniq_sp_lid[iMN_sind]];

    #pragma omp parallel 
    {
        int tid = omp_get_thread_num();
        int *thread_P_list = P_lists + tid * _Simint_NSHELL_SIMD;
        double *thread_integrals;
        int thread_nints, thread_npairs;
        void *thread_multi_shellpair;
        CMS_Simint_createThreadMultishellpair(&thread_multi_shellpair);
        
        #pragma omp for schedule(dynamic)
        for (int iMN = iMN_sind; iMN < iMN_eind; iMN++)
        {
            int M = uniq_sp_lid[iMN];
            int N = uniq_sp_rid[iMN];
            int startM = shell_bf_sind[M];
            int endM   = shell_bf_sind[M + 1];
            int startN = shell_bf_sind[N];
            int endN   = shell_bf_sind[N + 1];
            int dimM   = endM - startM;
            int dimN   = endN - startN;
            double scrval0 = sp_scrval[M * nshell + N];
            
            for (int iAM = 0; iAM <= simint->df_max_am; iAM++)
            {
                thread_npairs = 0;
                int iP_start = simint->df_am_shell_spos[iAM];
                int iP_end   = simint->df_am_shell_spos[iAM + 1];
                for (int iP = iP_start; iP < iP_end; iP++)
                {
                    int P = simint->df_am_shell_id[iP];
                    double scrval1 = df_sp_scrval[P];
                    if (scrval0 * scrval1 < scrtol2) continue;
                    
                    thread_P_list[thread_npairs] = P;
                    thread_npairs++;
                    
                    if (thread_npairs == _Simint_NSHELL_SIMD)
                    {
                        CMS_Simint_computeDFShellQuartetBatch(
                            simint, tid, M, N, thread_P_list, thread_npairs, 
                            &thread_integrals, &thread_nints,
                            &thread_multi_shellpair
                        );
                        
                        if (thread_nints > 0)
                        {
                            copy_3center_integral_results(
                                thread_npairs, thread_P_list, thread_nints, thread_integrals,
                                df_shell_bf_sind, pqA, nbf, df_nbf,
                                startM, endM, startN, endN, dimN, pqA_M_offset
                            );
                        }
                        
                        thread_npairs = 0;
                    }
                }  // for (int iP = iP_start; iP < iP_end; iP++)
                
                if (thread_npairs > 0)
                {
                    CMS_Simint_computeDFShellQuartetBatch(
                        simint, tid, M, N, thread_P_list, thread_npairs, 
                        &thread_integrals, &thread_nints,
                        &thread_multi_shellpair
                    );
                    
                    if (thread_nints > 0)
                    {
                        copy_3center_integral_results(
                            thread_npairs, thread_P_list, thread_nints, thread_integrals,
                            df_shell_bf_sind, pqA, nbf, df_nbf,
                            startM, endM, startN, endN, dimN, pqA_M_offset
                        );
                    }
                } 
            }  // for (int iAM = 0; iAM <= simint->df_max_am; iAM++)
        }  // for (int iMN = iMN_sind; iMN < iMN_eind; iMN++)
        
        CMS_Simint_freeThreadMultishellpair(&thread_multi_shellpair);
    }  // #pragma omp parallel 
}

static void calc_DF_2center_integrals(TinySCF_t TinySCF)
{
    // Fast enough, need not to batch shell quartets
    double *Jpq           = TinySCF->Jpq;
    int df_nbf            = TinySCF->df_nbf;
    int df_nshell         = TinySCF->df_nshells;
    int *df_shell_bf_sind = TinySCF->df_shell_bf_sind;
    Simint_t simint       = TinySCF->simint;
    double *df_sp_scrval  = TinySCF->df_sp_scrval;
    double scrtol2        = TinySCF->shell_scrtol2;
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int thread_nints;
        double *thread_integrals;
        
        #pragma omp for schedule(dynamic)
        for (int M = 0; M < df_nshell; M++)
        {
            double scrval0 = df_sp_scrval[M];
            for (int N = M; N < df_nshell; N++)
            {
                double scrval1 = df_sp_scrval[N];
                if (scrval0 * scrval1 < scrtol2) continue;

                CMS_Simint_computeDFShellPair(simint, tid, M, N, &thread_integrals, &thread_nints);
                
                if (thread_nints <= 0) continue;
                
                int startM = df_shell_bf_sind[M];
                int endM   = df_shell_bf_sind[M + 1];
                int startN = df_shell_bf_sind[N];
                int endN   = df_shell_bf_sind[N + 1];
                int dimM   = endM - startM;
                int dimN   = endN - startN;
                
                for (int iM = startM; iM < endM; iM++)
                {
                    int im = iM - startM;
                    for (int iN = startN; iN < endN; iN++)
                    {
                        int in = iN - startN;
                        double I = thread_integrals[im * dimN + in];
                        Jpq[iM * df_nbf + iN] = I;
                        Jpq[iN * df_nbf + iM] = I;
                    }
                }
            }  // for (int N = i; N < df_nshell; N++)
        }  // for (int M = 0; M < df_nshell; M++)
    }  // #pragma omp parallel
}

static void calc_inverse_sqrt_Jpq(TinySCF_t TinySCF)
{
    double *Jpq = TinySCF->Jpq;
    int df_nbf  = TinySCF->df_nbf;
    
    size_t df_mat_mem_size = DBL_SIZE * df_nbf * df_nbf;
    double *tmp_mat0  = ALIGN64B_MALLOC(df_mat_mem_size);
    double *tmp_mat1  = ALIGN64B_MALLOC(df_mat_mem_size);
    double *df_eigval = ALIGN64B_MALLOC(DBL_SIZE * df_nbf);
    assert(tmp_mat0 != NULL && tmp_mat1 != NULL);
    // Diagonalize Jpq = U * S * U^T, the eigenvectors are stored in tmp_mat0
    memcpy(tmp_mat0, Jpq, df_mat_mem_size);
    LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', df_nbf, tmp_mat0, df_nbf, df_eigval);
    // Apply inverse square root to eigen values to get the inverse squart root of Jpq
    for (int i = 0; i < df_nbf; i++)
        df_eigval[i] = 1.0 / sqrt(df_eigval[i]);
    // Right multiply the S^{-1/2} to U
    #pragma omp parallel for
    for (int irow = 0; irow < df_nbf; irow++)
    {
        double *tmp_mat0_ptr = tmp_mat0 + irow * df_nbf;
        double *tmp_mat1_ptr = tmp_mat1 + irow * df_nbf;
        memcpy(tmp_mat1_ptr, tmp_mat0_ptr, DBL_SIZE * df_nbf);
        for (int icol = 0; icol < df_nbf; icol++)
            tmp_mat0_ptr[icol] *= df_eigval[icol];
    }
    // Get Jpq^{-1/2} = U * S^{-1/2} * U', Jpq^{-1/2} is stored in Jpq
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, df_nbf, df_nbf, df_nbf, 
                1.0, tmp_mat0, df_nbf, tmp_mat1, df_nbf, 0.0, Jpq, df_nbf);
    ALIGN64B_FREE(tmp_mat0);
    ALIGN64B_FREE(tmp_mat1);
    ALIGN64B_FREE(df_eigval);
}

// Formula: df_tensor(i, j, k) = dot(pqA(i, j, 1:df_nbf), Jpq_invsqrt(1:df_nbf, k))
static void generate_df_tensor(TinySCF_t TinySCF, int M_bf_sind, int M_bf_eind)
{
    double *df_tensor = TinySCF->df_tensor;
    double *pqA = TinySCF->pqA;
    double *Jpq = TinySCF->Jpq;

    int Mrows  = M_bf_eind - M_bf_sind;
    int nbf    = TinySCF->nbasfuncs;
    int df_nbf = TinySCF->df_nbf;
    int my_df_nbf        = TinySCF->my_df_nbf;
    int my_df_nbf_offset = TinySCF->my_df_nbf_offset;
    
    size_t offset = (size_t) (M_bf_sind * nbf) * (size_t) my_df_nbf;
    double *df_tensor_M0 = df_tensor + offset;
    double *Jpq_my_ptr   = Jpq + my_df_nbf_offset;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nbf * Mrows, my_df_nbf, df_nbf,
                1.0, pqA, df_nbf, Jpq_my_ptr, df_nbf, 0.0, df_tensor_M0, my_df_nbf);
}

void TinySCF_build_DF_tensor(TinySCF_t TinySCF)
{
    double st, et;

    if (TinySCF->my_rank == 0) printf("---------- DF tensor construction ----------\n");
    
    // Calculate the Coulomb metric matrix
    st = get_wtime_sec();
    calc_DF_2center_integrals(TinySCF);
    et = get_wtime_sec();
    if (TinySCF->my_rank == 0) printf("* 2-center integral : %.3lf (s)\n", et - st);

    // Factorize the Jpq
    st = get_wtime_sec();
    calc_inverse_sqrt_Jpq(TinySCF);
    et = get_wtime_sec();
    if (TinySCF->my_rank == 0) printf("* matrix inv-sqrt   : %.3lf (s)\n", et - st);

    double eri3_t = 0.0, build_tensor_t = 0.0;

    int *left_shell_sind = (int*) malloc(INT_SIZE * (TinySCF->nshells + 1));
    assert(left_shell_sind != NULL);
    left_shell_sind[0] = 0;
    int prev_shell = 0;
    for (int i = 1; i < TinySCF->num_uniq_sp; i++)
    {
        int new_shell = TinySCF->uniq_sp_lid[i];
        if (prev_shell == new_shell) continue;
        while (prev_shell < new_shell)
        {
            prev_shell++;
            left_shell_sind[prev_shell] = i;
        }
    }
    left_shell_sind[TinySCF->nshells] = TinySCF->num_uniq_sp;

    int *P_lists = (int*) malloc(sizeof(int) * _Simint_NSHELL_SIMD * TinySCF->nthreads);
    assert(P_lists != NULL);
    
    size_t pqA_band_size  = (size_t) TinySCF->max_dim  * (size_t) TinySCF->nbasfuncs * (size_t) TinySCF->df_nbf;
    
    for (int M = 0; M < TinySCF->nshells; M++)
    {
        // Since we reuse the pqA buffer, we need to reset it as 0, otherwise
        // those screened positions may have old vaules
        #pragma omp parallel for
        for (size_t i = 0; i < pqA_band_size; i++) TinySCF->pqA[i] = 0;

        // Calculate 3-center density fitting integrals
        int iMN_sind = left_shell_sind[M];
        int iMN_eind = left_shell_sind[M + 1];
        st = get_wtime_sec();
        calc_DF_3center_integrals(TinySCF, P_lists, iMN_sind, iMN_eind);
        et = get_wtime_sec();
        eri3_t += et - st;

        // Form the density fitting tensor
        int M_bf_sind = TinySCF->shell_bf_sind[M];
        int M_bf_eind = TinySCF->shell_bf_sind[M + 1];
        st = get_wtime_sec();
        generate_df_tensor(TinySCF, M_bf_sind, M_bf_eind);
        et = get_wtime_sec();
        build_tensor_t += et - st;
    }

    if (TinySCF->my_rank == 0) printf("* 3-center integral : %.3lf (s)\n", eri3_t);
    if (TinySCF->my_rank == 0) printf("* build DF tensor   : %.3lf (s)\n", build_tensor_t);
    
    free(P_lists);
    free(left_shell_sind);

    if (TinySCF->my_rank == 0) printf("---------- DF tensor construction finished ----------\n");
}