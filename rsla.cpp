//
// Created by chlorinepentoxide on 7/1/25.
//

#include "rsla_types.h"
#include "rsla.h"
#include "mplapack_mpfr.h"


#ifdef USE_ARMA
#include <armadillo>
#endif

#ifdef USE_MPLAPACK
#include "mpblas_mpfr.h"
#include "mplapack_mpfr.h"
#endif

namespace rsla {

#ifdef USE_ARMA

    arma::mat cast (RSMatrix& A) {
        return { *A.getmemptr(), A.rows(), A.cols() };
    }

    RSMatrix cast (arma::mat& A) {
        return RSMatrix(A.memptr(), A.n_rows, A.n_cols);
    }

    arma::vec cast (RSVector& v) {
        return {*v.getmemptr(), v.size()};
    }

    RSVector cast (arma::vec& v) {
        return {v.memptr(), v.size()};
    }

    RSMatrix inv(RSMatrix& A) {
        arma::mat i = arma::inv(cast(A));
        return cast(i);
    }

    RSMatrix pinv(RSMatrix& A) {
        arma::mat i = arma::pinv(cast(A));
        return cast(i);
    }

    real condition_number(RSMatrix& A) {
        real c = arma::cond(cast(A));
        return c;
    }

    RSMatrix multiply(RSMatrix& A, RSMatrix& B) {
        arma::mat cA = cast(A);
        arma::mat cB = cast(B);
        arma::mat cC = cA * cB;
        return cast(cC);
    }

    RSVector multiply(RSMatrix &A, RSVector &v) {
        arma::mat cA = cast(A);
        arma::vec cV = cast(v);
        arma::vec cC = cA * cV;
        return cast(cC);
    }

    RSEigenResult eigen(RSMatrix& A) {
        arma::mat cA = cast(A);
        arma::cx_vec eigenvalues;
        arma::cx_mat eigenvectors;
        arma::eig_gen(eigenvalues, eigenvectors, cA);
        RSVector eval_real (eigenvalues.size());
        RSVector eval_imag (eigenvalues.size());
        for(size_t i = 0; i < eigenvalues.size(); i++) {
            auto e = eigenvalues(i);
            eval_real.set(i, e.real());
            eval_imag.set(i, e.imag());
        }
        RSMatrix evec_real (eigenvectors.n_rows, eigenvectors.n_cols);
        RSMatrix evec_imag (eigenvectors.n_rows, eigenvectors.n_cols);
        for(size_t i = 0; i < eigenvectors.n_rows; i++) {
            for(size_t j = 0; j < eigenvectors.n_cols; j++) {
                auto e = eigenvectors(i, j);
                evec_real.set(i, j, e.real());
                evec_imag.set(i, j, e.imag());
            }
        }
        return RSEigenResult { eval_real, eval_imag, evec_real, evec_imag };
    }

    RSEigenResult eigen(RSMatrix& A, RSMatrix& B) {
        arma::mat cA = cast(A);
        arma::mat cB = cast(B);
        arma::cx_vec eigenvalues;
        arma::cx_mat eigenvectors;
        arma::eig_pair(eigenvalues, eigenvectors, cA, cB);
        RSVector eval_real (eigenvalues.size());
        RSVector eval_imag (eigenvalues.size());
        for(size_t i = 0; i < eigenvalues.size(); i++) {
            auto e = eigenvalues(i);
            eval_real.set(i, e.real());
            eval_imag.set(i, e.imag());
        }
        RSMatrix evec_real (eigenvectors.n_rows, eigenvectors.n_cols);
        RSMatrix evec_imag (eigenvectors.n_rows, eigenvectors.n_cols);
        for(size_t i = 0; i < eigenvectors.n_rows; i++) {
            for(size_t j = 0; j < eigenvectors.n_cols; j++) {
                auto e = eigenvectors(i, j);
                evec_real.set(i, j, e.real());
                evec_imag.set(i, j, e.imag());
            }
        }
        return RSEigenResult { eval_real, eval_imag, evec_real, evec_imag };
    }

    RSVector solve(RSMatrix &A, RSVector &b) {
        arma::mat cA = cast(A);
        arma::vec cb = cast(b);
        arma::vec cx = arma::solve(cA, cb);
        return cast(cx);
    }

#endif

#ifdef USE_MPLAPACK

    real infinity_norm (RSMatrix& matrix_A) {
        char        NORM = 'I';
        integer     M    = matrix_A.rows();
        integer     N    = matrix_A.cols();
        real*       A    = *matrix_A.getmemptr();
        integer     LDA  = M;
        real*       WORK = new real [M];

        // Call Rlange
        real norm = Rlange (&NORM, M, N, A, LDA, WORK);

        delete[] WORK;

        return norm;
    }


    real condition_number (RSMatrix& matrix_A) {
        char         NORM  = 'I'; // RSLA uses Infinity-norm by default for condition number estimates
        integer      N     = matrix_A.cols();
        real*        A     = *matrix_A.getmemptr();
        integer      LDA   = matrix_A.rows();
        real         ANORM = infinity_norm (matrix_A);
        real         RCOND;
        real*        WORK  = new real [4 * N];
        integer    * IWORK = new integer [N];
        integer      INFO;

        Rgecon (&NORM, N, A, LDA, ANORM, RCOND, WORK, IWORK, INFO);

        delete[] WORK;
        delete[] IWORK;

        if(INFO == -5) {
            std::cerr << "RSLA_MPLAPACK ERROR: Internal computation of infinity-norm encountered an error \n";
            exit(703);
        }

        if(INFO > 0) {
            std::cerr << "RSLA_MPLAPACK ERROR: Condition Number is Inf/NaN \n";
            exit(704);
        }

        if(INFO != 0) {
            std::cerr << "RSLA_MPLAPACK ERROR: Abnormal Internal Error RCOND \n";
            exit(705);
        }

        return RCOND;
    }

    RSMatrix multiply (RSMatrix& matrix_A, RSMatrix& matrix_B) {
        if(matrix_A.cols() != matrix_B.rows()) {
            std::cerr <<  "RSLA_MPLAPACK ERROR: Incompatible Matrix-Matrix Shapes." << "\n";
            exit(700);
        }

        char        TRANSA = 'N';
        char        TRANSB = 'N';
        integer     M      = matrix_A.rows();
        integer     N      = matrix_B.cols();
        integer     K      = matrix_A.cols();
        real        ALPHA  = 1;
        real*       A      = *matrix_A.getmemptr();
        integer     LDA    = M;
        real*       B      = *matrix_B.getmemptr();
        integer     LDB    = K;
        real        BETA   = 0;
        real*       C      = new real [M * N];
        integer     LDC    = M;

        // CALL MPBLAS Rgemm
        Rgemm (&TRANSA, &TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC);

        const RSMatrix matrix_C (C, M, N);

        delete[] C;

        return matrix_C;
    }

    RSVector multiply (RSMatrix& matrix_A, RSVector& vector_B) {
        if(matrix_A.cols() != vector_B.size()) {
            std::cerr <<  "RSLA_MPLAPACK ERROR: Incompatible Matrix-Vector Shapes." << "\n";
            exit(701);
        }

        char        TRANS = 'N';
        integer     M     = matrix_A.rows();
        integer     N     = matrix_A.cols();
        real        ALPHA = 1;
        real*       A     = *matrix_A.getmemptr();
        integer     LDA   = M;
        real*       X     = *vector_B.getmemptr();
        integer     INCX  = 1;
        real        BETA  = 0;
        real*       Y     = new real [M];
        integer     INCY  = 1;

        // Call MPBLAS Rgemv
        Rgemv (&TRANS, M, N, ALPHA, A, LDA, X, INCX, BETA, Y, INCY);

        const RSVector vector_Y (Y, M);

        delete[] Y;

        return vector_Y;
    }

    RSVector fast_solve (RSMatrix& matrix_A, RSVector& vector_b) {
        if(matrix_A.rows() != vector_b.size()) {
            std::cerr <<  "RSLA_MPLAPACK ERROR: Incompatible Matrix-Vector Shapes." << "\n";
            exit(701);
        }

        char        TRANS = 'N';
        integer     M     = matrix_A.rows();
        integer     N     = matrix_A.cols();
        integer     NRHS  = 1;
        real*       A     = new real [M * N];
        integer     LDA   = M;
        real*       B     = new real [M];
        integer     LDB   = vector_b.size();
        real*       WORK  = new real[1];
        integer     LWORK = -1;
        integer     INFO  = 0;

        // xgels is known to overwrite matrices -- therefore copy memory to avoid corruption
        std::copy (*matrix_A.getmemptr(), *matrix_A.getmemptr() + M * N, A);
        std::copy (*vector_b.getmemptr(), *vector_b.getmemptr() + M, B);

        // Workspace Query
        Rgels (&TRANS, M, N, NRHS, A, LDA, B, LDB, WORK, LWORK, INFO);

        LWORK = (integer) WORK [0];

        delete[] WORK;
        WORK = new real [LWORK];

        // Actual Query
        Rgels (&TRANS, M, N, NRHS, A, LDA, B, LDB, WORK, LWORK, INFO);

        delete[] A;
        delete[] WORK;

        if(INFO != 0) {
            std::cout << "RSLA_MPLAPACK ERROR: Solve failed to converge.";
            exit(711);
        }

        const RSVector vector_X (B, N);

        delete[] B;

        return vector_X;
    }

    // Uses DAQ-SVD for solving; see fast_solve for SVD-free solver
    RSVector solve (RSMatrix& matrix_A, RSVector& vector_b) {
        if(matrix_A.rows() != vector_b.size()) {
            std::cerr <<  "RSLA_MPLAPACK ERROR: Incompatible Matrix-Vector Shapes." << "\n";
            exit(701);
        }

        integer      M     = matrix_A.rows();
        integer      N     = matrix_A.cols();
        integer      NRHS  = 1;
        real*        A     = new real [M * N];
        integer      LDA   = M;
        real*        B     = new real [M];
        integer      LDB   = vector_b.size();
        real*        S     = new real [(M < N) ? M : N];
        real         RCOND = -1;
        integer      RANK  = 0;
        real*        WORK  = new real [1];
        integer      LWORK = -1;
        integer    * IWORK = new integer[1];
        integer      INFO  = 0;

        // xgelsd is known to overwrite matrices -- therefore copy memory to avoid corruption
        std::copy(*matrix_A.getmemptr(), *matrix_A.getmemptr() + M * N, A);
        std::copy(*vector_b.getmemptr(), *vector_b.getmemptr() + M, B);

        Rgelsd (M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, IWORK, INFO);

        LWORK = (integer) WORK [0];
        const uinteger LIWORK = IWORK [0];
        delete[] WORK;
        delete[] IWORK;
        WORK = new real [LWORK];
        IWORK = new integer [LIWORK];

        Rgelsd (M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, IWORK, INFO);

        delete[] A;
        delete[] S;
        delete[] WORK;
        delete[] IWORK;

        if(INFO != 0) {
            std::cout << "RSLA_MPLAPACK ERROR: Solve failed to converge.";
            exit(711);
        }

        const RSVector vector_X (B, N);

        delete[] B;

        return vector_X;
    }

    RSEigenResult eigen (RSMatrix& matrix_A) {
        if(matrix_A.rows() != matrix_A.cols()) {
            std::cerr << "RSLA_MPLAPACK ERROR: Eigenvalue decomposition over non-square matrices not possible. \n";
            exit(710);
        }

        char        JOBVL  = 'N';
        char        JOBVR  = 'V';
        integer     N      = matrix_A.rows();
        real*       A      = new real [N * N];
        integer     LDA    = N;
        real*       WR     = new real [N];
        real*       WI     = new real [N];
        real*       VL     = new real [1];
        integer     LDVL   = 1;
        real*       VR     = new real[N * N];
        integer     LDVR   = N;
        real*       WORK   = new real [1];
        integer     LWORK  = -1;
        integer     INFO;

        // xgeev is known to overwrite matrices -- therefore copy matrices to avoid corruption
        std::copy (*matrix_A.getmemptr(), *matrix_A.getmemptr() + N * N, A);

        // Workspace query
        Rgeev (&JOBVL, &JOBVR, N, A, LDA, WR, WI, VL, LDVL, VR, LDVR, WORK, LWORK, INFO);

        LWORK = (integer) WORK[0];
        delete[] WORK;
        WORK = new real [LWORK];

        // Actual query
        Rgeev (&JOBVL, &JOBVR, N, A, LDA, WR, WI, VL, LDVL, VR, LDVR, WORK, LWORK, INFO);

        delete[] A;
        delete[] WORK;

        if(INFO != 0) {
            std::cerr << "RSLA_MPLAPACK ERROR: Eigenvalue decomposition failed! \n";
            exit(711);
        }

        RSVector eigenvalues_real (N);
        RSVector eigenvalues_imag (N);
        RSMatrix eigenvector_real (N, N);
        RSMatrix eigenvector_imag (N, N);

        for(integer i = 0; i < N; i++) {
            real rez = WR[i];
            eigenvalues_real.set(i, rez);
            for(integer j = 0; j < N; j++) eigenvector_real.set(j, i, VR[j * N + i]);
            real imz = WI[i];
            if(imz != 0) {
                eigenvalues_imag.set(i, imz);
                eigenvalues_imag.set(i + 1, -1 * imz);
                for(integer j = 0; j < N; j++) {
                    eigenvector_imag.set(j, i, VR[j * N + i + 1]);
                    eigenvector_imag.set(j, i + 1, -1 * VR[j * N + i + 1]);
                }
                i++;
            }
        }

        delete[] WR;
        delete[] WI;
        delete[] VR;
        delete[] VL;

        return {eigenvalues_real, eigenvalues_imag, eigenvector_real, eigenvector_imag};
    }

    RSEigenResult eigen (RSMatrix& matrix_A, RSMatrix& matrix_B) {
        if(matrix_A.rows() != matrix_A.cols() || matrix_A.rows() != matrix_B.rows() || matrix_B.rows() != matrix_B.cols()) {
            std::cerr << "RSLA_MPLAPACK ERROR: Eigenvalue decomposition over non-square matrices not possible. \n";
            exit(710);
        }

        char        JOBVL  = 'N';
        char        JOBVR  = 'V';
        integer     N      = matrix_A.rows();
        real*       A      = new real [N * N];
        integer     LDA    = N;
        real*       B      = new real [N * N];
        integer     LDB    = N;
        real*       ALPHAR = new real [N];
        real*       ALPHAI = new real [N];
        real*       BETA   = new real [N];
        real*       VL     = new real [1];
        integer     LDVL   = 1;
        real*       VR     = new real [N * N];
        integer     LDVR   = N;
        real*       WORK   = new real [1];
        integer     LWORK  = -1;
        integer     INFO;

        // xggev is known to overwrite matrices -- therefore copy matrices to avoid corruption
        std::copy(*matrix_B.getmemptr(), *matrix_B.getmemptr() + N * N, B);
        std::copy(*matrix_A.getmemptr(), *matrix_A.getmemptr() + N * N, A);

        // Workspace query
        Rggev3 (&JOBVL, &JOBVR, N, A, LDA, B, LDB, ALPHAR, ALPHAI, BETA, VL, LDVL, VR, LDVR, WORK, LWORK, INFO);

        LWORK = integer (WORK[0]);
        delete[] WORK;
        WORK = new real [LWORK];

        Rggev3 (&JOBVL, &JOBVR, N, A, LDA, B, LDB, ALPHAR, ALPHAI, BETA, VL, LDVL, VR, LDVR, WORK, LWORK, INFO);

        delete[] A;
        delete[] B;
        delete[] WORK;

        if(INFO != 0) {
            std::cerr << "RSLA_MPLAPACK ERROR: Eigenvalue decomposition failed! \n";
            exit(711);
        }

        const RSVector eigenvalues_real (N);
        const RSVector eigenvalues_imag (N);
        const RSMatrix eigenvector_real (N, N);
        const RSMatrix eigenvector_imag (N, N);

        for(uinteger i = 0; i < N; i++) {
            real rez = ALPHAR[i] / BETA[i];
            eigenvalues_real.set(i, rez);
            for(integer j = 0; j < N; j++) eigenvector_real.set(j, i, VR[j * N + i]);
            real imz = ALPHAI[i] / BETA[i];
            if(imz != 0) {
                eigenvalues_imag.set(i, imz);
                eigenvalues_imag.set(i + 1, -1 * imz);
                for(integer j = 0; j < N; j++) {
                    eigenvector_imag.set(j, i, VR[j * N + i + 1]);
                    eigenvector_imag.set(j, i + 1, -1 * VR[j * N + i + 1]);
                }
                i++;
            }
        }

        delete[] ALPHAR;
        delete[] ALPHAI;
        delete[] BETA;
        delete[] VR;
        delete[] VL;

        return {eigenvalues_real, eigenvalues_imag, eigenvector_real, eigenvector_imag};

    }

    // TODO Experiment with dgedxxx routines (not wrapped by LAPACKE or MPLAPACK)

#endif

}