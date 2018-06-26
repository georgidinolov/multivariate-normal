#include "MultivariateNormal.hpp"

MultivariateNormal::MultivariateNormal()
{}

void MultivariateNormal::rmvnorm(const gsl_rng *r,
				 const int n,
				 const gsl_vector *mean,
				 const gsl_matrix *var,
				 gsl_vector *result) const {
  /* multivariate normal distribution random number generator */
  /*
   *	n	dimension of the random vetor
   *	mean	vector of means of size n
   *	var	variance matrix of dimension n x n
   *	result	output variable with a sigle random vector normal distribution generation
   */
  int k;
  gsl_matrix *work = gsl_matrix_alloc(n,n);

  gsl_matrix_memcpy(work,var);
  gsl_linalg_cholesky_decomp(work);

  for(k=0; k<n; k++) {
    gsl_vector_set( result, k, gsl_ran_ugaussian(r) );
  }

  gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, work, result);
  gsl_vector_add(result,mean);

  gsl_matrix_free(work);
}

void MultivariateNormal::rinvwishart(const gsl_rng *r,
				     const int p,
				     const int n,
				     const gsl_matrix *var,
				     gsl_matrix *result) const {
  /* inverse wishart distribution random number generator */
  /*
   *	n	degrees of freedom
   *	p	dimension
   *	var	scale matrix
   *	result	output variable
   * Sampling based on the Bartlett decomposition of the RV
   */

  gsl_matrix * wishart_sample = gsl_matrix_alloc(p,p);
  gsl_matrix * work = gsl_matrix_alloc(p,p);
  gsl_matrix_memcpy(work, var);
  gsl_linalg_cholesky_decomp(work);
  gsl_linalg_cholesky_invert(work);
  
  rwishart(r,
	   p,
	   n,
	   work,
	   wishart_sample);

  // Inverting the sample
  gsl_matrix_memcpy(result, wishart_sample);
  gsl_linalg_cholesky_decomp(result);
  gsl_linalg_cholesky_invert(result);
    
  gsl_matrix_free(wishart_sample);
  gsl_matrix_free(work);
}

void MultivariateNormal::rwishart(const gsl_rng *r,
				  const int p,
				  const int n,
				  const gsl_matrix *var,
				  gsl_matrix *result) const {
  /* wishart distribution random number generator */
  /*
   *	n	degrees of freedom
   *	p	dimension
   *	var	scale matrix
   *	result	output variable
   * Sampling based on the Bartlett decomposition of the RV
   */

  gsl_matrix *work = gsl_matrix_alloc(p,p);
  gsl_matrix_memcpy(work,var);
  gsl_linalg_cholesky_decomp(work);

  // this way we don't need to worry about freeing the matrix
  double A_array [p*p];
  gsl_matrix_view A_view = gsl_matrix_view_array(A_array, p, p);
  gsl_matrix * A = &A_view.matrix;
  gsl_matrix_set_zero(A);

  for (int i=0; i<p; ++i) {
    for (int j=0; j<=i; ++j) {

      // sample from the chi-square distribution
      if (j == i) {
  	double degree_freedom = n - (i+1) + 1;
  	gsl_matrix_set(A, i,j,
  		       sqrt(gsl_ran_chisq(r, degree_freedom)));
      } else {
  	gsl_matrix_set(A, i,j,
  		       gsl_ran_ugaussian(r));
      }
    }
  }

  // Compute A * A^T
  double AAt_array [p*p]; // this is the target location of the product
  gsl_matrix_view AAt_view = gsl_matrix_view_array(AAt_array, p, p);
  gsl_matrix * AAt = &AAt_view.matrix;
  gsl_matrix_memcpy(AAt, A);

  // AAt = 1.0 * A (which in this case is initially contained in location pointed to by AAt)  * A^T
  gsl_blas_dtrmm(CblasRight, CblasLower, CblasTrans, CblasNonUnit,
  		 1.0, // alpha = 1.0
  		 A, AAt);

  gsl_matrix * LAAt = AAt;
  // LAAt = 1.0 * L * AAt (this is target location as well)
  gsl_blas_dtrmm(CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
  		 1.0, // alpha = 1.0
  		 work, LAAt);

  gsl_matrix_memcpy(result, LAAt);
  // LAAtLt = 1.0 * LAAt * Lt
  gsl_blas_dtrmm(CblasRight, CblasLower, CblasTrans, CblasNonUnit,
  		 1.0, // alpha = 1.0
  		 work, result);
  
  gsl_matrix_free(work);
}

// arma::vec rmvnorm(const gsl_rng *r,
// 		  const int n,
// 		  arma::vec mean,
// 		  arma::mat var)
// {
//   gsl_vector * mu = gsl_vector_alloc(n);
//   gsl_vector * result = gsl_vector_alloc(n);
//   gsl_matrix * Sigma = gsl_matrix_alloc(n,n);

//   // Assigning mean and covariance to gsl objects memory
//   for (int k=0; k<n; ++k) {
//     gsl_vector_set(mu, k, mean(k));
//     for (int l=0; l<n; ++l) {
//       gsl_matrix_set(Sigma,k,l,var(k,l));
//     }
//   }

//   rmvnorm(r, n, mu, Sigma, result);

//   arma::vec output = arma::vec (n);
//   for (int k=0; k<n; ++k) {
//     output(k) = gsl_vector_get(result, k);
//   }

//   gsl_vector_free(mu);
//   gsl_vector_free(result);
//   gsl_matrix_free(Sigma);
//   return output;
// }

double MultivariateNormal::dmvnorm(const int n,
				   const gsl_vector *x,
				   const gsl_vector *mean,
				   const gsl_matrix *var) const {
  /* multivariate normal density function    */
  /*
   *	n	dimension of the random vetor
   *	mean	vector of means of size n
   *	var	variance matrix of dimension n x n
   */
  int s;
  double ax,ay;
  gsl_vector *ym, *xm;
  gsl_matrix *work = gsl_matrix_alloc(n,n),
    *winv = gsl_matrix_alloc(n,n);
  gsl_permutation *p = gsl_permutation_alloc(n);

  gsl_matrix_memcpy( work, var );
  gsl_linalg_LU_decomp( work, p, &s );
  gsl_linalg_LU_invert( work, p, winv );
  ax = gsl_linalg_LU_det( work, s );
  gsl_matrix_free( work );
  gsl_permutation_free( p );

  xm = gsl_vector_alloc(n);
  gsl_vector_memcpy( xm, x);
  gsl_vector_sub( xm, mean );
  ym = gsl_vector_alloc(n);
  gsl_blas_dsymv(CblasUpper,1.0,winv,xm,0.0,ym);
  gsl_matrix_free( winv );
  gsl_blas_ddot( xm, ym, &ay);
  gsl_vector_free(xm);
  gsl_vector_free(ym);
  ay = exp(-0.5*ay)/sqrt( pow((2*M_PI),n)*ax );

  return ay;
}

double MultivariateNormal::log_dmvnorm(const int n,
				   const gsl_vector *x,
				   const gsl_vector *mean,
				   const gsl_matrix *var) const {
  /* multivariate normal density function    */
  /*
   *	n	dimension of the random vetor
   *	mean	vector of means of size n
   *	var	variance matrix of dimension n x n
   */
  int s;
  double ln_ax,ay;
  gsl_vector *ym, *xm;
  gsl_matrix *work = gsl_matrix_alloc(n,n),
    *winv = gsl_matrix_alloc(n,n);
  gsl_permutation *p = gsl_permutation_alloc(n);

  gsl_matrix_memcpy( work, var );
  gsl_linalg_LU_decomp( work, p, &s );
  gsl_linalg_LU_invert( work, p, winv );
  ln_ax = gsl_linalg_LU_lndet( work );
  gsl_matrix_free( work );
  gsl_permutation_free( p );

  xm = gsl_vector_alloc(n);
  gsl_vector_memcpy( xm, x);
  gsl_vector_sub( xm, mean );
  ym = gsl_vector_alloc(n);
  gsl_blas_dsymv(CblasUpper,1.0,winv,xm,0.0,ym);

  // printf("ym = ");
  // for (unsigned i=0; i<n; ++i) {
  //   printf("%f ", gsl_vector_get(ym,i));
  // }
  // printf("\n");
  // printf("cov = ");
  // for (unsigned i=0; i<n; ++i) {
  //   for (unsigned j=0; j<n; ++j) {
  //     printf("%f ", gsl_matrix_get(var,i,j));
  //   }
  //   printf("\n");
  // }

  gsl_matrix_free( winv );
  gsl_blas_ddot( xm, ym, &ay);
  gsl_vector_free(xm);
  gsl_vector_free(ym);

  ay = -0.5*ay - 0.5* ( n*log(2*M_PI) + ln_ax );
  //  printf("ay = %f; ln_ax = %f;\n", ay, ln_ax);
  return ay;
}

double MultivariateNormal::dmvnorm_log(const int n,
				       const gsl_vector *x,
				       const gsl_vector *mean,
				       const gsl_matrix *var) const
{
  /* multivariate normal density function    */
  /*
   *	n	dimension of the random vetor
   *	mean	vector of means of size n
   *	var	variance matrix of dimension n x n
   */
  double out = 0;
  if ((n==1) || (n>2)) {
    int s;
    double ax,ay;
    gsl_vector *ym, *xm;
    gsl_matrix *work = gsl_matrix_alloc(n,n),
      *winv = gsl_matrix_alloc(n,n);
    gsl_permutation *p = gsl_permutation_alloc(n);

    gsl_matrix_memcpy( work, var );
    gsl_linalg_LU_decomp( work, p, &s );
    gsl_linalg_LU_invert( work, p, winv );
    ax = gsl_linalg_LU_det( work, s );
    gsl_matrix_free( work );
    gsl_permutation_free( p );

    xm = gsl_vector_alloc(n);
    gsl_vector_memcpy( xm, x);
    gsl_vector_sub( xm, mean );
    ym = gsl_vector_alloc(n);
    gsl_blas_dsymv(CblasUpper,1.0,winv,xm,0.0,ym);
    gsl_matrix_free( winv );
    gsl_blas_ddot( xm, ym, &ay);
    gsl_vector_free(xm);
    gsl_vector_free(ym);
    ay = (-0.5*ay)- 0.5*log( pow((2*M_PI),n)*ax );
    out = ay;
  } else {

    double sigma_x = std::sqrt(gsl_matrix_get(var,0,0));
    double sigma_y = std::sqrt(gsl_matrix_get(var,1,1));
    double rho = gsl_matrix_get(var,0,1)/(sigma_x*sigma_y);

    double xx = gsl_vector_get(x,0);
    double yy = gsl_vector_get(x,1);

    double x0 = gsl_vector_get(mean,0);
    double y0 = gsl_vector_get(mean,1);

    out =
      -1.0*(log(2*M_PI) +
	    log(sigma_x) +
	    log(sigma_y) +
	    0.5*log(1-rho*rho)) +
      -0.5/(1.0-rho*rho) *
      ( (xx-x0)*(xx-x0)/(sigma_x*sigma_x) +
	(yy-y0)*(yy-y0)/(sigma_y*sigma_y) -
	2.0*rho*(xx-x0)*(yy-y0)/(sigma_x*sigma_y) );
  }

  return out;
}

// double dmvnorm_log(const int n,
// 		   const std::vector<double>& xx,
// 		   const std::vector<double>& mmean,
// 		   const gsl_matrix *var){
//   /* multivariate normal density function    */
//   /*
//    *	n	dimension of the random vetor
//    *	mean	vector of means of size n
//    *	var	variance matrix of dimension n x n
//    */

//   int s;
//   double ax,ay;
//   gsl_vector *ym, *xm;

//   gsl_vector *x = gsl_vector_alloc(n);
//   gsl_vector *mean = gsl_vector_alloc(n);
//   for (int i=0; i<n; ++i) {
//     gsl_vector_set(x, i, xx[i]);
//     gsl_vector_set(mean, i, mmean[i]);
//   }

//   gsl_matrix *work = gsl_matrix_alloc(n,n),
//     *winv = gsl_matrix_alloc(n,n);
//   gsl_permutation *p = gsl_permutation_alloc(n);

//   gsl_matrix_memcpy( work, var );
//   gsl_linalg_LU_decomp( work, p, &s );
//   gsl_linalg_LU_invert( work, p, winv );
//   ax = gsl_linalg_LU_det( work, s );
//   gsl_matrix_free( work );
//   gsl_permutation_free( p );

//   xm = gsl_vector_alloc(n);
//   gsl_vector_memcpy( xm, x);
//   gsl_vector_sub( xm, mean );
//   ym = gsl_vector_alloc(n);
//   gsl_blas_dsymv(CblasUpper,1.0,winv,xm,0.0,ym);
//   gsl_matrix_free( winv );
//   gsl_blas_ddot( xm, ym, &ay);
//   gsl_vector_free(xm);
//   gsl_vector_free(ym);
//   ay = -0.5*ay - 0.5*(log( pow((2*M_PI),n) ) +
// 		      log( ax ));

//   gsl_vector_free(x);
//   gsl_vector_free(mean);

//   return ay;
// }

// int rmvt(const gsl_rng *r,
// 	 const int n,
// 	 const gsl_vector *location,
// 	 const gsl_matrix *scale,
// 	 const int dof,
// 	 gsl_vector *result){
//   /* multivariate Student t distribution random number generator */
//   /*
//    *	n	 dimension of the random vetor
//    *	location vector of locations of size n
//    *	scale	 scale matrix of dimension n x n
//    *	dof	 degrees of freedom
//    *	result	 output variable with a single random vector normal distribution generation
//    */
//   int k;
//   gsl_matrix *work = gsl_matrix_alloc(n,n);
//   double ax = 0.5*dof;

//   ax = gsl_ran_gamma(r,ax,(1/ax));     /* gamma distribution */

//   gsl_matrix_memcpy(work,scale);
//   gsl_matrix_scale(work,(1/ax));       /* scaling the matrix */
//   gsl_linalg_cholesky_decomp(work);

//   for(k=0; k<n; k++)
//     gsl_vector_set( result, k, gsl_ran_ugaussian(r) );

//   gsl_blas_dtrmv(CblasLower, CblasNoTrans, CblasNonUnit, work, result);
//   gsl_vector_add(result, location);

//   gsl_matrix_free(work);

//   return 0;
// }

// double dmvt_log(const int n,
// 		const gsl_vector *x,
// 		const gsl_vector *location,
// 		const gsl_matrix *scale,
// 		const int dof){
//   /* multivariate Student t density function */
//   /*
//    *	n	 dimension of the random vetor
//    *	location vector of locations of size n
//    *	scale	 scale matrix of dimension n x n
//    *	dof	 degrees of freedom
//    */
//   int s;
//   double ax,ay,az=0.5*(dof + n);
//   gsl_vector *ym, *xm;
//   gsl_matrix *work = gsl_matrix_alloc(n,n),
//     *winv = gsl_matrix_alloc(n,n);
//   gsl_permutation *p = gsl_permutation_alloc(n);

//   gsl_matrix_memcpy( work, scale );
//   gsl_linalg_LU_decomp( work, p, &s );
//   gsl_linalg_LU_invert( work, p, winv );
//   ax = gsl_linalg_LU_det( work, s );
//   gsl_matrix_free( work );
//   gsl_permutation_free( p );

//   xm = gsl_vector_alloc(n);
//   gsl_vector_memcpy( xm, x);
//   gsl_vector_sub( xm, location );
//   ym = gsl_vector_alloc(n);
//   gsl_blas_dsymv(CblasUpper,1.0,winv,xm,0.0,ym);
//   gsl_matrix_free( winv );
//   gsl_blas_ddot( xm, ym, &ay);
//   gsl_vector_free(xm);
//   gsl_vector_free(ym);

//   ay = -az*log(1+ay/dof) +
//     gsl_sf_lngamma(az) -
//     gsl_sf_lngamma(0.5*dof) -
//     0.5*( n*log(dof) + n*log(M_PI) + log(ax) );

//   return ay;
// }

// double dmvt_log(const int n,
// 		const std::vector<double>& xx,
// 		const std::vector<double>& llocation,
// 		const gsl_matrix *scale,
// 		const int dof){
//   /* multivariate Student t density function */
//   /*
//    *	n	 dimension of the random vetor
//    *	location vector of locations of size n
//    *	scale	 scale matrix of dimension n x n
//    *	dof	 degrees of freedom
//    */
//   int s;
//   double ax,ay,az=0.5*(dof + n);
//   gsl_vector *ym, *xm;

//   gsl_vector *x = gsl_vector_alloc(n);
//   gsl_vector *location = gsl_vector_alloc(n);
//   for (int i=0; i<n; ++i) {
//     gsl_vector_set(x, i, xx[i]);
//     gsl_vector_set(location, i, llocation[i]);
//   }

//   gsl_matrix *work = gsl_matrix_alloc(n,n),
//     *winv = gsl_matrix_alloc(n,n);
//   gsl_permutation *p = gsl_permutation_alloc(n);

//   gsl_matrix_memcpy( work, scale );
//   gsl_linalg_LU_decomp( work, p, &s );
//   gsl_linalg_LU_invert( work, p, winv );
//   ax = gsl_linalg_LU_det( work, s );
//   gsl_matrix_free( work );
//   gsl_permutation_free( p );

//   xm = gsl_vector_alloc(n);
//   gsl_vector_memcpy( xm, x);
//   gsl_vector_sub( xm, location );
//   ym = gsl_vector_alloc(n);
//   gsl_blas_dsymv(CblasUpper,1.0,winv,xm,0.0,ym);
//   gsl_matrix_free( winv );
//   gsl_blas_ddot( xm, ym, &ay);
//   gsl_vector_free(xm);
//   gsl_vector_free(ym);

//   ay = -az*log(1+ay/dof) +
//     gsl_sf_lngamma(az) -
//     gsl_sf_lngamma(0.5*dof) -
//     0.5*( n*log(dof) + n*log(M_PI) + log(ax) );

//   gsl_vector_free(x);
//   gsl_vector_free(location);

//   return ay;
// }
