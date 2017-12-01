#include <cmath>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_linalg.h>

class MultivariateNormal
{
public:
  MultivariateNormal();
  
  void rmvnorm(const gsl_rng *r, 
	       const int n, 
	       const gsl_vector *mean, 
	       const gsl_matrix *var, 
	       gsl_vector *result) const;

  void rinvwishart(const gsl_rng *r,
  		   const int p,
  		   const int n,
  		   const gsl_matrix *var,
  		   gsl_matrix *result) const;
  
  void rwishart(const gsl_rng *r,
		const int p,
		const int n,
		const gsl_matrix *var,
		gsl_matrix *result) const;
  
  double dmvnorm(const int n,
		 const gsl_vector *x, 
		 const gsl_vector *mean, 
		 const gsl_matrix *var) const;
  
  double dmvnorm_log(const int n, 
		     const gsl_vector *x, 
		     const gsl_vector *mean, 
		     const gsl_matrix *var) const;
};

// double dmvnorm_log(const int n, 
// 		   const std::vector<double>& xx, 
// 		   const std::vector<double>& mmean, 
// 		   const gsl_matrix *var);

// arma::vec rmvnorm(const gsl_rng *r,
// 		  const int n,
// 		  arma::vec mean,
// 		  arma::mat var); 

// int rmvt(const gsl_rng *r,
// 	 const int n, 
// 	 const gsl_vector *location, 
// 	 const gsl_matrix *scale, 
// 	 const int dof, 
// 	 gsl_vector *result);

// double dmvt_log(const int n, 
// 		const gsl_vector *x, 
// 		const gsl_vector *location, 
// 		const gsl_matrix *scale, 
// 		const int dof);

// double dmvt_log(const int n, 
// 		const std::vector<double>& xx, 
// 		const std::vector<double>& llocation, 
// 		const gsl_matrix *scale, 
// 		const int dof);
