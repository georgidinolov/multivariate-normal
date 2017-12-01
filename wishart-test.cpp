#include <iostream>
#include "MultivariateNormal.hpp"

int main() {
  long unsigned seed = 1;
  const gsl_rng_type *Type;
  gsl_rng_env_setup();
  Type = gsl_rng_default;
  gsl_rng * r_ptr = gsl_rng_alloc(Type);
  gsl_rng_set(r_ptr, seed);
  
  MultivariateNormal mtvn = MultivariateNormal();

  long unsigned dimension = 13;
  gsl_matrix *covariance = gsl_matrix_alloc(dimension, dimension);
  
  for (unsigned i=0; i<dimension; ++i) {
    gsl_matrix_set(covariance, i, i, 1.0);
    for (unsigned j=i+1; j<dimension; ++j) {
      gsl_matrix_set(covariance, i, j, 0.5);
      gsl_matrix_set(covariance, j, i, 0.5);
    }
  }

  for (unsigned i=0; i<dimension; ++i) {
    for (unsigned j=0; j<dimension; ++j) {

      if (j < dimension-1) {
  	std::cout << gsl_matrix_get(covariance, i,j) << " ";
      } else {
  	std::cout << gsl_matrix_get(covariance, i,j) << "\n";
      }
    }
  }

  // 10 samples from the Wishart distribution.
  unsigned N_samples = 1e6;
  gsl_matrix * result = gsl_matrix_alloc(dimension, dimension);
  gsl_matrix * sum_result = gsl_matrix_alloc(dimension, dimension);
  gsl_matrix * sum_result_inverse = gsl_matrix_alloc(dimension, dimension);

  for (unsigned i=0; i<N_samples; ++i) {

    mtvn.rwishart(r_ptr,
  		  dimension,
  		  dimension + 2,
  		  covariance,
  		  result);
    gsl_matrix_add(sum_result, result);

    mtvn.rinvwishart(r_ptr,
		     dimension,
		     dimension + 2,
		     covariance,
		     result);
    gsl_matrix_add(sum_result_inverse, result);
  }
  gsl_matrix_scale(sum_result, 1.0/N_samples);
  gsl_matrix_scale(sum_result_inverse, 1.0/N_samples);

  for (unsigned i=0; i<dimension; ++i) {
    for (unsigned j=0; j<dimension; ++j) {

      if (j < dimension-1) {
  	std::cout << gsl_matrix_get(sum_result, i,j) << " ";
      } else {
  	std::cout << gsl_matrix_get(sum_result, i,j) << "\n";
      }
    }
  }

  for (unsigned i=0; i<dimension; ++i) {
    for (unsigned j=0; j<dimension; ++j) {

      if (j < dimension-1) {
  	std::cout << gsl_matrix_get(sum_result_inverse, i,j) << " ";
      } else {
  	std::cout << gsl_matrix_get(sum_result_inverse, i,j) << "\n";
      }
    }
  }
  									    
  gsl_matrix_free(covariance);
  gsl_matrix_free(result);
  gsl_matrix_free(sum_result);
  gsl_rng_free(r_ptr);
  return 0;
}
