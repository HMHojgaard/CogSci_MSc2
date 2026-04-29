
// Simple Bayesian Agent (SBA).
// No free parameters — evidence is counted at face value.
// Jeffreys prior pseudo-counts: alpha0 = beta0 = 0.5.
data {
  int<lower=1> N;
  array[N] int<lower=1, upper=8> first_rating;
  array[N] int<lower=1, upper=8> group_rating;
  array[N] int<lower=1, upper=8> second_rating; // changing choice to second rating
}

transformed data {
  array[N] int<lower=0, upper=7> first_rating_transformed;
  array[N] int<lower=0, upper=7> group_rating_transformed;
  array[N] int<lower=0, upper=7> second_rating_transformed;
  
  for (n in 1:N) {
    first_rating_transformed[n]  = first_rating[n]  - 1;
    group_rating_transformed[n]  = group_rating[n]  - 1;
    second_rating_transformed[n] = second_rating[n] - 1;
  }
}

model {
  // Vectorized likelihood with fixed weights = 1
  vector[N] alpha_post = 0.5 + to_vector(first_rating_transformed) + to_vector(group_rating_transformed);
  vector[N] beta_post  = 0.5 + (7 - to_vector(first_rating_transformed))
                             + (7 - to_vector(group_rating_transformed));
                             
  target += beta_binomial_lpmf(second_rating_transformed | 7, alpha_post, beta_post); 
  // 7 choices plus 1 puts us on the scale from 1 to 8
}

generated quantities {
  vector[N] log_lik;
  array[N] int posterior_pred;

  for (i in 1:N) {
    real alpha_post = 0.5 + first_rating_transformed[i] + group_rating_transformed[i];
    real beta_post  = 0.5 + (7 - first_rating_transformed[i]) + (7 - group_rating_transformed[i]);

    log_lik[i]        = beta_binomial_lpmf(second_rating_transformed[i] | 7, alpha_post, beta_post);
    posterior_pred[i] = beta_binomial_rng(7, alpha_post, beta_post) +1;
  }
}

