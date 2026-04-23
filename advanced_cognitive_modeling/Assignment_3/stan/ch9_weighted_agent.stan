
// Weighted Bayesian Agent (raw parameterisation).
// w_d, w_s > 0: independent weights on direct and social evidence.
// Jeffreys prior pseudo-counts (0.5) consistent with SBA.

data {
  int<lower=1> N;
  array[N] int<lower=1, upper=8> first_rating;
  array[N] int<lower=1, upper=8> group_rating;
  array[N] int<lower=1, upper=8> second_rating;
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

parameters {
  real<lower=0, upper=1> rho;  // ratio of weights
  real<lower=0> kappa;  // sum of weights
}

transformed parameters {
  real<lower=0> weight_direct = rho * kappa;  // the weight for own observation is rho times kappa
  real<lower=0> weight_social = (1.0 - rho) * kappa;  // the weight for group observations is 1 minus rho times kappa 
}

model {
  // weakly centered prior for rho
  target += beta_lpdf(rho |  2,2); 
  
  //lognormal prior for kappa, concetration aroun .5 to 5
  target += lognormal_lpdf(kappa | log(2), 0.5); 
  
  // Vectorized likelihood
  vector[N] alpha_post = 0.5 + weight_direct * to_vector(first_rating_transformed)
                             + weight_social * to_vector(group_rating_transformed);
  vector[N] beta_post  = 0.5 + weight_direct * (7 - to_vector(first_rating_transformed))
                             + weight_social * (7 - to_vector(group_rating_transformed));
                             
  target += beta_binomial_lpmf(second_rating_transformed | 7, alpha_post, beta_post);
}

generated quantities {
  vector[N] log_lik;
  array[N] int prior_pred;
  array[N] int posterior_pred;
  real lprior;
  //accumulator for joint prior log density required by priorsense
  lprior = beta_lpdf(rho | 2,2) + lognormal_lpdf(kappa | log(2),0.5);
  
  //prior samples for predictive checks
  real rho_prior = beta_rng(2,2);
  real kappa_prior = lognormal_rng(log(2),0.5);
  real wd_prior = rho_prior * kappa_prior;
  real ws_prior = (1-rho_prior) * kappa_prior;
  
  
  for (i in 1:N) {
  // posterior prediction
    real alpha_post = 0.5 + weight_direct * first_rating_transformed[i] + weight_social * group_rating_transformed[i];
    real beta_post  = 0.5 + weight_direct * (7 - first_rating_transformed[i]) 
                         + weight_social * (7 - group_rating_transformed[i]);

    log_lik[i]        = beta_binomial_lpmf(second_rating_transformed[i] | 7, alpha_post, beta_post);
    posterior_pred[i] = beta_binomial_rng(7, alpha_post, beta_post) + 1;
    
  //prior predictions using sampled prior weights
    real ap = 0.5 + wd_prior * first_rating_transformed[i] + ws_prior * group_rating_transformed[i];
    real bp = 0.5 + wd_prior * (7 - first_rating_transformed[i]) + ws_prior * (7 - group_rating_transformed[i]);
    prior_pred[i] = beta_binomial_rng(7, ap, bp) + 1;
  }
}

