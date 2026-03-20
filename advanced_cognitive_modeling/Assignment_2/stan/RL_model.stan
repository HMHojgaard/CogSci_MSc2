
data {
  int<lower=1> n;
  array[n] int<lower=0, upper=1> h;
  array[n] int<lower=0, upper=1> h_opponent;
  array[n] int<lower=0, upper=1> feedback;
  real<lower=0> prior_alpha_a;
  real<lower=0> prior_alpha_b;
}

parameters {
  real<lower=0, upper=1> alpha;
  real<lower=0, upper=1> tau;
}

transformed parameters {
  real<lower=0, upper=20> tau_scaled = tau * 20;
}

model {
  array[n] real V;

  target += beta_lpdf(alpha | prior_alpha_a, prior_alpha_b);
  target += beta_lpdf(tau | 2, 8); 

  V[1] = 0.5;
  target += bernoulli_lpmf(h[1] | 0.5);

  for (t in 2:n) {
    V[t] = V[t-1] + alpha * (feedback[t-1] - V[t-1]);
    real p = 1 / (1 + exp(-tau_scaled * V[t]));
    target += bernoulli_lpmf(h[t] | p);
  }
}

generated quantities {

  // Prior samples
  real<lower=0, upper=1> alpha_prior = beta_rng(prior_alpha_a, prior_alpha_b);
  real<lower=0, upper=1> tau_prior   = beta_rng(2, 8); 
  real<lower=0, upper=20> tau_prior_scaled = tau_prior * 20;

  // Prior Predictive Check 
  array[n] real V_prior; // stores V at each trial while computing prior simulation
  array[n] int  h_prior_rep; // stores simulated choices while computing prior simulation

  V_prior[1] = 0.5;
  h_prior_rep[1] = bernoulli_rng(0.5);

  for (t in 2:n) {
    V_prior[t] = V_prior[t-1] + alpha_prior * (feedback[t-1] - V_prior[t-1]);
    real p_prior = 1 / (1 + exp(-tau_prior_scaled * V_prior[t]));
    h_prior_rep[t] = bernoulli_rng(p_prior);
  }

  // total right choices from prior simulation (used for histo)
  int prior_rep_sum = sum(h_prior_rep); 

  // Posterior Predictive Check 
  // same structure, but now with the posterior values of alpha and tau
  array[n] real V_post;
  array[n] int  h_post_rep;

  V_post[1] = 0.5;
  h_post_rep[1] = bernoulli_rng(0.5);

  for (t in 2:n) {
    V_post[t] = V_post[t-1] + alpha * (feedback[t-1] - V_post[t-1]);
    real p_post = 1 / (1 + exp(-tau_scaled * V_post[t]));
    h_post_rep[t] = bernoulli_rng(p_post);
  }

  int post_rep_sum = sum(h_post_rep);

  // Log-likelihood
  array[n] real log_lik;
  log_lik[1] = bernoulli_lpmf(h[1] | 0.5);
  for (t in 2:n) {
    real V_ll;
    V_ll = 0.5;
    for (s in 2:t) {
      V_ll = V_ll + alpha * (feedback[s-1] - V_ll);
    }
    real p_ll = 1 / (1 + exp(-tau_scaled * V_ll));
    log_lik[t] = bernoulli_lpmf(h[t] | p_ll);
  }
}

