

// 1. Data Block: Declares the data Stan expects from R
data {
  int<lower=1> n;       		   // Number of trials (must be at least 1)
  array[n] int<lower=0, upper=1> h; // Array 'h' of length 'n' containing choices (0 or 1)
  array[n] int<lower=0, upper=1> h_opponent; // Array 'h' of length 'n' containing opponents choices (0 or 1)
  array[n] int<lower=0, upper=1> feedback; // Array 'h' of length 'n' containing opponents choices (0 or 1)
  real <lower = 0> prior_alpha_a; // added prior variable for alpha - it might be good to add for tau as well
  real <lower = 0> prior_alpha_b;
  
}

parameters {
real <lower=0, upper = 1> alpha; 	    	  //not a vector → Learning rate is fixed (1 no.)
real <lower=0, upper = 1> tau;		  //not a vector → no change in outcome after 20
}

transformed parameters {
real <lower=0, upper=20> tau_scaled = tau*20;   //rescale tau to 0-20
}


model {
  array[n] real V;

  // Prior: Our belief about alpha and tau *before* seeing the data.
  // 'target +=' adds the log-probability density to the overall model log-probability.

  target += beta_lpdf(alpha | prior_alpha_a, prior_alpha_b); // lpdf = log probability density function
  target += beta_lpdf(tau | 2, 8);


  // Likelihood: How the data 'h' depend on the parameter 'theta'.
  // The model assesses how likely the observed sequence 'h' is given a value of 'theta'.

  V[1] = 0.5;
  target += bernoulli_lpmf(h[1]| 0.5); // lpmf = log probability mass function (for discrete data)

  for (t in 2:n){
    V[t]=V[t-1]+alpha*(feedback[t-1]-V[t-1]); // reward and expected value are both for picking right
    real p=1/(1+exp(-tau_scaled*V[t]));

  target += bernoulli_lpmf(h[t]| p); // lpmf = log probability mass function (for discrete data)
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

