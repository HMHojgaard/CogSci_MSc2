
// 1. Data Block: Declares the data Stan expects from R
data {
  int<lower=1> n;       		   // Number of trials (must be at least 1)
  array[n] int<lower=0, upper=1> h; // Array 'h' of length 'n' containing choices (0 or 1)
  array[n] int<lower=0, upper=1> h_opponent; // Array 'h' of length 'n' containing opponents choices (0 or 1)
  array[n] int<lower=0, upper=1> feedback; // Array 'h' of length 'n' containing opponents choices (0 or 1)
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

  // Prior: Our belief about theta *before* seeing the data.
  // 'target +=' adds the log-probability density to the overall model log-probability.

  target += beta_lpdf(alpha | 1, 1); // lpdf = log probability density function
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

