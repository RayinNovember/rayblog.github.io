---
title: "Bayesian Analysis - Idea and Sampling Methods"
output: 
  md_document:
    latex_engine: xelatex
    toc: true
    toc_float:
      toc_collapsed: true
    toc_depth: 4
    theme: united
date: "2024-05-17"
---





## What is Bayesian Analysis?

The paradigm of Bayesian analysis can be summarized as: For each possible explanation of the sample, count all the ways the sample could happen, explanations with more ways to produce the sample are more plausible.

In Bayesian statistics, plausibility of the explanations is called posterior:

`\(\underbrace {f(\theta|data)}_{posterior} =\underbrace {f(data|\theta)}_{likelihood}\underbrace{f(\theta)}_{prior}/\underbrace{f(data)}_{marginal\\likelihood}\)`

Here, `\(\theta\)` represents the a possible explanation for the data. Specifically, it is the set of parameters that define the underlying data-generating process. 

Posterior answers the following question: How likely is a particular underlying data-generating mechanism given the data? In other words, it evaluates how plausible each data-generating mechanism is given the data.

Posterior consists of three parts:

(1). Prior: 

Prior is the probability distribution of data-generating mechanisms. It embodies our belief in the credibility of possible data-generating mechanisms and can be thought of as weights of the mechanisms before data is considered. 

(2). Likelihood: 

Probability of the data with one particular data-generating mechanism

(3). Marginal Likelihood:

The marginal likelihood, `\(f(data) = \int f(data|\theta)f(\theta)d\theta\)`, is the probability of the data averaged across all data mechanisms. It's calculated as follows: For each possible data-generating mechanism, check probability of the mechanism (prior), multiply it by the probability of observing the observed data with the mechanism (likelihood), and add the results together for all data-generating mechanisms. 

Back to the expression of posterior, the numerator `\(f(data|\theta)f(\theta)\)` is the probability of observing the data under one mechanism; the denominator is a sum of all `\(f(data|\theta)f(\theta)\)` quantities. Therefore, the posterior represents the proportion of one mechanism out of all mechanisms. 

Probabilities are just scaled counts so that they add up to one. So an intuitive explanation is that the posterior of one data-generating mechanism represents the proportion of ways it generates data out of all possible ways of generating the observed data by other mechanisms. Mechanisms with more ways to generate the data are more plausible. 

With the posterior distribution, one can tell which possible parameter(s) as specified in the prior are more plausible for explaining the observed data. Therefore, the posterior distribution can be seen as a distribution of plausibility over all possibilities of the data-generating parameters.

It is the pursuit of the posterior distribution that lies at the heart of the Bayesian paradigm. 

For a comparison between the Bayesian approach and the Frequentist approach, we review the steps each school takes for statistical inference.

Frequentist inference involves the following steps:

(1) Yielding a point estimate with Maximum Likelihood Estimation(MLE) from the sample 

(2) Getting standard error of the MLE estimator

(3) Relying on the Central Limit Theorem to decide whether to reject the parameter value specified in the null hypothesis

Bayesian inference involves the following steps: 

(1) Establishing a model and posterior distribution for the targets of interest

(2) Generating samples from the posterior distribution 

(3) Making inferences based on the sampled posterior distribution

So the posterior distribution is all that Bayesians want. But how do we get it?

## Why Sampling?

There are three options to get the posterior distribution which is expressed as:

`\(p(\theta|X) = \dfrac{p(X|\theta)P(\theta)}{\int p(X|\theta)P(\theta)d\theta}\)`

(1) Solve the integral in the denominator numerically. 

This is rarely the case in practice. 

(2) Approximate the integral in the denominator. 

Often time the integral can not be solved but can be approximated. A variety of approximation methods are available. But such approximation methods are limited as they do not scale well as the number of parameters increases. 

(3) Sampling from the posterior distribution.

The limits of the first two options leave us with the third option - we draw samples from the posterior distribution to form a close representaion of the actual posterior distribution.

The third option may sound counter-intuitive. After all, how do we draw samples from a distribution that we don't know?  

## How Markov Chain Monte Carlo (MCMC) Works

It turns out that we don't need the actual posterior distribution to draw samples from it. Markov Chain Monte Carlo (MCMC) offers a solution.

The idea behind Markov Chain Monte Carlo(MCMC) is to construct a Markov Chain whose stationary distribution is the posterior distribution. Once a Markov Chain reaches its stationary distribution, the probabilities of getting different states/values in the chain no longer change and thus the values can be treated as coming from the stationary distribution afterwards. 

So if there's a Markov Chain whose stationary distribution is the posterior distribution we need, we can move through the chain to get samples from the target posterior distribution.

But how do we construct such Markov Chain? What exactly is constructed?

When a Markov Chain reaches its stationary distribution (assuming it exists), the following equation holds:

`\(\pi_iP_{ij}=\pi_jP_{ji}\)`

The equation is called detailed balance equation.

`\(\pi\)` is the stationary distribution.

`\(\pi_i, \pi_j\)` are probabilities of reaching states i and j respectively. 

`\(P_{ij}\)` is the transition probability of state i to state j.

`\(P_{ji}\)` is the probability of going from state j to state i.

In our case, the states are the values of the posterior distribution and the stationary distribution is the posterior distribution. 

`\(\pi_i = p(\theta_i|X)\)` 

`\(\pi_j = p(\theta_j|X)\)`

The detailed balance equation now becomes:

`\(p(\theta_i|X)P_{ij}=p(\theta_j|X)P_{ji}\)`

This still doesn't help much as we don't know `\(p(\theta_i)\)` or `\(p(\theta_j)\)` nor do we know `\(P_{ij}\)` or `\(P_{ji}\)`.

But we do know part of `\(p(\theta_i)\)`. 

The posterior can be expressed as:

`\(p(\theta|X) = f(x)/NC\)`

where `\(f(x) = p(X|\theta)P(\theta)\)` is known and `\(NC=\int p(X|\theta)P(\theta)d\theta\)` is the unknown integral.

Plug in `\(f(i)/NC\)` for both sides of the detailed balance equation, we can see that the unknown integrals cancel each other. we can now use the known function `\(f(x)\)` only for the detailed balance equation:

`\(\dfrac{f(i)}{NC} P_{ij}= \dfrac{f(j)}{NC} P_{ji}\)`

`\(f(i) P_{ij}= f(j) P_{ji}\)` 

Therefore, as long as we figure out a transition rule and consequently, `\(P_{ij}\)` and `\(P_{ji}\)`, we are able to transit through the states of the stationary distribution, meaning we can sample the values of the posterior distribution.

### Metropolis-Hastings (MH) Sampling

The Metropolis-Hastings Algorithm designs the transition rule by breaking the transition between states into two steps: proposal and acceptance.

(1). Proposal

To go to a new state, the new state has to be proposed first. By MH Sampling, the new state is proposed via a known distribution with pdf `\(g(x)\)`. Let `\(g(j|i)\)` denote the probability of proposing state j at state i. Similarly, `\(g(i|j)\)` is the probability of proposing state i at state j. For example, if we use normal distribution for `\(g(x)\)`, then to propose a new state, we would draw a sample from the normal distribution centered at the current state.

(2). Acceptance

The second step is to determine whether the proposal is accepted. The MH Algorithm defines an acceptance function for the purpose:

`\(a(i,j)=\dfrac{\pi_jg(i|j)}{\pi_ig(j|i)} = \dfrac{f(j)g(i|j)}{f(i)g(j|i)}\)`. 

MH Sampling accepts the proposed state j with probability `\(\alpha(i,j) = min \{1,\dfrac{f(j)g(i|j)}{f(i)g(j|i)}\}\)`. 

Similarly, MH Sampling accepts proposed state i with probability `\(\alpha(j,i) = min \{1,\dfrac{f(i)g(j|i)}{f(j)g(i|j)}\}\)`.

The idea behind the `\(min\{\}\)` function is to make sure we always accept the proposal if `\(a(i,j)\)`>=1, and accept the proposal with probability `\(a(i,j)\)` if `\(a(i,j)\)`<1,

Why? Because when `\(a(i,j)>=1\)`, it indicates that `\(f(j)g(i|j)>=f(i)g(j|i)\)`. If the proposal distribution is symmetric, then `\(g(i|j) = g(j|i)\)`, meaning `\(f(j)>=f(i)\)`. We want to sample the more possible outcomes from the posterior distribution.

Now we confirm that the detailed balance equation holds with the transition rule. consider the following two cases:

(1). When `\(f(j)g(i|j)>=f(i)g(j|i)\)` 

`\(\alpha(i,j) = min \{1,\dfrac{f(j)g(i|j)}{f(i)g(j|i)}\}=1\)` 

`\(\alpha(j,i) = min \{1,\dfrac{f(i)g(j|i)}{f(j)g(i|j)}\}=\dfrac{f(i)g(j|i)}{f(j)g(i|j)}\)`

Then the detailed balance equation becomes:

`\(f(i)P_{ij}=f(i)\underbrace{g(j|i)}_{propose} \underbrace{a(i,j)}_{accept}=f(i)g(j|i)=f(j)g(i|j)*\dfrac{f(i)g(j|i)}{f(j)g(i|j)}=f(j)g(i|j)a(j,i)=f(j)P_{ji}\)`

(2). When `\(f(j)g(i|j)<f(i)g(j|i)\)` 

`\(\alpha(j,i)=1\)`

`\(\alpha(i,j) = min \{1,\dfrac{f(j)g(i|j)}{f(i)g(j|i)}\}=\dfrac{f(j)g(i|j)}{f(i)g(j|i)}\)` 

Thus detailed balance equation becomes:

`\(f(i)P_{ij}=f(i)g(j|i)a(i,j)=f(i)g(j|i) \dfrac{f(j)g(i|j)}{f(i)g(j|i)}=f(j)g(i|j)=f(j)g(i|j)a(j,i)=f(j)P_{ji}\)`

Therefore, detailed balance holds with the MH algorithm.

Put together, the MH Algorithm is:

For `\(t=1,2,...,T\)`, do:

(1) Draw `\(\theta_j \sim g(\theta)\)` where `\(g(\theta)\)` is the proposal distribution

(2) Calculate the ratio `\(R = \dfrac{f(\theta_j)g(\theta_i|\theta_j)}{f(\theta_i)g(\theta_j|\theta_i)}\)`

(3) Accept the proposed value with probability `\(\alpha(i,j) = min \{1,\dfrac{f(j)g(i|j)}{f(i)g(j|i)}\}\)`. In programming, this step is achieved via an equivalent step: Generate a random draw u from uniform distribution `\(U(0,1)\)` and compare u with R. If `\(R>u\)`, then accept the proposed value. Otherwise reject the proposed value.

(4) Repeat step (1) through step(3) till enough draws are obtained.

### Gibbs Sampling

Gibbs sampling is a type of MCMC method that is particularly useful at sampling from multivariate distributions. 

The concept behind Gibbs Sampling is that when we need to sample for multiple parameters, we can sample for one parameter `\(\theta_i\)` at a time while keeping other parameters at the current values. 

Suppose there are k parameters, the Gibbs Sampling algorithm is: 

1. Set a vector of length k as the initial values of the k parameters. Denote this vector as `\(\Theta^0\)` (parameter values at round 0)

2. Sample `\(\theta_1^1|\theta_2^0,\theta_3^0,...\theta_k^0\)`

3. Sample `\(\theta_2^1|\theta_1^1,\theta_3^0,...\theta_k^0\)`

4. Sample `\(\theta_3^1|\theta_1^1,\theta_2^1,\theta_4^0,...\theta_k^0\)`

5. Sample `\(\theta_k^1|\theta_1^1,\theta_2^1,,...\theta_{k-1}^1\)`

6. return to step 1. 

Instead of sampling directly from the full joint posterior distribution, Gibbs Sampling finds the conditional posterior distribution for each parameter conditional on all other parameters (this is also called full conditional) and draws samples for one parameter at a time. That is, within a loop, a sample is drawn for each of the parameters using their full conditionals. The results from one complete loop can be seen as a draw from the joint posterior distribution.

In fact, Gibbs sampling is a special case of the MH algorithm. It uses the full conditional distribution of the parameter - `\(g(\theta_1|\theta_2,\theta_3,...Data)\)` as the proposal distribution. 

Gibbs Sampling satisfies detailed balance. Here's why.

Consider the acceptance probability:

`\(\alpha(\theta_i,\theta^*_i) =  min\{1,\dfrac{f(\theta^*_i,\theta_{-i})g(\theta_i|\theta^*_i)}{f(\theta_i,\theta_{-i})g(\theta^*_i|\theta_i)}\}\)`

where `\(\theta_i\)` is the current value of the parameter under consideration;

`\(\theta^*_i\)` the proposed value of the parameter under consideration;

`\(\theta_{-i}\)` are the current values of other parameters.

For the acceptance ratio, the first term of its numerator is: `\(f(\theta^*_i,\theta_{-i})=f(\theta_i^*|\theta_{-i})f(\theta_{-i})\)` by conditional probability.

The second term of the numerator is the proposal function. Since the Gibbs sampler uses full conditional as the proposal function, the proposal distribution does not depend on `\(\theta_i\)` or `\(\theta^*_i\)`, thus we have:

`\(g(\theta_i|\theta^*_i)=g(\theta_i)=f(\theta_i|\theta_{-i})\)`

Similarly, the first term of its denominator is: 

`\(f(\theta_i,\theta_{-i})=f(\theta_i|\theta_{-i})f(\theta_{-i})\)` by conditional probability.

The second term of the denominator is:  

`\(g(\theta_i^*|\theta_i) = g(\theta_i^*) = f(\theta_i^*|\theta_{-i})\)`

Plug in the terms into the acceptance ratio, we have:

`\(\dfrac{f(\theta_i^*|\theta_{-i})f(\theta_{-i})f(\theta_i|\theta_{-i})}{f(\theta_i|\theta_{-i})f(\theta_{-i})f(\theta^*_i|\theta_{-i})}=1\)`

Therefore, the acceptance probability is always 1.

#### Bayesian Linear Regression with Gibbs Sampling

A common application of Gibbs Sampling is the linear regression model. In a linear regression model, response variable is assumed to be a linear combination of a set of predictors plus some normally distributed zero-mean noise: 

`\(y_i = X_i^T\beta+\epsilon_i\)` 

`\(\epsilon_i \sim N(0,\sigma^2)\)` 

Equivalently, `\(y_i \sim N(X_i^T\beta,\sigma^2)\)`

The goal is to estimate the coefficient parameter `\(\beta\)` (intercept included) and the variance of the error term `\(\sigma^2\)`.

The full posterior is: 

`\(p(\theta|Data)\propto p(Data|\theta) * p(\theta)\)`

`\(p(\beta,\sigma^2|X,Y) \propto \underbrace{[\prod_{i=1}^np(y_i|\beta,\sigma^2,X_i)]}_{likelihood}\underbrace{p(\beta)p(\sigma^2)}_{priors}\)`

For Likelihood, since `\(y_i\sim N(X_i^T\beta,\sigma^2)\)`, we have:

`\(\prod_{i=1}^np(y_i|\beta,\sigma^2,X_i) = \prod^n_{i=1}(2\pi\sigma^2)^{-1/2}exp\{-\dfrac{1}{2\sigma^2}(y_i-X_i^T\beta)^2\}\)`

For the priors, a uniform prior is often used for the regression parameters. That is: `\(p(\beta) \propto 1\)`.

`\(1/\sigma^2\)` is used for `\(\sigma\)`. That is: `\(p(\sigma)\propto 1/\sigma^2\)`.

The joint posterior is then:

`\(p(\beta,\sigma^2|X,Y) \propto (\sigma^2)^{-(n/2+1)}exp\{-\dfrac{1}{2\sigma^2}(Y-X\beta)^T(Y-X\beta)\}\)`

For Gibbs Sampling, we need to break down the full posterior into a product of full conditional probabilities. Deriving the full conditionals is a simple but tedious matter of selecting only the terms that contain the parameters of interest. 

It can be shown that the full conditional of the regression coefficients `\(\beta\)` is:

`\(\beta \sim N((X^TX)^{-1}(X^TY),\sigma^2_e(X^TX)^{-1})\)`

The full conditional of `\(\sigma\)` is Inverse Gamma with parameters `\(\alpha=n/2\)` and `\(\beta=e^Te/2\)`: 

`\(p(\sigma^2|\beta,X,Y) \propto (\sigma^2)^{-(n/2+1)}exp\{-\dfrac{e^Te}{2\sigma^2}\}\)`

where `\(e^Te\)` is the sum of the square error terms under the given values of `\(\beta\)`.

Define `\(V_\beta = (X^TX)^-\)` for easier coding.


```r
library(MASS)
boston_data <- data(Boston)
boston_lm <- lm(medv~.,data=Boston)

x <- as.matrix(Boston[,1:13])
x <- cbind(rep(1,506),x)
y <- as.matrix(Boston[,14])
N = 506
P = 14

#Function for sampling from p(beta|sigma2,X,y)
sample.beta <- function(sigma2, V.beta, beta.hat)
					{
                    		return(mvrnorm(1,beta.hat,sigma2*V.beta))
                    }
                    
#Function for sampling from p(sigma2|X,y)
sample.sigma2 <- function(s2,N,P)
					{
                       return(1/rgamma(1, (N-P)/2, ((N-P)/2)*s2))
                    }

V.beta <- solve(t(x)%*%x)
beta.hat <- V.beta%*%t(x)%*%y
s2 <- t(y-x%*%beta.hat)%*%(y-x%*%beta.hat)/(N-P)

#Set number of samples
K <- 30000

#Initialize variables for the posterior samples
beta.post.samples = matrix(NA, K, P)
sigma2.post.samples = rep(NA, K)

for(i in 1:K)
	{
      	sigma2.curr <- sample.sigma2(s2, N, P)
      	sigma2.post.samples[i] <- sigma2.curr
      	beta.curr <- sample.beta(sigma2.curr, V.beta, beta.hat)
      	beta.post.samples[i,] <- beta.curr
     }
```
      
The posterior samples are visualized below.


```r
covariate_names <- names(Boston[,1:13])
boxplot(as.data.frame(beta.post.samples[,-1]),names=covariate_names)
abline(h=0)
```

<img src="/blogs/Bayesian-Data-Analysis_files/figure-html/unnamed-chunk-2-1.png" width="672" />

The posterior samples of the parameters can be summarized by their mean. The posterior means are actually pretty close to the estimated coefficients produced by traditional Frequentist approach.


```r
#Compare posterior means with Frequentist Results
posterior_mean <- apply(beta.post.samples, 2, mean)
boston_result <- data.frame(freq_result = coef(boston_lm),
                            bayes_result = apply(beta.post.samples, 2, mean))
boston_result
```

```
##               freq_result  bayes_result
## (Intercept)  3.645949e+01  3.645315e+01
## crim        -1.080114e-01 -1.081400e-01
## zn           4.642046e-02  4.657443e-02
## indus        2.055863e-02  2.037269e-02
## chas         2.686734e+00  2.685971e+00
## nox         -1.776661e+01 -1.771932e+01
## rm           3.809865e+00  3.808257e+00
## age          6.922246e-04  5.579795e-04
## dis         -1.475567e+00 -1.476974e+00
## rad          3.060495e-01  3.061821e-01
## tax         -1.233459e-02 -1.234186e-02
## ptratio     -9.527472e-01 -9.523644e-01
## black        9.311683e-03  9.311077e-03
## lstat       -5.247584e-01 -5.247805e-01
```


### Relationshio between MH Sampling and Gibbs Sampling

In practice, Gibbs Sampling is often used in combination with MH Sampling for multivariate analyses. 

When the full conditional of a parameter is known, the distribution of the parameter conditional on all other parameters is used for proposal generation.

When the full conditional of a parameter is unknown, a known distribution is used for proposal generation, which leads to MH Sampling.

Consider the example where the treatment success probabilities of five medications are estimated and compared. 


```r
Y <- c(17,12,22,35,14)
N <- c(29,18,28,45,30)
J <- 5
```

`\(N_j\)` is the number of individuals who took medication j and `\(Y_i\)` is the number of individuals who experience relief from taking medication j.

The parameters of interest are probabilities of treatment success `\(\theta_j, j=1,2,3,4,5\)`. 

The data model is:

`\(y_j|\theta_j \sim Bin(N_j,\theta_j),j=1,2,3,4,5\)`

To approach the problem with a hierarchical model, we assume the logits of the parameters come from an underlying normal distribution:

`\(logit(\theta_j) = log(\dfrac{\theta_j}{1-\theta_j}) \sim N(\mu,\tau^2)\)`

where `\(\mu\)` and `\(\tau^2\)` are hyper-parameters with hyper-priors of `\(\mu\sim Unif(a,b) \propto 1\)`, and `\(\tau^2 \sim IG(\alpha,\beta)\)`

The joint posterior distribution is then: 

`\(p(\theta,\mu,\tau^2|Y,N) \propto \prod^J_{j=1}[p(y_j|\theta_j)p(\theta_j|\mu,\tau)]p(\mu)p(\tau^2)\)`

To use Gibbs Sampler, full conditionals for `\(\theta_j\)`, `\(\mu\)`, and `\(\tau^2\)` need to be derived. The derivation requires selecting terms that contain parameter of interest and discarding other terms as constants. Rearrangement and simplification are also needed to identify the posterior distribution. 

The full conditional distribution of `\(\theta_j\)`, `\(p(\theta_j|\theta_{-j},\mu,\tau^2)\)` is:

`\(p(\theta_j|\theta_{-j},\mu,\tau^2) \propto p(y_j|\theta_j)p(\theta_j|\mu,\tau)\)`

`\(\propto \theta_j^{y_j}(1-\theta_j)^{n_j-y_j} * exp\{-\frac{1}{2\tau^2}(logit(\theta_j)-\mu)^2\}[(\theta_j)(1-\theta_j)]^{-1}\)`

Note: The term `\([(\theta_j)(1-\theta_j)]^{-1}\)` is due to random variable transformation, since we want to know the pdf of `\(\theta_j\)` and we know the pdf of `\(logit(\theta_j)=log(\dfrac{\theta_j}{1-\theta_j})\)`.



```r
#Functions for samples from the full conditional distributions

logit <- function(p) return(log(p/(1-p)))

sample.logit.theta.j <- function(y.j, N.j, mu,tau.sq, logit.theta.j, proposal.sd)
					{
                  		logit.theta.old <- logit.theta.j
                  		logit.theta.new <- rnorm(1, logit.theta.old, proposal.sd)
                  	
                      theta.new <- exp(logit.theta.new) / (1+exp(logit.theta.new))
                      theta.old <- exp(logit.theta.old) / (1+exp(logit.theta.old))
                      alpha <- (y.j*log(theta.new) + (N.j-y.j)*log(1-theta.new) -
                      					.5*(logit.theta.new-mu)^2/tau.sq - log(theta.new) -log(1-theta.new)) -
                      					(y.j*log(theta.old) + (N.j-y.j)*log(1-theta.old) -
                      					.5*(logit.theta.old-mu)^2/tau.sq - log(theta.old) - log(1-theta.old))
                      if(log(runif(1,0,1)) < alpha){
                              	return(c(logit.theta.new,1))
                              	}else{
                                    return(c(logit.theta.old,0))
                                }
						}

sample.mu <- function(logit.theta, tau.sq,J)  return(rnorm(1,mean(logit.theta),sqrt(tau.sq/J)))

sample.tau.sq <- function(logit.theta, mu,J, alpha, beta)
					{
                 		alpha.post <- (J/2)+alpha
                 		beta.post <- sum(.5*sum((logit.theta-mu)^2)+beta)
                    	return(1/rgamma(1,shape=alpha.post,scale=1/beta.post))
                    }
```

Set up MCMC


```r
K <- 10000
accept <- matrix(NA, K, J)
proposal.sd <- 1

#Set hyperprior parameters
alpha <- .01
beta <- .01
                                     
#Initialize parameters
logit.theta.post <- rep(logit(.5), J)
mu.post <- 0
tau.sq.post <- 1

#Storage of samples
theta.samples <- matrix(NA, K, J)
mu.samples <- rep(NA, K)
tau.sq.samples <- rep(NA, K)

# Run MCMC
set.seed(1234)

for(k in 1:K)
	{
      #print(k)
      for(j in 1:J)
      	{
       		temp <- sample.logit.theta.j(Y[j],N[j],mu.post,tau.sq.post,logit.theta.post[j],proposal.sd)
            logit.theta.post[j] <- temp[1]
            accept[k,j] <- temp[2]
     	}
      theta.samples[k,] <- exp(logit.theta.post) / (1+exp(logit.theta.post))      
      mu.post <- sample.mu(logit.theta.post,tau.sq.post,J)
      mu.samples[k] <- mu.post
      tau.sq.post <- sample.tau.sq(logit.theta.post,mu.post,J,alpha,beta)
      tau.sq.samples[k] <- tau.sq.post
}

samples.keep <- 1001:K
```

The posterior samples are visualized below.


```r
par(mfrow=c(1,1),mar=c(4,4,4,2), ask=F)
boxplot(as.data.frame(cbind(theta.samples[samples.keep,], 
	exp(mu.samples[samples.keep])/(1+exp(mu.samples[samples.keep])))), 
	names=c(as.character(1:J),"inv.logit.mu"), 
	main="Posterior of the thetas and the inv.logit.mu", 
	ylim=c(0,1))
points(1:J, Y/N, pch=19, col='green')
```

<img src="/blogs/Bayesian-Data-Analysis_files/figure-html/unnamed-chunk-7-1.png" width="672" />

### Hamiltonian MC
 
When MH generates proposals, it is equally likely to proposal values from the regions of low posterior probability as the regions of high posterior probability. Such feature leads to cases where the proposed values get rejected quite often, causing a significant efficiency loss.

Hamiltonian Monte Carlo (HMC) is a sampling method design to fix the efficiency problem. It is a variation of the MH algorithm in that it follows the same steps of the MH algorithm except that it uses a guided proposal generation scheme so that the proposals tend to be in the region of higher posterior probability.

To get proposed values from the regions of high posterior probability, We want to make the proposal in the direction that the posterior probability increases. Equivalently, we want proposals in the direction that the log posterior probability density function decreases. 

HMC uses the gradient of the log posterior to direct the Markov chain towards regions of higher posterior density, where most samples are taken. As a result, a well-tuned HMC chain will accept proposals at a much higher rate than the traditional MH algorithm.

For an intuitive understanding of how HMC generates proposals, think of rolling a marble o valley curve with a random force and at random direction, and capturing it at a certain amount of time. By laws of physics, the marble tends to visit the bottom of the valley more often. In the case of HMC, the valley curve is the log posterior distribution. Since the force to roll the marble is random, the position of the marble after a certain amount of time can be seen as a random sample of the log posterior distribution. 

The question is: How do we know the path and location of the marble?

Again we resort to physics. Assuming the curve is frictionless, as the marble moves, its potential energy and kinetic energy turns into each other. In physics, the Hamiltonian equation is a system of differential equations that can be used to solve the path of the marble based on the relationship between potential energy and kinetic energy.

The energy level of an object is described as:

`\(H(\theta,p) = U(\theta)+K(p)\)`, 

where `\(U(\theta)\)` is potential energy. 

With HMC, `\(U(\theta)\)` is the negative log posterior: `\(U(\theta) = -logf(\theta)\)`; 

`\(K(p)\)` is kinetic energy; 

`\(p\)` is a momentum vector: `\(p\sim N_k(0,M)\)` where M is a user-specified covariance matrix and often an identity matrix.

By physics formulation, we have: `\(H(\theta,p) = -logf(\theta)+\frac{1}{2}p^TM^{-1}p\)`

The trajectories of the marble are governed by the Hamiltonian equations:

`\(\dfrac{dp}{dt} = -\frac{\partial H(\theta,p)}{\partial\theta} = -\frac{\partial U(\theta)}{\partial \theta}=\nabla_{\theta}logf(\theta)\)`

`\(\dfrac{d\theta}{dt} = -\frac{\partial H(\theta,p)}{\partial\theta} = -\frac{\partial K(p)}{\partial \theta}=M^{-1}p\)`

The Hamiltonian equations involve continuous time, we need to discretise it by breaking it into small steps.

In HMC, the Hamiltonian equations are solved iteratively with an algorithm called leadfrog. The leapfrog algorithm breaks the path of the marble into multiple small steps of length `\(\epsilon\)`. 

The leapfrog algorithm is expressed as:

`\(p(t+\epsilon/2) = p(t)+(\epsilon/2)\nabla_{\theta}log(f(\theta(t))\)`

`\(\theta(t+\epsilon) = \theta(t)+\epsilon M^{-1}p(t+\epsilon/2)\)`

`\(p(t+\epsilon) = p(t+\epsilon/2)+(\epsilon/2) \nabla_{\theta}logf(\theta(t+\epsilon))\)`

where p is momentum and `\(\theta\)` is potential.

In practice, the trajectory of the marble is solved by running leapfrog multiple times with each step of length `\(\epsilon\)`.

The Hamiltonian MC algorithm is:

(1). Get 

(2).


#### Acceptance Function

After a proposal is generated, the following acceptance function is used to evaluate whether the proposal is accepted: 

`\(a = \underbrace {\dfrac{p(x|\theta^*)p(\theta^*)N(P^*|0,1)}{p(x|\theta)p(\theta)N(P|0,1)}}_{\text{Matropolis}}*\underbrace{\dfrac{p(\theta,P|\theta^*,P^*)}{p(\theta^*,P^*|\theta,P)}}_{\text{Hastings}}\)`

It is the same MH acceptance function except that it now contains an additional component `\(P\)` for the random momentum. `\(\theta^*\)` and `\(P*\)` are the proposed parameter values and momentum respectively.

It can be shown that by flipping the sign of `\(P^*\)` for momentum proposals, we can ignore the Hastings part of the acceptance ratio:

`\(a = \dfrac{p(x|\theta^*)p(\theta^*)N(-P^*|0,1)}{p(x|\theta)p(\theta)N(P|0,1)}\)`

Since the momentum is a random draw from (multi-)normal distribution, `\(N(-P^*|0,1) = N(P^*|0,1)\)`.

The acceptance probability is: 

`\(a = min\{1, \dfrac{p(x|\theta^*)p(\theta^*)N(P^*|0,1)}{p(x|\theta)p(\theta)N(P|0,1)}\}\)`

For more details on the acceptance probability of HMC, check out the video [The intuition behind the Hamiltonian Monte Carlo algorithm](https://www.youtube.com/watch?v=a-wydhEuAm0)

#### Code Example - Linear Regression

This section references codes provided in the following two posts: [Learning Hamiltonian Monte Carlo in R](https://arxiv.org/pdf/2006.16194); [Hamiltonian Monte Carlo in R](https://jonnylaw.rocks/posts/2019-07-31-hmc/).

In this code example we consider the case of linear regression: 

`\(y_i = X_i^T\beta+\epsilon_i\)`

where `\(X_i\)` is the covariate vector for subject i: `\(X_i^T = (x_{i0},...,x_{iq})\)`. Here `\(x_{i0}\)` is frequently set to one as an intercept term.

The log likelihood of the linear regression model can be expressed as:

`\(logf(y|\beta,\sigma^2)\propto -nlog\sigma - \dfrac{1}{2\sigma^2}(y-X\beta)^T(y-X\beta)\)` 

where X is the full design matrix `\(X = (X_1^T,...,X_n^T)^T\)`; `\(\beta = (\beta_0,...,\beta_q)^T\)` is the regression coefficient vector including the intercept term; `\(\sigma^2\)` is the variance of the error term.   

To get the log posterior function, we need the priors for `\(\beta\)` and `\(\sigma\)`

For `\(\beta\)` we use an uninformative (uniform) prior: `\(p(\beta) \propto 1\)`

For `\(\sigma\)` we use an inverse gamma prior with hyperparameters a and b: `\(\dfrac{b^a}{\Gamma(a)}(\sigma^2)^{-a-1}exp(-\dfrac{b}{\sigma^2})\)`

Taking the log transformation of `\(\sigma^2\)` to expand its support from `\((0,\inf)\)` to `\(\mathbb{R}\)`, we have:

`\(\gamma  = log\sigma^2=2log \sigma\)`

`\(f(\gamma|a,b)=\dfrac{b^a}{\Gamma(a)}exp(-a\gamma-\dfrac{b}{e^{\gamma}})\)`

`\(logf(\gamma|a,b) \propto -\alpha\gamma-be^{-\gamma}\)`

Putting everything together, the posterior distribution is:

`\(f(\beta,\sigma^2|y,X,a,b) \propto f(y|\beta,\sigma^2)f(\beta)f(\sigma^2|a,b)\)`

`\(f(\beta,\sigma^2|y,X,a,b) \propto -nlog\sigma - \dfrac{1}{2\sigma^2}(y-X\beta)^T(y-X\beta) -\alpha\gamma-be^{-\gamma}\)`

`\(f(\beta,\sigma^2|y,X,a,b) \propto -(\dfrac{n}{2}+a)\gamma - \dfrac{e^{-\gamma}}{2}(y-X\beta)^T(y-X\beta)-be^{-\gamma}\)`

To initiate leapfrog steps, the log posterior function and its gradient are needed: 

`\(\nabla_\beta logf(\beta,\gamma|y,X,a,b) \propto e^{-\gamma}X^T(y-X\beta)\)`

`\(\nabla_\gamma logf(\beta,\gamma|y,X,a,b) \propto -(\dfrac{n}{2}+a)+\dfrac{e^{-\gamma}}{2}(y-X\beta)^T(y-X\beta)+be^{-\gamma}\)`

Now the gradient function can be defined:


```r
warp <- warpbreaks
y <- warp$breaks
X <- model.matrix(breaks~wool*tension,data=warp)

# find log posterior
log_posterior <- function(theta, y, X, a=1e-4, b=1e-4, sig2beta=1e3) {
k <- length(theta)
beta_param <- as.numeric(theta[1:(k-1)])
gamma_param <- theta[k]
n <- nrow(X)
result <- -(n/2+a)*gamma_param - exp(-gamma_param)/2 *
t(y - X%*%beta_param) %*%
(y - X%*%beta_param) - b*exp(-gamma_param) -
1/2* t(beta_param) %*% beta_param / sig2beta
return(result)
}

# find gradient of log posterior function
gradient <- function(theta, y=y, X=X, a=1e-4, b=1e-4) {
  k <- length(theta)
  beta_param <- as.numeric(theta[1:(k-1)])
  gamma_param <- theta[k]
  n <- nrow(X)
  grad_beta <- exp(-gamma_param) * t(X) %*% (y - X%*%beta_param)
  grad_gamma <- -(n/2 + a) + exp(-gamma_param)/2 * t(y - X%*%beta_param) %*% (y - X%*%beta_param) + b*exp(-gamma_param) 
  c(as.numeric(grad_beta), as.numeric(grad_gamma))
}
```

Then define the leapfrog function:


```r
# one leapfrog step
leapfrog_step <- function(y,X,step_size, position, momentum, d) {
  momentum1 <- momentum + gradient(position,y=y,X=X) * 0.5 * step_size
  position1 <- position + step_size * momentum1
  momentum2 <- momentum1 + gradient(position1,y=y,X=X) * 0.5 * step_size

  matrix(c(position1, momentum2), ncol = d*2)
}

#multiple leapfrog steps
leapfrogs <- function(y,X,step_size,l,position, momentum,d) {
  for (i in 1:l) {
    pos_mom <- leapfrog_step(y=y,X=X,step_size, position, momentum,d)
    position <- pos_mom[seq_len(d)]
    momentum <- pos_mom[-seq_len(d)]
  }
  pos_mom
}
```

The proposals of `\(\beta\)` and `\(\gamma\)` generated by the leapfrog algorithm are used for the acceptance function. 


```r
hmc_step <- function(log_posterior, y, X, step_size, l, position) {
  d <- length(position)
  momentum <- rnorm(d)
  pos_mom <- leapfrogs(y,X, step_size, l, position, momentum, d)
  propPosition <- pos_mom[seq_len(d)]
  propMomentum <- pos_mom[-seq_len(d)]
  a <- log_posterior(propPosition,y,X) + sum(dnorm(propMomentum, log = T)) - log_posterior(position,y,X) - sum(dnorm(momentum, log = T))
  if (log(runif(1)) < a) {
    propPosition
  } else {
    position
  }
}


hmc <- function(log_posterior, y, X, step_size, l, initP, m) {
  out <- matrix(NA_real_, nrow = m, ncol = length(initP))
  out[1, ] <- initP
  for (i in 2:m) {
    out[i, ] <- hmc_step(log_posterior, y, X, step_size, l, out[i-1,])
  }
  out
}

hmc_result <- hmc(log_posterior,y,X,c(rep(2e-1, 6), 2e-2),20,initP=c(rep(0,6),1),10000)

colMeans(hmc_result)
```

```
## [1]  43.668417 -12.977073 -17.712836 -17.286922  17.555635   7.174171   4.876992
```

### hmclearn Package

The [hmclearn package](https://cran.r-project.org/web/packages/hmclearn/hmclearn.pdf) offers functions that simplify the above steps. It still requires log posterior and gradient function which are given for different models in the resource linked above. 


```r
library(hmclearn)
```

```
## 
## Attaching package: 'hmclearn'
```

```
## The following object is masked _by_ '.GlobalEnv':
## 
##     hmc
```

```r
N <- 20000
eps_vals <- c(rep(2e-1, 6), 2e-2)

fm1_hmc <- hmclearn::hmc(N, theta.init = c(rep(0, 6), 1), epsilon = eps_vals, L = 20, 
                         logPOSTERIOR = log_posterior,
                         glogPOSTERIOR = gradient,
                         varnames = c(colnames(X), "log_sigma_sq"),
                         param = list(y = y, X = X), chains = 1,parallel = FALSE)

summary(fm1_hmc, burnin=200)
```

```
## Summary of MCMC simulation
```

```
##                      2.5%         5%        25%        50%        75%       95%
## (Intercept)     35.668542  36.846558  40.480288  42.880165  45.217340 48.575290
## woolB          -23.549844 -21.888332 -17.302565 -14.004022 -10.664247 -5.711810
## tensionM       -28.219038 -26.555604 -21.804445 -18.423257 -14.981901 -9.719287
## tensionH       -27.787228 -26.118759 -21.220476 -17.923641 -14.503588 -9.566358
## woolB:tensionM   3.777893   6.010062  13.212303  18.061518  22.764766 29.475085
## woolB:tensionH  -6.536300  -4.099362   2.895146   7.686632  12.378316 18.907345
## log_sigma_sq     4.413300   4.474713   4.663775   4.798693   4.945274  5.160732
##                    97.5%
## (Intercept)    49.716921
## woolB          -4.002202
## tensionM       -8.038178
## tensionH       -7.665580
## woolB:tensionM 31.700126
## woolB:tensionH 21.152953
## log_sigma_sq    5.234516
```

```r
freq_model <- lm(breaks~wool*tension,data=warpbreaks)
freq_param <- c(coef(freq_model),2*log(sigma(freq_model)))
diagplots(fm1_hmc, burnin=200, comparison.theta=freq_param)
```

```
## $histogram
```

```
## Warning: The dot-dot notation (`..density..`) was deprecated in ggplot2 3.4.0.
## ℹ Please use `after_stat(density)` instead.
## ℹ The deprecated feature was likely used in the hmclearn package.
##   Please report the issue to the authors.
## This warning is displayed once every 8 hours.
## Call `lifecycle::last_lifecycle_warnings()` to see where this warning was
## generated.
```

<img src="/blogs/Bayesian-Data-Analysis_files/figure-html/unnamed-chunk-8-1.png" width="672" />


## Diagnostics

### Representativeness

The values in the Markov chain must be representative of the posterior distribution. In other words, we need to check whether the chain has converged. There are mainly two ways to examine whether a chain is representative enough: 

(1) Trace Plot and Density Plot

This is to examine the chain trajectory with the sampled values plotted against the steps. A trace plot of convergence should show the sampled values change drastically in the preliminary steps but then hover around narrow range of values on the long run. 

To enhance the visibility of representativeness it is preferrable to have multiple chains with different initial values. The chains should overlap with each other after certain number of steps. How do we set up different initial values? One approach is to start all parameters at 0 for one chain, use MLE estimates for a second chain, and use wildly inappropriate values for the third chain.

Another useful visual is density plot of the sample values. A density plot is a smoothed histogram that allows us to check the distribution of the sampled values. Representative chains should have density plots where the Highest Density Interval overlaps with each other. 


```r
fm2_hmc <- hmclearn::hmc(N, theta.init = freq_param, epsilon = eps_vals, L = 20, 
                         logPOSTERIOR = log_posterior,
                         glogPOSTERIOR = gradient,
                         varnames = c(colnames(X), "log_sigma_sq"),
                         param = list(y = y, X = X), chains = 1,parallel = FALSE)


fm3_hmc <- hmclearn::hmc(N, theta.init = c(rep(-100, 6), 100), epsilon = eps_vals, L = 20, 
                         logPOSTERIOR = log_posterior,
                         glogPOSTERIOR = gradient,
                         varnames = c(colnames(X), "log_sigma_sq"),
                         param = list(y = y, X = X), chains = 1,parallel = FALSE)


testplot <- diagplots(fm1_hmc, burnin=200, plotfun = 1)
diagplots(fm2_hmc, burnin=200, plotfun = 1)
```

```
## $trace
```

<img src="/blogs/Bayesian-Data-Analysis_files/figure-html/unnamed-chunk-9-1.png" width="672" />

```r
diagplots(fm3_hmc, burnin=200, plotfun = 1)
```

```
## $trace
```

<img src="/blogs/Bayesian-Data-Analysis_files/figure-html/unnamed-chunk-9-2.png" width="672" />


(2) Numerical Description of Convergence

One popular numerical measure of representativeness is Gelman and Rubin’s statistic. It checks how much variance there is between chains vs how much variance there is within chains.values close to 1 indicate that the chain has sufficiently converged. A general guideline suggests that values less than 1.05 are good, between 1.05 and 1.10 are ok, and above 1.10 have not converged well.


```r
psrf(fm1_hmc)
```

```
## [1] NA NA NA NA NA NA NA
```

### Accuracy

After making sure the chains are reasonably representative of the posterior distribution, we need to make sure we have a large enough sample for accurate numerical estimates of the posterior distribution.

(1) ACF

The first thing to check is that the samples from the chain can be treated as independent draws on the long run. Auto-correlation function (ACF) is a useful measure for checking how samples are correlated with each other. In practice, we want the ACF values to be small enough to ignore after a certain lag.


```r
mcmc_acf(fm1_hmc,lags=500,burnin = 1000)
```

<img src="/blogs/Bayesian-Data-Analysis_files/figure-html/unnamed-chunk-11-1.png" width="672" />

(2) Effective Sample Size

Given the samples from a Markov chain is auto-correlated for a while till the correlation can be ignored and the samples can be treated as independent sequence, one would be interested in the effective sample size that a chain provides. If the total number of steps in a chain is N, then the effective sample size is defined as:

`\(ESS = N/(1+2\sum^\infty_{k=1}ACF(k))\)`

where `\(ACF(k)\)` is the autoccorelation of the chain at lag k. In practice, the sum in the definition of ESS may be stopped when ACF(k)<0.0.5


```r
neff(fm1_hmc)
```

```
## [1]  40  52  57  61 703 673  40
```

As a general guidance, it is recommended that ESS should be at least 10000. 







