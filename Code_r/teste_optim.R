rm(list = ls())
library(maxLik)
source("loglike_teste.r")
source("der_loglike_teste.r")
## ML estimation of exponential duration model:
#t <- rexp(10, 2)
#loglik <- function(x) -((x[1] - 5)^2 + (x[2] - 6)^2)

## Estimate with numeric gradient and hessian
#a <- maxLik(loglik, start=c(1, 1))
lc1 <- -7.101201 # sum(log(z)) / j
lc2 <- 0.0010654 # sum(z) / j
#
c1 <- 4.0
c2 <- lc2
#
a <- maxBFGS(loglike_teste, start=c(c1, c2), control = list(iterlim = 1))
res1 <- optim(c(c1, c2), loglike_teste, der_loglike_teste, method = "BFGS")
#
grad <- gradient(a)
hess <- hessian(a)
print(a$estimate)
der_grad <- der_loglike_teste(c(c1, c2))
hess_optim <- optimHess(a$estimate, loglike_teste, der_loglike_teste)
# Extract the gradients evaluated at each observation
#library( sandwich )
#estfun( a )

## Estimate with analytic gradient.
## Note: it returns a vector
#gradlik <- function(theta) 1/theta - t
#b <- maxLik(loglik, gradlik, start=1)
#gradient(a)
#estfun( b )
