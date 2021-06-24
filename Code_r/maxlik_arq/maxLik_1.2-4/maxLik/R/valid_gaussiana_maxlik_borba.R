rm(list = ls())
library(datasets)
#  R package call in maxLik
require(miscTools)
require(utils)
# Call source of the maxLik package 
source("maxBFGS.R")
source("maxOptim.R")
source("checkFuncArgs.R")
source("prepareFixed.R")
source("callWithoutSumt.R")
source("callWithoutArgs.R")
source("sumt.R")
source("logLikFunc.R")
source("logLikGrad.R")
source("numericGradient.R")
source("sumGradients.R")
source("addFixedPar.R")
source("observationGradient.R")
source("logLikHess.R")
source("numericHessian.R")
#
n <- 100
z <- rnorm(n, mean = 1, sd = 2)
loglik <- function(theta) {
  mu <- theta[1]
  sigma <- theta[2]
  logLikFunValue <- -n*log(sqrt(2*pi)) - n*log(sigma) - 0.5*sum((z - mu)^2/sigma^2)
  return(logLikFunValue)
}
res1 <- maxBFGS(loglik, start=c(mu=0, sigma=1), print.level= 0)
pt1 <- res1$estimate
#
method = "BFGS"
reltol = 1e-08 
iterlim = 200
parscale= c(1, 1)
alpha = "NULL"
beta  = "NULL"
gamma = "NULL"
temp  = "NULL"
tmax  = "NULL"
#fixed <-"NULL"
start <- c(mu=0, sigma=1)
#fixed <- prepareFixed( start = start, activePar = NULL, fixed = fixed )
grad = NULL
hess = NULL
# define gradiente e hessiana
#gradOptim <- logLikGrad
control <- list(trace=max(0, 0),
                  REPORT=1,
                  fnscale=-1,
                  reltol=reltol,
                  maxit=iterlim,
                  parscale=parscale,
                  alpha=alpha, beta=beta, gamma=gamma,
                  temp=temp, tmax=tmax )
#
res2 <- optim(start, fn=loglik, control = control, method = "BFGS" )
pt2 <- res2$par