rm(list = ls())
library(datasets)
#  R package call in maxLik
require(miscTools)
#require(maxLik)
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
source("maximMessage.R")
#
source("/home/aborba/github/mb2021/Code_r/loglike.r")
der_loglike_teste <- function(x){
  der <- rep(0,2) 
  der[1]  <- log(x[1] / x[2]) + x[2] - digamma(x[1]) + lc1 - lc2 / x[2]
  der[2]  <- - x[1] / x[2] + (x[1] / x[2]^2) * lc2
  return(der)
}
#
mat <- scan('/home/aborba/github/mb2021/Data/flevoland_1.txt')
#
r <- 120
nr <- 100
mat <- matrix(mat, ncol = r, byrow = TRUE)
N <- r
z <- rep(0, N)
k <- 5
z <- mat[k, 1: N]
zaux1 <- rep(0, N)
conta = 0
for (i in 1 : N){
  if (z[i] > 0){
    conta <- conta + 1
    zaux1[conta] = z[i]
  }
}
indx  <- which(zaux1 != 0)
N <- floor(max(indx))
z     <-  zaux1[1:N]
j <- 10
r1 <- 1
r2 <- sum(z[1: j]) / j
lc1 <- sum(log(z[1: j])) / j
lc2 <- r2
res1 <- maxBFGS(loglike, start=c(r1, r2))
pt1 <- res1$estimate
#
method = "BFGS"
reltol = 1e-08 
iterlim = 200
parscale= c(1, 1)
alpha = NULL
beta  = NULL
gamma = NULL
temp  = NULL
tmax  = NULL
fixed <-NULL
start <- c(r1, r2)
grad = NULL
hess = NULL
gradOptim <- logLikGrad
#
control <- list(trace=max(0, 0),
                  REPORT=1,
                  fnscale=-1,
                  reltol=reltol,
                  maxit=iterlim,
                  parscale=parscale,
                  alpha=alpha, beta=beta, gamma=gamma,
                  temp=temp, tmax=tmax )
#
res2 <- optim( par = start, fn = logLikFunc, control = control,
                 method = method, gr = gradOptim, fnOrig = loglike,
                 gradOrig = grad)
pt2 <- res2$par