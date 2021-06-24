rm(list = ls())
#require(ggplot2)
#require(latex2exp)
#require(GenSA)
#require(maxLik)
require(stats)
require(miscTools)
#
#source("func_obj_l_L_mu.r")
source("/home/aborba/github/mb2021/Code_r/loglike.r")
#source("loglikd.r")
#
n <- 100
zz <- rnorm(n, mean = 1, sd = 2)
loglik <- function(theta) {
  mu <- theta[1]
  sigma <- theta[2]
  logLikFunValue <- -n*log(sqrt(2*pi)) - n*log(sigma) - 0.5*sum((zz - mu)^2/sigma^2)
  return(logLikFunValue)
}
# 
#setwd("..")
#setwd("Data")
# channels hh(1), hv(2), and vv(3)
mat <- scan('/home/aborba/github/mb2021/Data/flevoland_1.txt')
#setwd("..")
#setwd("Code_r")
########## setup to Flevoland
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
res <- maxBFGS(loglike, start=c(r1, r2))
#res <- maxBFGS(loglik, start=c(0, 1))
pt1 <- res$estimate
print(j)
print(r2)
print(loglike(c(r1, r2)))
method = "BFGS"
reltol = 1e-08 
iterlim = 200
parscale= c(1, 1)
alpha = NULL
beta  = NULL
gamma = NULL
temp  = NULL
tmax  = NULL
#fixed <- NULL
start <- c(r1, r2)
#start <- c(mu=0, sigma=1)
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
print("Inicio optim")
res1 <- optim(par = start, fn=loglike, control = control, method = method )
pt2 <- res1$par
#res1 <-optim(c(r1, r2),fn=loglike, method = "BFGS")
#pt2 <- res1$par

#print(res$estimate)