### Various tests for constrained optimization
###
options(digits=4)

logLikMix <- function(param) {
   rho <- param[1]
   if(rho < 0 || rho > 1)
       return(NA)
   mu1 <- param[2]
   mu2 <- param[3]
   ll <- log(rho*dnorm(x - mu1) + (1 - rho)*dnorm(x - mu2))
#   ll <- sum(ll)
   ll
}

gradLikMix <- function(param) {
   rho <- param[1]
   if(rho < 0 || rho > 1)
       return(NA)
   mu1 <- param[2]
   mu2 <- param[3]
   f1 <- dnorm(x - mu1)
   f2 <- dnorm(x - mu2)
   L <- rho*f1 + (1 - rho)*f2
   g <- matrix(0, length(x), 3)
   g[,1] <- (f1 - f2)/L
   g[,2] <- rho*(x - mu1)*f1/L
   g[,3] <- (1 - rho)*(x - mu2)*f2/L
#   colSums(g)
   g
}

hessLikMix <- function(param) {
   rho <- param[1]
   if(rho < 0 || rho > 1)
       return(NA)
   mu1 <- param[2]
   mu2 <- param[3]
   f1 <- dnorm(x - mu1)
   f2 <- dnorm(x - mu2)
   L <- rho*f1 + (1 - rho)*f2
   dldrho <- (f1 - f2)/L
   dldmu1 <- rho*(x - mu1)*f1/L
   dldmu2 <- (1 - rho)*(x - mu2)*f2/L
   h <- matrix(0, 3, 3)
   h[1,1] <- -sum(dldrho*(f1 - f2)/L)
   h[2,1] <- h[1,2] <- sum((x - mu1)*f1/L - dldmu1*dldrho)
   h[3,1] <- h[1,3] <- sum(-(x - mu2)*f2/L - dldmu2*dldrho)
   h[2,2] <- sum(rho*(-f1 + (x - mu1)^2*f1)/L - dldmu1^2)
   h[2,3] <- h[3,2] <- -sum(dldmu1*dldmu2)
   h[3,3] <- sum((1 - rho)*(-f2 + (x - mu2)^2*f2)/L - dldmu2^2)
   h
}
### --------------------------
library(maxLik)
## mixed normal
set.seed(1)
x <- c(rnorm(1000, mean=-1), rnorm(1000, mean=1))

cat("Test for inequality constraints\n")
## Inequality constraints: x + y + z < 0.5
A <- matrix(c(-1, 0, 0,
              0, -1, 0,
              0, 0, 1), 3, 3, byrow=TRUE)
B <- rep(0.5, 3)
start <- c(0.4, 0, 0.9)
## analytic gradient
a <- maxLik(logLikMix, grad=gradLikMix, hess=hessLikMix,
            start=start,
            constraints=list(ineqA=A, ineqB=B),
            print.level=1)
print( summary( a ), digits = 2 )
## No analytic gradient
a <- maxLik(logLikMix, 
            start=start,
            constraints=list(ineqA=A, ineqB=B),
            print.level=1)
print( summary( a ), digits = 2 )
## No analytic gradient, BFGS
a <- maxLik(logLikMix, 
            start=start,
            method="bfgs",
            constraints=list(ineqA=A, ineqB=B),
            print.level=1)
print( summary( a ), digits = 2 )
## ----
cat("Test for equality constraints\n")
A <- matrix(c(0, 1, 2), 1, 3)
B <- 0
## default, analytic gradient
a <- maxLik(logLikMix, grad=gradLikMix, hess=hessLikMix,
            start=start,
            constraints=list(eqA=A, eqB=B),
            print.level=1)
print( summary( a ), digits = 2 )
## BFGS, numeric gradient
a <- maxLik(logLikMix, 
            start=start, method="bfgs",
            constraints=list(eqA=A, eqB=B),
            print.level=2, SUMTRho0=1)
print( summary( a ), digits = 2 )
## BHHH, analytic gradient (numeric does not converge?)
try( maxLik(logLikMix, gradLikMix,
            start=start, method="bhhh",
            constraints=list(eqA=A, eqB=B),
            print.level=2, SUMTRho0=1) )

### ------------------ Now test extra parameters for the function ----
logLikMix2 <- function(param, rho) {
   mu1 <- param[1]
   mu2 <- param[2]
   ll <- log(rho*dnorm(x - mu1) + (1 - rho)*dnorm(x - mu2))
#   ll <- sum(ll)
   ll
}

gradLikMix2 <- function(param, rho) {
   mu1 <- param[1]
   mu2 <- param[2]
   f1 <- dnorm(x - mu1)
   f2 <- dnorm(x - mu2)
   L <- rho*f1 + (1 - rho)*f2
   g <- matrix(0, length(x), 2)
   g[,1] <- rho*(x - mu1)*f1/L
   g[,2] <- (1 - rho)*(x - mu2)*f2/L
#   colSums(g)
   g
}

hessLikMix2 <- function(param, rho) {
   mu1 <- param[1]
   mu2 <- param[2]
   f1 <- dnorm(x - mu1)
   f2 <- dnorm(x - mu2)
   L <- rho*f1 + (1 - rho)*f2
   dldrho <- (f1 - f2)/L
   dldmu1 <- rho*(x - mu1)*f1/L
   dldmu2 <- (1 - rho)*(x - mu2)*f2/L
   h <- matrix(0, 2, 2)
   h[1,1] <- sum(rho*(-f1 + (x - mu1)^2*f1)/L - dldmu1^2)
   h[1,2] <- h[2,1] <- -sum(dldmu1*dldmu2)
   h[2,2] <- sum((1 - rho)*(-f2 + (x - mu2)^2*f2)/L - dldmu2^2)
   h
}

## ----------
A <- matrix(c(1, 2), 1, 2)
B <- 0
start <- c(0, 1)
## nr, numeric gradient
a <- maxLik(logLikMix2,
            start=start, method="nr",
            constraints=list(eqA=A, eqB=B),
            print.level=1, SUMTRho0=1, rho=0.5)
print( summary( a ), digits = 2 )
## nr, numeric hessian
a <- maxLik(logLikMix2, gradLikMix2, 
            start=start, method="nr",
            constraints=list(eqA=A, eqB=B),
            print.level=1, SUMTRho0=1, rho=0.5)
print( summary( a ), digits = 2 )
## nr, analytic hessian
a <- maxLik(logLikMix2, gradLikMix2, hessLikMix2,
            start=start, method="nr",
            constraints=list(eqA=A, eqB=B),
            print.level=1, SUMTRho0=1, rho=0.5)
print( summary( a ), digits = 2 )
## BHHH
a <- maxLik(logLikMix2, gradLikMix2, 
            start=start, method="bhhh",
            constraints=list(eqA=A, eqB=B),
            print.level=1, SUMTRho0=1, rho=0.5)
print( summary( a ), digits = 2 )
## BHHH, analytic
a <- maxLik(logLikMix2, gradLikMix2, 
            start=start, method="bhhh",
            constraints=list(eqA=A, eqB=B),
            print.level=1, SUMTRho0=1, rho=0.5)
print( summary( a ), digits = 2 )
## bfgs, no analytic gradient
a <- maxLik(logLikMix2, 
            start=start, method="bfgs",
            constraints=list(eqA=A, eqB=B),
            print.level=2, SUMTRho0=1, rho=0.5)
print( summary( a ), digits = 2 )
## bfgs, analytic gradient
a <- maxLik(logLikMix2, 
            start=start, method="bfgs",
            constraints=list(eqA=A, eqB=B),
            print.level=2, SUMTRho0=1, rho=0.5)
print( summary( a ), digits = 2 )
## SANN, analytic gradient
a <- maxLik(logLikMix2, gradLikMix2,
            start=start, method="SANN",
            constraints=list(eqA=A, eqB=B),
            print.level=1, SUMTRho0=1, rho=0.5)
print( summary( a ), digits = 2 )
## NM, numeric
a <- maxLik(logLikMix2, 
            start=start, method="nm",
            constraints=list(eqA=A, eqB=B),
            print.level=1, SUMTRho0=1, rho=0.5)
print( summary( a ), digits = 2 )
f <- function(theta) exp(-theta %*% theta)
## NR, multiple constraints
A <- matrix(c(1, 0, 1,
              1, 1, 0), 2, 3, byrow=TRUE)
B <- c(-1, -1)
a <- maxNR(f, start=c(1,1.1,2), constraints=list(eqA=A, eqB=B))
coef(a)
## Error handling for equality constraints
A <- matrix(c(1, 1), 1, 2)
B <- -1
try(a <- maxNR(f, start=c(1, 2, 3), constraints=list(eqA=A, eqB=B)))
                           # ncol(A) != length(start)
A <- matrix(c(1, 1), 1, 2)
B <- c(-1, 2)
try(a <- maxNR(f, start=c(1, 2), constraints=list(eqA=A, eqB=B)))
                           # nrow(A) != nrow(B)
##                           
## -------------- inequality constraints ----------------
##
A <- matrix(c(-1, 0,
              0,  1), 2,2, byrow=TRUE)
B <- c(1,1)
start <- c(0.8, 0.9)
##
a <- maxLik(logLikMix2, gradLikMix2,
            start=start, method="bfgs",
            constraints=list(ineqA=A, ineqB=B),
            rho=0.5)
print( summary( a ), digits = 2 )
##
a <- maxLik(logLikMix2, 
            start=start, method="bfgs",
            constraints=list(ineqA=A, ineqB=B),
            rho=0.5)
print( summary( a ), digits = 2 )
##
a <- maxLik(logLikMix2, gradLikMix2,
            start=start, method="nm",
            constraints=list(ineqA=A, ineqB=B),
            rho=0.5)
print( summary( a ), digits = 2 )
## ---------- test vector B for inequality  --------------
B1 <- c(1,-2)
a <- maxLik(logLikMix2, gradLikMix2,
            start=c(0.5, 2.5), method="bfgs",
            constraints=list(ineqA=A, ineqB=B1),
            rho=0.5)
coef(a)
                           # components should be larger than
                           # (-1, -2)
## ---- Now test error handling: insert wrong A and B forms ----
A2 <- c(-1, 0, 0, 1)
try(maxLik(logLikMix2, gradLikMix2,
           start=start, method="bfgs",
           constraints=list(ineqA=A2, ineqB=B),
           print.level=1, rho=0.5)
    )
                           # should explain that matrix needed
A2 <- matrix(c(-1, 0, 0, 1), 1, 4)
try(maxLik(logLikMix2, gradLikMix2,
           start=start, method="bfgs",
           constraints=list(ineqA=A2, ineqB=B),
           print.level=1, rho=0.5)
    )
                           # should explain that wrong matrix
                           # dimension
B2 <- 1:3
try(maxLik(logLikMix2, gradLikMix2,
           start=start, method="bfgs",
           constraints=list(ineqA=A, ineqB=B2),
           print.level=1, rho=0.5)
    )
                           # A & B do not match
B2 <- matrix(1,2,2)
try(maxLik(logLikMix2, gradLikMix2,
           start=start, method="bfgs",
           constraints=list(ineqA=A, ineqB=B2),
           print.level=1, rho=0.5)
    )
                           # B must be a vector

## fixed parameters with constrained optimization, BFGS.  Thanks to Bob Loos for finding this error.
## Optimize 3D hat with one parameter fixed (== 2D hat).
## Add an equality constraint on that
hat3 <- function(param) {
   ## Hat function.  Hessian negative definite if sqrt(x^2 + y^2) < 0.5
   x <- param[1]
   y <- param[2]
   z <- param[3]
   exp(-x^2-y^2-z^2)
}
library(maxLik)
sv <- c(1,1,1)
## constraints: x + y + z >= 2.5
A <- matrix(c(1,1,1), 1, 3)
B <- -2.5
constraints <- list(ineqA=A, ineqB=B)
res <- maxBFGS(hat3, start=sv, constraints=constraints, fixed=3)
summary(res)
