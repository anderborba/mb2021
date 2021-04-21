der_loglike_teste <- function(x){
  der <- rep(0,2) 
  der[1]  <- log(x[1] / x[2]) + x[2] - digamma(x[1]) + lc1 - lc2 / x[2]
  der[2]  <- - x[1] / x[2] + (x[1] / x[2]^2) * lc2
  #der[1] <- -2 * (x[1] - 5)
  #der[2] <- -2 * (x[2] - 6)
  return(der)
}