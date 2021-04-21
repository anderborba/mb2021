loglike_teste <- function(param){
  L    <- param[1]
  mu   <- param[2]
  aux1 <- L * log(L)
  aux2 <- L * lc1
  aux3 <- L * log(mu) 
  aux4 <- log(gamma(L))
  aux5 <- (L / mu) * lc2
  ll   <- aux1 + aux2 - aux3 - aux4 - aux5
  #ll <- -((param[1] - 5)^2 + (param[2] - 6)^2)
  return(ll)
}

