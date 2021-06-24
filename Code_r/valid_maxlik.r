rm(list = ls())
require(maxLik)
library(datasets)
data(CO2)
hist(CO2$uptake)
N <- nrow(CO2)
x <- CO2$uptake
loglik <- function(theta) {
  mu <- theta[1]
  sigma <- theta[2]
  loglik <- -N*log(sqrt(2*pi)) - N*log(sigma) - 0.5*sum((x - mu)^2/sigma^2)
}
res <- maxLik(loglik, start=c(mu=30, sigma=10), method = "BFGS")
res1 <- maxBFGS(loglik, start=c(mu=30, sigma=10))
pt  <- res$estimate
pt1 <- res1$estimate
# imprime em arquivo no diretorio  ~/Data/
dfev <- data.frame(CO2$uptake)
names(dfev) <- NULL
setwd("..")
setwd("Data")
sink("co2_dataset.txt")
print(dfev)
sink()
setwd("..")
setwd("Code_r")
