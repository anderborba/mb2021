\name{summary.maxim}
\alias{summary.maxim}
\title{Summary method for maximisation/minimisation}
\description{
  Summarises the maximisation results
}
\usage{
\method{summary}{maxim}( object, hessian=FALSE, unsucc.step=FALSE, ... )
}
\arguments{
  \item{object}{optimisation result, object of class
    \code{maxim}.  See \code{\link{maxNR}}.}
  \item{hessian}{logical, whether to display Hessian matrix.}
  \item{unsucc.step}{logical, whether to describe last unsuccesful step
    if \code{code} == 3}
  \item{\ldots}{currently not used.}
}
\value{
  Object of class \code{summary.maxim}, intended to print with
  corresponding print method.  There are following components:
  \item{type}{type of maximisation.}
  \item{iterations}{number of iterations.}
  \item{code}{exit code (see \code{\link{maxNR}}.)}
  \item{message}{a brief message, explaining code.}
  \item{unsucc.step}{description of last unsuccessful step, only if
    requested and \code{code} == 3}
  \item{maximum}{function value at maximum}
  \item{estimate}{matrix with following columns:
    \itemize{
      \item{results}{coefficient estimates at maximum}
      \item{gradient}{estimated gradient at maximum}
    }
  }
  \item{constraints}{information about the constrained optimization.
    Passed directly further from \code{maxim}-object.  \code{NULL} if
    unconstrained maximization.
  }
  \item{hessian}{estimated hessian at maximum, only if requested}
}
\author{Ott Toomet \email{siim@obs.ee}}
\seealso{\code{\link{maxNR}}}
\examples{
## minimize a 2D quadratic function:
f <- function(b) {
  x <- b[1]; y <- b[2];
    val <- (x - 2)^2 + (y - 3)^2
    attr(val, "gradient") <- c(2*x - 4, 2*y - 6)
    attr(val, "hessian") <- matrix(c(2, 0, 0, 2), 2, 2)
    val
}
## Note that NR finds the minimum of a quadratic function with a single
## iteration.  Use c(0,0) as initial value.  
result1 <- maxNR( f, start = c(0,0) ) 
summary( result1 )
## Now use c(1000000, -777777) as initial value and ask for hessian
result2 <- maxNR( f, start = c( 1000000, -777777)) 
summary( result2 )
}
\keyword{methods}
\keyword{print}
