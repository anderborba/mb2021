maxSANN <- function(fn, grad=NULL, hess=NULL,
                    start, fixed = NULL,
                    print.level=0,
                    iterlim=10000,
                    constraints = NULL,
                    tol=1e-8, reltol=tol,
                    finalHessian=TRUE,
                    cand = NULL,
                    temp=10, tmax=10, parscale=rep(1, length=length(start)),
                    random.seed = 123, ... ) {
   ## Wrapper of optim-based 'SANN' optimization
   ## 
   ## contraints    constraints to be passed to 'constrOptim'
   ## finalHessian:   how (and if) to calculate the final Hessian:
   ##            FALSE   not calculate
   ##            TRUE    use analytic/numeric Hessian
   ##            bhhh/BHHH  use information equality approach
   ##
   ## ... : further arguments to fn()
   ##
   ## Note: grad and hess are for compatibility only, SANN uses only fn values

   # save seed of the random number generator
   if( exists( ".Random.seed" ) ) {
      savedSeed <- .Random.seed
   }

   # set seed for the random number generator (used by 'optim( method="SANN" )')
   set.seed( random.seed )

   # restore seed of the random number generator on exit
   # (end of function or error)
   if( exists( "savedSeed" ) ) {
      on.exit( assign( ".Random.seed", savedSeed, envir = sys.frame() ) )
   } else {
      on.exit( rm( .Random.seed, envir = sys.frame() ) )
   }

   result <- maxOptim( fn = fn, grad = grad, hess = hess,
      start = start, method = "SANN", fixed = fixed,
      print.level = print.level, iterlim = iterlim, constraints = constraints,
      tol = tol, reltol = reltol,
                      finalHessian=finalHessian,
                      parscale = parscale,
      temp = temp, tmax = tmax, random.seed = random.seed, cand = cand,
      ... )

   return(result)
}

