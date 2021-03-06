
stdEr.maxLik <- function(x, ...) {
   if(!inherits(x, "maxLik"))
       stop("'stdEr.maxLik' called on a non-'maxLik' object")
   ## Here we should actually coerce the object to a 'maxLik' object, dropping all the subclasses...
   ## Instead, we force the program to use maxLik-related methods
   if(!is.null(vc <- vcov.maxLik(x))) {
      s <- sqrt(diag(vc))
      names(s) <- names(coef.maxLik(x))
      return(s)
   }
   return(NULL)
                           # if vcov is not working, return NULL
}

