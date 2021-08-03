#from rpy2.robjects.vectors import FloatVector
#from rpy2.robjects.packages import importr
#import rpy2.rinterface as ri
#stats = importr('stats')
#maxLik = importr('maxLik')

# Rosenbrock Banana function as a cost function
# (as in the R man page for optim())
#def loglike(x):
#    x1 = x[0]
#    x2 = x[1]
#    return ((x2-1)**2 + (x1-1)**2)

# wrap the function f so it can be exposed to R
#loglike_func = ri.rternaliarze(loglike)
#
# starting parameters
#start_params = FloatVector((2, 2))
#
# call R's optim() with our cost funtion
#res = stats.optim(start_params, cost_fr)
#res = maxLik.maxBFGS(loglike_func, start = start_params)
#
from rpy2.robjects.vectors import FloatVector
from rpy2.robjects.packages import importr
import rpy2.rinterface as ri
stats = importr('stats')
maxLik = importr('maxLik')
# Rosenbrock Banana function as a cost function
# (as in the R man page for optim())
def cost_f(x):
    x1 = x[0]
    x2 = x[1]
    return (x2 - x1 * x1)**2 + (1 - x1)**2

def loglike(x, a, b, c):
    x1 = x[0]
    x2 = x[1]
    return -((x1 - a)**2 + (x2 - b)**2) + c
# wrap the function f so it can be exposed to R
cost_fr      = ri.rternalize(cost_f)
loglike_func = ri.rternalize(lambda xx:loglike(xx,aa,bb,cc))

# starting parameters
start_params = FloatVector((-1.2, 1))
varx = FloatVector((-1.2, 1))

# call R's optim() with our cost funtion
#res  = stats.optim(start_params, cost_fr)
#res1 = stats.optim(varx, loglike_func)
aa = 3
bb = 5
cc = 10
res2 = maxLik.maxBFGS(loglike_func, start = varx)
