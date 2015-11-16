### HAD
HAD is a single header C++ reverse-mode [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) library using operator overloading, with focus on second-order derivatives (Hessian).  
It implements the edge_pushing algorithm (see "Hessian Matrices via Automatic Differentiation", Gower and Mello 2010) to efficiently compute the second derivatives.  
HAD stores the first and second order derivatives coefficients in a single STL vector while recording the function, reducing the number of memory allocation calls (similar to Adept).  
Currently HAD does not support "checkpointing" which can be important for long functions.

### Usage
To compute the derivatives, declare an ADGraph object and rewrite your function with a special data-type "AReal":
```
#include "had.h"
using namespace had;
DECLARE_ADGRAPH();
...
ADGraph adGraph;
AReal x = AReal(1.0);
AReal y = AReal(2.0);

AReal z = sin(x) * cos(y);
```
suppose we want to compute dz/dx and d^2z/dxdy, we need to first propagate the derivatives from z:
```
SetAdjoint(z, 1.0);
PropagateAdjoint();
```
then we obtain the derivatives by calling the GetAdjoint function:
```
double dzdx  = GetAdjoint(x);
double dzdxy = GetAdjoint(x, y);
```
the derivatives are now stored in dzdx and dzdxy.

Finally, remember to clean up the adGraph object in the end if you want to use it again.
```
adGraph.Clear();
```
Note that the edge_pushing algorithm requires all the independent variables (in the above case, x and y) to be declared and be assigned values before any computation happens, or the algorithm can gives incorrect result.

See test.cpp for more usage.  

### Comparison to other libraries
Below is a quick comparison to other operator-overloading-based automatic differentiation libraries.  
In short, to the author's knowledge, HAD is currently the only open source automatic differentiation library that fully utilizes the symmetry of Hessian computation graph, meanwhile stores the computation graph in a compact and efficient manner.  

#### Adept: http://www.met.reading.ac.uk/clouds/adept/  
Adept is a very fast automatic differentiation library that uses expression templates and implicit stored computation graph to better utilize compiler optimization.  
For first derivatives computation, it can be more efficient than HAD because HAD does not use expression templates.  
However, Adept does not support second derivatives.

#### CppAD: http://www.coin-or.org/CppAD/
#### ADOL-C: https://projects.coin-or.org/ADOL-C  
CppAD and ADOL-C both store the whole computation graph with symbolic representation to compute the derivatives.  
This makes them able to record the computation graph once, and feed it with different input variables to obtain derivatives at different points.  
However, since the derivatives comptuation now requires a table lookup with the type of the symbol, they are less efficient compare to HAD.  
They also need to re-record the computation graph once the conditions for the if-else statements or loops are changed.  
Most crucially, they compute the Hessian by interleaving forward and reverse automatic differentiation, which does not fully utilize the symmetry of Hessian computation graph.

#### Stan-math: https://github.com/stan-dev/math  
Stan-math is a carefully tuned auto-diff library with comparable performance with Adept and supports second and even higher-order derivatives computation.  
It also aims for a full compatability with Eigen (HAD has not intensively tested for this yet).  
However, for second-order derivatives they also resort to interleaving between forward and reverse passes.  
In order to obtain a n-dimensional Hessian matrix, they have to run n forward-reverse passes of the same function, which results in many repeated computation.  

#### autodiff.h: http://www.mitsuba-renderer.org/files/eigen/autodiff.h  
autodiff is a single header automatic differentiation library for C++ which can do first and second-order derivatives just like HAD.  
However, autodiff uses forward-mode automatic differentiation which is known to be inefficient compare to reverse-mode when the output dimension is smaller than the input dimension (O(n^2) v.s. O(n) where n is the input dimension of the Hessian matrix).  
The fact that HAD stores the derivatives coefficients in a STL vector also makes it more efficient.  
