HAD is a single header C++ reverse-mode automatic differentiation library using operator overloading, with focus on 
second-order derivatives (Hessian).  
It implements the edge_pushing algorithm (see "Hessian Matrices via Automatic Differentiation", 
Gower and Mello 2010) to efficiently compute the second derivatives.
The computation graph is built implicitly similar to the Adept library (http://www.met.reading.ac.uk/clouds/adept/),
only the floating point weights are stored.
HAD also stores the derivatives coefficients in a STL vector when recording the function, reducing the number of
memory allocation calls.
Currently HAD does not support "checkpointing" which is important for very very long functions.

To compute the derivatives, declare a ADGraph object and rewrite your function with a special data-type "AReal":
```
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

See test.cpp for more usage.
The library depends on Eigen (http://eigen.tuxfamily.org/).

Below is a quick comparison to other auto-diff libraries:

Adept: http://www.met.reading.ac.uk/clouds/adept/
Adept uses expression templates and implicit stored computation graph to better utilize compiler optimization.
For first derivatives computation, it can be faster than HAD because HAD does not use expression templates.
However, Adept does not support second derivatives.

CppAD: http://www.coin-or.org/CppAD/ & ADOL-C: https://projects.coin-or.org/ADOL-C
CppAD and ADOL-C store the whole computation graph with symbolic representation to compute the derivatives.
This makes them able to record the computation graph once, and feed it with different input variables to obtain 
derivatives at different points.  
However, since the derivatives comptuation now requires a table lookup with the type of the symbol, they are less 
efficient compare to HAD.
They also need to re-record the computation graph once the conditions for the if-else statements or loops are changed.
Most crucially, they compute the Hessian by interleaving forward and reverse automatic differentiation, 
which does not fully utilize the symmetry of Hessian computation graph.

Stan-math: https://github.com/stan-dev/math
Stan-math is a carefully tuned auto-diff library with comparable performance with Adept and supports second and even 
higher-order derivatives computation.
It also has a full compatability with the Eigen library (for HAD, I haven't intensively tested it yet).
However, for second-order derivatives they also resort to interleaving between forward and reverse passes.  
In order to obtain a n-dimensional Hessian matrix, they have to run n forward-reverse passes of the same function, 
which results in many repeated computation.  They also do not utilize the symmetry of Hessian.

autodiff.h: http://www.mitsuba-renderer.org/files/eigen/autodiff.h
autodiff is a single header automatic differentiation library for C++ which can do first and second-order derivatives 
just like HAD.
However, autodiff uses forward-mode automatic differentiation which is known to be inefficient compare to reverse-mode 
when the output dimension is smaller than the input dimension (O(n^2) v.s. O(n) where n is the input dimension of the
Hessian matrix).
The fact that HAD stores the derivatives coefficients in a STL vector also make it more efficient.