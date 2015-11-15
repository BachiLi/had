#include "had.h"

#include <cassert>
#include <cmath>

using namespace std;
using namespace had;

DECLARE_ADGRAPH();

#define NearEqualAssert(a, b) \
    assert(std::fabs((a) - (b)) < 1e-8)

void TestAdd() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(1.0));
    AReal x1 = AReal(Real(2.0));
    AReal x2 = AReal(Real(3.0));

    AReal y = x0 + x0 + x1 + x2;
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);
    Real dydx2 = GetAdjoint(x2);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx0x2 = GetAdjoint(x0, x2);
    Real dydx1x1 = GetAdjoint(x1, x1);
    Real dydx1x2 = GetAdjoint(x1, x2);
    Real dydx2x2 = GetAdjoint(x2, x2);

    NearEqualAssert(dydx0, Real(2.0));
    NearEqualAssert(dydx1, Real(1.0));
    NearEqualAssert(dydx2, Real(1.0));

    NearEqualAssert(dydx0x0, Real(0.0));
    NearEqualAssert(dydx0x1, Real(0.0));
    NearEqualAssert(dydx0x2, Real(0.0));
    NearEqualAssert(dydx1x1, Real(0.0));
    NearEqualAssert(dydx1x2, Real(0.0));
    NearEqualAssert(dydx2x2, Real(0.0));
}

void TestMinus() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(1.0));
    AReal x1 = AReal(Real(2.0));
    AReal x2 = AReal(Real(3.0));

    AReal y = x0 - x0 - x1 - x2;
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);
    Real dydx2 = GetAdjoint(x2);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx0x2 = GetAdjoint(x0, x2);
    Real dydx1x1 = GetAdjoint(x1, x1);
    Real dydx1x2 = GetAdjoint(x1, x2);
    Real dydx2x2 = GetAdjoint(x2, x2);

    NearEqualAssert(dydx0, Real(0.0));
    NearEqualAssert(dydx1, Real(-1.0));
    NearEqualAssert(dydx2, Real(-1.0));

    NearEqualAssert(dydx0x0, Real(0.0));
    NearEqualAssert(dydx0x1, Real(0.0));
    NearEqualAssert(dydx0x2, Real(0.0));
    NearEqualAssert(dydx1x1, Real(0.0));
    NearEqualAssert(dydx1x2, Real(0.0));
    NearEqualAssert(dydx2x2, Real(0.0));
}

void TestMultiply() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(1.0));
    AReal x1 = AReal(Real(2.0));
    AReal x2 = AReal(Real(3.0));

    AReal y = x0 * x0 * x1 * x2;
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);
    Real dydx2 = GetAdjoint(x2);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx0x2 = GetAdjoint(x0, x2);
    Real dydx1x1 = GetAdjoint(x1, x1);
    Real dydx1x2 = GetAdjoint(x1, x2);
    Real dydx2x2 = GetAdjoint(x2, x2);

    NearEqualAssert(dydx0, Real(2.0) * x0.val * x1.val * x2.val);
    NearEqualAssert(dydx1, x0.val * x0.val * x2.val);
    NearEqualAssert(dydx2, x0.val * x0.val * x1.val);

    NearEqualAssert(dydx0x0, Real(2.0) * x1.val * x2.val);
    NearEqualAssert(dydx0x1, Real(2.0) * x0.val * x2.val);
    NearEqualAssert(dydx0x2, Real(2.0) * x0.val * x1.val);
    NearEqualAssert(dydx1x1, Real(0.0));
    NearEqualAssert(dydx1x2, x0.val * x0.val);
    NearEqualAssert(dydx2x2, Real(0.0));
}

void TestDivision() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(1.0));
    AReal x1 = AReal(Real(2.0));
    AReal x2 = AReal(Real(3.0));

    AReal y = (x0 * x0) / (x1 * x2);
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);
    Real dydx2 = GetAdjoint(x2);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx0x2 = GetAdjoint(x0, x2);
    Real dydx1x1 = GetAdjoint(x1, x1);
    Real dydx1x2 = GetAdjoint(x1, x2);
    Real dydx2x2 = GetAdjoint(x2, x2);

    NearEqualAssert(dydx0, Real(2.0) * x0.val / (x1.val * x2.val));
    NearEqualAssert(dydx1, - (x0.val * x0.val) / (x1.val * x1.val * x2.val));
    NearEqualAssert(dydx2, - (x0.val * x0.val) / (x1.val * x2.val * x2.val));

    NearEqualAssert(dydx0x0, Real(2.0) / (x1.val * x2.val));
    NearEqualAssert(dydx0x1, - Real(2.0) * x0.val / (x1.val * x1.val * x2.val));
    NearEqualAssert(dydx0x2, - Real(2.0) * x0.val / (x1.val * x2.val * x2.val));
    NearEqualAssert(dydx1x1, (Real(2.0) * x0.val * x0.val) / (x1.val * x1.val * x1.val * x2.val));
    NearEqualAssert(dydx1x2, x0.val * x0.val / (x1.val * x1.val * x2.val * x2.val));
    NearEqualAssert(dydx2x2, (Real(2.0) * x0.val * x0.val) / (x1.val * x2.val * x2.val * x2.val));
}

void TestSqrt() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(1.0));
    AReal x1 = AReal(Real(2.0));
    AReal x2 = AReal(Real(3.0));

    AReal y = sqrt(x0 * x1 + x2);
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);
    Real dydx2 = GetAdjoint(x2);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx0x2 = GetAdjoint(x0, x2);
    Real dydx1x1 = GetAdjoint(x1, x1);
    Real dydx1x2 = GetAdjoint(x1, x2);
    Real dydx2x2 = GetAdjoint(x2, x2);


    NearEqualAssert(dydx0, x1.val / (Real(2.0) * sqrt(x0.val * x1.val + x2.val)));
    NearEqualAssert(dydx1, x0.val / (Real(2.0) * sqrt(x0.val * x1.val + x2.val)));
    NearEqualAssert(dydx2, Real(1.0) / (Real(2.0) * sqrt(x0.val * x1.val + x2.val)));

    NearEqualAssert(dydx0x0, - (x1.val * x1.val) / (Real(4.0) * pow(x0.val * x1.val + x2.val, Real(3.0/2.0))));
    NearEqualAssert(dydx0x1, (x0.val * x1.val + Real(2.0) * x2.val) / (Real(4.0) * pow(x0.val * x1.val + x2.val, Real(3.0/2.0))));
    NearEqualAssert(dydx0x2, - (x1.val) / (Real(4.0) * pow(x0.val * x1.val + x2.val, Real(3.0/2.0))));
    NearEqualAssert(dydx1x1, - (x0.val * x0.val) / (Real(4.0) * pow(x0.val * x1.val + x2.val, Real(3.0/2.0))));
    NearEqualAssert(dydx1x2, - (x0.val) / (Real(4.0) * pow(x0.val * x1.val + x2.val, Real(3.0/2.0))));
    NearEqualAssert(dydx2x2, - Real(1.0) / (Real(4.0) * pow(x0.val * x1.val + x2.val, Real(3.0/2.0))));
}

void TestPow() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(1.0));
    AReal x1 = AReal(Real(2.0));

    AReal y = pow(x0 + x1, Real(5.3));
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx1x1 = GetAdjoint(x1, x1);

    NearEqualAssert(dydx0, Real(5.3) * pow(x0.val + x1.val, Real(4.3)));
    NearEqualAssert(dydx1, Real(5.3) * pow(x0.val + x1.val, Real(4.3)));

    NearEqualAssert(dydx0x0, Real(22.79) * pow(x0.val + x1.val, Real(3.3)));
    NearEqualAssert(dydx0x1, Real(22.79) * pow(x0.val + x1.val, Real(3.3)));
    NearEqualAssert(dydx1x1, Real(22.79) * pow(x0.val + x1.val, Real(3.3)));
}

void TestExp() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(1.0));
    AReal x1 = AReal(Real(2.0));

    AReal y = exp(x0 * x0 + x1);
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx1x1 = GetAdjoint(x1, x1);

    NearEqualAssert(dydx0, Real(2.0) * x0.val * exp(x0.val * x0.val + x1.val));
    NearEqualAssert(dydx1, exp(x0.val * x0.val + x1.val));

    NearEqualAssert(dydx0x0, Real(2.0) * (Real(2.0) * x0.val * x0.val + Real(1.0)) * exp(x0.val * x0.val + x1.val));
    NearEqualAssert(dydx0x1, Real(2.0) * x0.val * exp(x0.val * x0.val + x1.val));
    NearEqualAssert(dydx1x1, exp(x0.val * x0.val + x1.val));
}

void TestLog() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(1.0));
    AReal x1 = AReal(Real(2.0));

    AReal y = log(x0 * x0 + x1);
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx1x1 = GetAdjoint(x1, x1);

    NearEqualAssert(dydx0, Real(2.0) * x0.val / (x0.val * x0.val + x1.val));
    NearEqualAssert(dydx1, Real(1.0) / (x0.val * x0.val + x1.val));

    NearEqualAssert(dydx0x0, - Real(2.0) * (x0.val * x0.val - x1.val) / ((x0.val * x0.val + x1.val) * (x0.val * x0.val + x1.val)));
    NearEqualAssert(dydx0x1, - Real(2.0) * x0.val / ((x0.val * x0.val + x1.val) * (x0.val * x0.val + x1.val)));
    NearEqualAssert(dydx1x1, - Real(1.0) / ((x0.val * x0.val + x1.val) * (x0.val * x0.val + x1.val)));
}

void TestSin() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(1.0));
    AReal x1 = AReal(Real(2.0));

    AReal y = sin(x0) * sin(x1);
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx1x1 = GetAdjoint(x1, x1);

    NearEqualAssert(dydx0, cos(x0.val) * sin(x1.val));
    NearEqualAssert(dydx1, sin(x0.val) * cos(x1.val));

    NearEqualAssert(dydx0x0, - sin(x0.val) * sin(x1.val));
    NearEqualAssert(dydx0x1, cos(x0.val) * cos(x1.val));
    NearEqualAssert(dydx1x1, - sin(x0.val) * sin(x1.val));
}

void TestCos() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(1.0));
    AReal x1 = AReal(Real(2.0));

    AReal y = cos(x0) * cos(x1);
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx1x1 = GetAdjoint(x1, x1);

    NearEqualAssert(dydx0, - sin(x0.val) * cos(x1.val));
    NearEqualAssert(dydx1, - cos(x0.val) * sin(x1.val));

    NearEqualAssert(dydx0x0, - cos(x0.val) * cos(x1.val));
    NearEqualAssert(dydx0x1,   sin(x0.val) * sin(x1.val));
    NearEqualAssert(dydx1x1, - cos(x0.val) * cos(x1.val));
}

void TestTan() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(1.0));
    AReal x1 = AReal(Real(2.0));

    AReal y = tan(x0) * tan(x1);
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx1x1 = GetAdjoint(x1, x1);

    NearEqualAssert(dydx0, tan(x1.val) / (cos(x0.val) * cos(x0.val)));
    NearEqualAssert(dydx1, tan(x0.val) / (cos(x1.val) * cos(x1.val)));

    NearEqualAssert(dydx0x0, Real(2.0) * tan(x0.val) * tan(x1.val) / (cos(x0.val) * cos(x0.val)));
    NearEqualAssert(dydx0x1, Real(1.0) / (cos(x0.val) * cos(x0.val) * cos(x1.val) * cos(x1.val)));
    NearEqualAssert(dydx1x1, Real(2.0) * tan(x0.val) * tan(x1.val) / (cos(x1.val) * cos(x1.val)));
}

void TestASin() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(0.3));
    AReal x1 = AReal(Real(0.6));

    AReal y = asin(x0) * asin(x1);
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx1x1 = GetAdjoint(x1, x1);

    NearEqualAssert(dydx0, asin(x1.val) / sqrt(Real(1.0) - x0.val * x0.val));
    NearEqualAssert(dydx1, asin(x0.val) / sqrt(Real(1.0) - x1.val * x1.val));

    NearEqualAssert(dydx0x0, x0.val * asin(x1.val) / pow(Real(1.0) - x0.val * x0.val, Real(3.0) / Real(2.0)));
    NearEqualAssert(dydx0x1, Real(1.0) / (sqrt(Real(1.0) - x0.val * x0.val) * sqrt(Real(1.0) - x1.val * x1.val)));
    NearEqualAssert(dydx1x1, x1.val * asin(x0.val) / pow(Real(1.0) - x1.val * x1.val, Real(3.0) / Real(2.0)));
}

void TestACos() {
    ADGraph adGraph;

    AReal x0 = AReal(Real(0.3));
    AReal x1 = AReal(Real(0.6));

    AReal y = acos(x0) * acos(x1);
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx0 = GetAdjoint(x0);
    Real dydx1 = GetAdjoint(x1);

    Real dydx0x0 = GetAdjoint(x0, x0);
    Real dydx0x1 = GetAdjoint(x0, x1);
    Real dydx1x1 = GetAdjoint(x1, x1);

    NearEqualAssert(dydx0, - acos(x1.val) / sqrt(Real(1.0) - x0.val * x0.val));
    NearEqualAssert(dydx1, - acos(x0.val) / sqrt(Real(1.0) - x1.val * x1.val));

    NearEqualAssert(dydx0x0, - x0.val * acos(x1.val) / pow(Real(1.0) - x0.val * x0.val, Real(3.0) / Real(2.0)));
    NearEqualAssert(dydx0x1, Real(1.0) / (sqrt(Real(1.0) - x0.val * x0.val) * sqrt(Real(1.0) - x1.val * x1.val)));
    NearEqualAssert(dydx1x1, - x1.val * acos(x0.val) / pow(Real(1.0) - x1.val * x1.val, Real(3.0) / Real(2.0)));
}

void TestCopy() {
    ADGraph adGraph;

    AReal x = AReal(Real(0.3));
    AReal tmp = x;
    AReal y = x * tmp;
    SetAdjoint(y, Real(1.0));
    PropagateAdjoint();

    Real dydx = GetAdjoint(x);

    Real dydxx = GetAdjoint(x, x);

    NearEqualAssert(dydx, Real(2.0) * x.val);

    NearEqualAssert(dydxx, Real(2.0));
}

int main(int argc, char *argv[]) {
    TestAdd();
    TestMinus();
    TestMultiply();
    TestDivision();
    TestSqrt();
    TestPow();
    TestExp();
    TestLog();
    TestSin();
    TestCos();
    TestTan();
    TestASin();
    TestACos();
    TestCopy();
    
    return 0;
}