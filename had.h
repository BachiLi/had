/**
    HAD is a single header C++ reverse-mode automatic differentiation library using operator overloading, with focus on 
    second-order derivatives (Hessian).  
    It implements the edge_pushing algorithm (see "Hessian Matrices via Automatic Differentiation", 
    Gower and Mello 2010) to efficiently compute the second derivatives.
    
    See https://github.com/BachiLi/had for more details.
    
    Author: Tzu-Mao Li


    The MIT License (MIT)

    Copyright (c) 2015 Tzu-Mao Li

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
**/

#ifndef HAD_H__
#define HAD_H__

#include <vector>
#include <cmath>
#ifdef WIN32
#define threadDefine thread_local
#endif
#ifdef __unix
#define USE_AATREE
#define threadDefine __thread
#endif

#include <vector>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI std::acos(-1)
#endif

namespace had {

// Change the following line if you want to use single precision floats
typedef double Real; 
typedef unsigned int VertexId;

struct ADGraph;
struct AReal;

extern threadDefine ADGraph* g_ADGraph;
// Declare this in your .cpp source
#define DECLARE_ADGRAPH() namespace had { threadDefine ADGraph* g_ADGraph = 0; }

AReal NewAReal(const Real val);

struct AReal {
    AReal() {}

    AReal(const Real val) {
        *this = NewAReal(val);
    }

    AReal(const Real val, const VertexId varId) : 
        val(val), varId(varId) {}

    Real val;
    VertexId varId;
};

struct ADEdge {
    ADEdge() {}
    ADEdge(const VertexId to, const Real w = Real(0.0)) : 
        to(to), w(w) {}

    VertexId to;
    Real w;
};

// We assume there is at most 2 outgoing edges from this vertex
struct ADVertex {
    ADVertex(const VertexId newId) {
        e1 = e2 = ADEdge(newId);
        w = soW = Real(0.0);
    }

    // If ei.to == the id of this vertex, then the edge does not exist
    ADEdge e1, e2;
    // first-order weight
    Real w;
    // second-order weights
    // for vertex with single outgoing edge, 
    // soW represents the second-order weight of the conntecting vertex (d^2f/dx^2)
    // for vertex with two outgoing edges,
    // soW represents the second-order weight between the conntecting vertices (d^2f/dxdy)
    // the system assumes d^2f/dx^2 & d^2f/dy^2 are both zero in the two outgoing edges case to save memory
    Real soW;
};

struct BTNode {
    BTNode() {}
    BTNode(const VertexId key, const Real val) : key(key), val(val) {
        left = right = - 1;
#ifdef USE_AATREE
        level = 1;
#endif
    }

    VertexId key;
    Real val;
    int left;
    int right;
#ifdef USE_AATREE
    int level;
#endif
};

struct BTree {
    BTree() {
        nodes.reserve(32);
        root = 0;
    }
#ifdef USE_AATREE
    inline void Skew() {
        if (nodes.size() == 0 ||
            nodes[root].left == -1) {
            return;
        }
        while (nodes[nodes[root].left].level == nodes[root].level) {
            int l = nodes[root].left;
            nodes[root].left = nodes[l].right;
            nodes[l].right = root;
            root = l;
        }
    }

    inline void Split() {
        if (nodes.size() == 0 ||
            nodes[root].right == -1 ||
            nodes[nodes[root].right].right == -1) {
            return;
        }
        while (nodes[root].level == nodes[nodes[nodes[root].right].right].level) {
            int r = nodes[root].right;
            nodes[root].right = nodes[r].left;
            nodes[r].left = root;
            nodes[r].level++;
            root = r;
        }
    }
#endif
    inline void Insert(const VertexId key, const Real val) {
        int index = root;
        if (nodes.size() > 0) {
            int *lastEdge;
            do {
                if (key == nodes[index].key) {
                    nodes[index].val += val;
                    return;
                }
                lastEdge = &(nodes[index].left) + (key > nodes[index].key);
                index = *lastEdge;
            } while (index >= 0);

            *lastEdge = nodes.size();
        }
        nodes.push_back(BTNode(key, val));
#ifdef USE_AATREE
        Skew();
        Split();
#endif
    }

    inline Real Query(const VertexId key) {
        int index = root;
        while (index >= 0 && index < (int)nodes.size()) {
            if (key == nodes[index].key) {
                return nodes[index].val;
            } else if (key < nodes[index].key) {
                index = nodes[index].left;
            } else {
                index = nodes[index].right;
            }
        }
        return Real(0.0);
    }

    inline void Clear() {
        nodes.clear();
        root = 0;
    }

    std::vector<BTNode> nodes;
    int root;
};

struct ADGraph {
    ADGraph() {
        g_ADGraph = this;
    }

    inline void Clear() {
        vertices.clear();
        soEdges.clear();
        selfSoEdges.clear();
    }

    std::vector<ADVertex> vertices;
    std::vector<BTree> soEdges;
    std::vector<Real> selfSoEdges;
};

inline AReal NewAReal(const Real val) {
    std::vector<ADVertex> &vertices = g_ADGraph->vertices;
    VertexId newId = vertices.size();
    vertices.push_back(ADVertex(newId));
    return AReal(val, newId);
}

inline void AddEdge(const AReal &c, const AReal &p, 
                    const Real w, const Real soW) {
    ADVertex &v = g_ADGraph->vertices[c.varId];
    v.e1 = ADEdge(p.varId, w);
    v.soW = soW;
}
inline void AddEdge(const AReal &c, 
                    const AReal &p1, const AReal &p2, 
                    const Real w1, const Real w2,
                    const Real soW) {
    ADVertex &v = g_ADGraph->vertices[c.varId];
    v.e1 = ADEdge(p1.varId, w1);
    v.e2 = ADEdge(p2.varId, w2);
    v.soW = soW;
}

////////////////////// Addition ///////////////////////////
inline AReal operator+(const AReal &l, const AReal &r) {
    AReal ret = NewAReal(l.val + r.val);
    AddEdge(ret, l, r, Real(1.0), Real(1.0), Real(0.0));
    return ret;
}
inline AReal operator+(const AReal &l, const Real r) {
    AReal ret = NewAReal(l.val + r);
    AddEdge(ret, l, Real(1.0), Real(0.0));
    return ret;
}
inline AReal operator+(const Real l, const AReal &r) {
    return r + l;
}
inline AReal& operator+=(AReal &l, const AReal &r) {
    return (l = l + r);
}
inline AReal& operator+=(AReal &l, const Real r) {
    return (l = l + r);
}
///////////////////////////////////////////////////////////

////////////////// Subtraction ////////////////////////////
inline AReal operator-(const AReal &l, const AReal &r) {
    AReal ret = NewAReal(l.val - r.val);
    AddEdge(ret, l, r, Real(1.0), -Real(1.0), Real(0.0));
    return ret;
}
inline AReal operator-(const AReal &l, const Real r) {
    AReal ret = NewAReal(l.val - r);
    AddEdge(ret, l, Real(1.0), Real(0.0));
    return ret;
}
inline AReal operator-(const Real l, const AReal &r) {
    AReal ret = NewAReal(l - r.val);
    AddEdge(ret, r, Real(-1.0), Real(0.0));
    return ret;
}
inline AReal& operator-=(AReal &l, const AReal &r) {
    return (l = l - r);
}
inline AReal& operator-=(AReal &l, const Real r) {
    return (l = l - r);
}
inline AReal operator-(const AReal &x) {
    AReal ret = NewAReal(-x.val);
    AddEdge(ret, x, Real(-1.0), Real(0.0));
    return ret;
}
///////////////////////////////////////////////////////////

////////////////// Multiplication /////////////////////////
inline AReal operator*(const AReal &l, const AReal &r) {
    AReal ret = NewAReal(l.val * r.val);
    AddEdge(ret, l, r, r.val, l.val, Real(1.0));
    return ret;
}
inline AReal operator*(const AReal &l, const Real r) {
    AReal ret = NewAReal(l.val * r);
    AddEdge(ret, l, r, Real(0.0));
    return ret;
}
inline AReal operator*(const Real l, const AReal &r) {
    return r * l;
}
inline AReal& operator*=(AReal &l, const AReal &r) {
    return (l = l * r);
}
inline AReal& operator*=(AReal &l, const Real r) {
    return (l = l * r);
}
///////////////////////////////////////////////////////////

////////////////// Inversion //////////////////////////////
inline AReal Inv(const AReal &x) {
    Real invX = Real(1.0) / x.val;
    Real invXSq = invX * invX;
    Real invXCu = invXSq * invX;
    AReal ret = NewAReal(invX);
    AddEdge(ret, x, -invXSq, Real(2.0) * invXCu);
    return ret;
}
inline Real Inv(const Real x) {
    return Real(1.0) / x;
}
///////////////////////////////////////////////////////////

////////////////// Division ///////////////////////////////
inline AReal operator/(const AReal &l, const AReal &r) {
    return l * Inv(r);
}
inline AReal operator/(const AReal &l, const Real r) {
    return l * Inv(r);
}
inline AReal operator/(const Real l, const AReal &r) {
    return l * Inv(r);
}
inline AReal& operator/=(AReal &l, const AReal &r) {
    return (l = l / r);
}
inline AReal& operator/=(AReal &l, const Real r) {
    return (l = l / r);
}
///////////////////////////////////////////////////////////

////////////////// Comparisons ////////////////////////////
inline bool operator<(const AReal &l, const AReal &r) {
    return l.val < r.val;
}
inline bool operator<=(const AReal &l, const AReal &r) {
    return l.val <= r.val;
}
inline bool operator>(const AReal &l, const AReal &r) {
    return l.val > r.val;
}
inline bool operator>=(const AReal &l, const AReal &r) {
    return l.val >= r.val;
}
inline bool operator==(const AReal &l, const AReal &r) {
    return l.val == r.val;
}
///////////////////////////////////////////////////////////

//////////////// Misc functions ///////////////////////////
inline Real square(const Real x) {
    return x * x;
}
inline AReal square(const AReal &x) {
    Real sqX = x.val * x.val;
    AReal ret = NewAReal(sqX);
    AddEdge(ret, x, Real(2.0) * x.val, Real(0.0));
    return ret;
}
inline AReal sqrt(const AReal &x) {
    Real sqrtX = std::sqrt(x.val);
    Real invSqrtX = Real(1.0) / sqrtX;
    AReal ret = NewAReal(sqrtX);
    AddEdge(ret, x, Real(0.5) * invSqrtX, - Real(0.25) * invSqrtX / x.val);
    return ret;
}
inline AReal pow(const AReal &x, const Real a) {
    Real powX = std::pow(x.val, a);
    AReal ret = NewAReal(powX);
    AddEdge(ret, x, a * std::pow(x.val, a - Real(1.0)),
                    a * (a - Real(1.0)) * std::pow(x.val, a - Real(2.0)));
    return ret;
}
inline AReal exp(const AReal &x) {
    Real expX = std::exp(x.val);
    AReal ret = NewAReal(expX);
    AddEdge(ret, x, expX, expX);
    return ret;
}
inline AReal log(const AReal &x) {
    Real logX = std::log(x.val);
    AReal ret = NewAReal(logX);
    Real invX = Real(1.0) / x.val;
    AddEdge(ret, x, invX, - invX * invX);
    return ret;
}
inline AReal sin(const AReal &x) {
    Real sinX = std::sin(x.val);
    AReal ret = NewAReal(sinX);
    AddEdge(ret, x, std::cos(x.val), -sinX);
    return ret;
}
inline AReal cos(const AReal &x) {
    Real cosX = std::cos(x.val);
    AReal ret = NewAReal(cosX);
    AddEdge(ret, x, -std::sin(x.val), -cosX);
    return ret;
}
inline AReal tan(const AReal &x) {
    Real tanX = std::tan(x.val);
    Real secX = Real(1.0) / std::cos(x.val);
    Real sec2X = secX * secX;
    AReal ret = NewAReal(tanX);
    AddEdge(ret, x, sec2X, Real(2.0) * tanX * sec2X);
    return ret;
}
inline AReal asin(const AReal &x) {
    Real asinX = std::asin(x.val);
    AReal ret = NewAReal(asinX);
    Real tmp = Real(1.0) / (Real(1.0) - x.val * x.val);
    Real sqrtTmp = std::sqrt(tmp);
    AddEdge(ret, x, sqrtTmp, x.val * sqrtTmp * tmp);
    return ret;
}
inline AReal acos(const AReal &x) {
    Real acosX = std::acos(x.val);
    AReal ret = NewAReal(acosX);
    Real tmp = Real(1.0) / (Real(1.0) - x.val * x.val);
    Real negSqrtTmp = -std::sqrt(tmp);
    AddEdge(ret, x, negSqrtTmp, x.val * negSqrtTmp * tmp);
    return ret;
}
///////////////////////////////////////////////////////////

inline void SetAdjoint(const AReal &v, const Real adj) {
    g_ADGraph->vertices[v.varId].w = adj;
}

inline Real GetAdjoint(const AReal &v) {
    return g_ADGraph->vertices[v.varId].w;
}

inline Real GetAdjoint(const AReal &i, const AReal &j) {
    if (i.varId == j.varId) {
        return g_ADGraph->selfSoEdges[i.varId];
    } else {
        return g_ADGraph->soEdges[std::max(i.varId, j.varId)].Query(std::min(i.varId, j.varId));
    }
}

inline VertexId SingleEdgePropagate(VertexId x, Real &a) {
    bool cont = g_ADGraph->vertices[x].e1.to != x &&
                g_ADGraph->vertices[x].e2.to == x;
    while (cont) {
        a *= g_ADGraph->vertices[x].e1.w;
        x = g_ADGraph->vertices[x].e1.to;
        cont = g_ADGraph->vertices[x].e1.to != x &&
               g_ADGraph->vertices[x].e2.to == x;
    }
    return x;
}

inline void PushEdge(const ADEdge &foEdge, const ADEdge &soEdge) {
    if (foEdge.to == soEdge.to) {
        g_ADGraph->selfSoEdges[foEdge.to] += Real(2.0) * foEdge.w * soEdge.w;
    } else {
        g_ADGraph->soEdges[std::max(foEdge.to, soEdge.to)].Insert(
            std::min(foEdge.to, soEdge.to), foEdge.w * soEdge.w);
    }
}

inline void PropagateAdjoint() {
    if (g_ADGraph->vertices.size() > g_ADGraph->soEdges.size()) {
        g_ADGraph->soEdges.resize(g_ADGraph->vertices.size());
    } else {
        for (int i = 0; i < (int)g_ADGraph->soEdges.size(); i++) {
            g_ADGraph->soEdges[i].Clear();
        }
    }
    g_ADGraph->selfSoEdges.resize(g_ADGraph->vertices.size(), Real(0.0));
    // Any chance for SSE/AVX parallism?

    for (VertexId vid = g_ADGraph->vertices.size() - 1; vid > 0; vid--) {
        ADVertex &vertex = g_ADGraph->vertices[vid];
        ADEdge &e1 = vertex.e1;
        ADEdge &e2 = vertex.e2;
        if (e1.to == vid) {
            continue;
        }

        // Pushing
        BTree &btree = g_ADGraph->soEdges[vid];
        std::vector<BTNode>::iterator it;
        if (e2.to == vid) {
            for (it = btree.nodes.begin(); it != btree.nodes.end(); it++) {
                ADEdge soEdge(it->key, it->val);
                PushEdge(e1, soEdge);
            }
        } else {
            for (it = btree.nodes.begin(); it != btree.nodes.end(); it++) {
                ADEdge soEdge(it->key, it->val);
                PushEdge(e1, soEdge);
                PushEdge(e2, soEdge);
            }
        }
        if (g_ADGraph->selfSoEdges[vid] != Real(0.0)) {
            g_ADGraph->selfSoEdges[e1.to] += e1.w * e1.w * g_ADGraph->selfSoEdges[vid];
            if (e2.to != vid) {
                g_ADGraph->selfSoEdges[e2.to] += e2.w * e2.w * g_ADGraph->selfSoEdges[vid];
                if (e1.to == e2.to) {
                    g_ADGraph->selfSoEdges[e2.to] += Real(2.0) * e1.w * e2.w * g_ADGraph->selfSoEdges[vid];
                } else {
                    g_ADGraph->soEdges[std::max(e1.to, e2.to)].Insert(std::min(e1.to, e2.to), 
                                                   e1.w * e2.w * g_ADGraph->selfSoEdges[vid]);
                }
            }
        }

        // release memory?

        Real a = vertex.w;
        if (a != Real(0.0)) {
            // Creating
            if (vertex.soW != Real(0.0)) {
                if (e2.to == vid) { // single-edge
                    g_ADGraph->selfSoEdges[e1.to] += a * vertex.soW;
                } else if (e1.to == e2.to) {
                    g_ADGraph->selfSoEdges[e1.to] += Real(2.0) * a * vertex.soW;
                } else {
                    g_ADGraph->soEdges[std::max(e1.to, e2.to)].Insert(std::min(e1.to, e2.to),
                        a * vertex.soW);
                }
            }
            // Adjoint
            vertex.w = Real(0.0);
            g_ADGraph->vertices[e1.to].w += a * e1.w;
            if (e2.to != vid) {
                g_ADGraph->vertices[e2.to].w += a * e2.w;
            }
        }
    }
}

} //namespace had

#endif // HAD_H__
