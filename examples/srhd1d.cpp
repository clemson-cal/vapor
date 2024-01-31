/**
================================================================================
Copyright 2023 - 2024, Jonathan Zrake

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
================================================================================
*/




#include <chrono>
#include <cstdio>
#include <cmath>
#include "vapor/array.hpp"
#include "vapor/executor.hpp"

using namespace vapor;
#define gamma (4.0 / 3.0)
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define index_density 0
#define index_pressure 2
#define index_energy 2




/**
 * 
 */
using cons_t = vec_t<double, 3>;
using prim_t = vec_t<double, 3>;




/**
 * 
 */
HD auto gamma_beta_squared(prim_t p) -> double
{
    return p[1] * p[1];
}

HD auto momentum_squared(cons_t u) -> double
{
    return u[1] * u[1];
}

HD auto lorentz_factor(prim_t p) -> double
{
    return sqrt(1.0 + gamma_beta_squared(p));
}

HD auto beta_component(prim_t p, int axis) -> double
{
    return p[axis + 1] / lorentz_factor(p);
}

HD auto enthalpy_density(prim_t p) -> double
{
    auto rho = p[index_density];
    auto pre = p[index_pressure];
    return rho + pre * (1.0 + 1.0 / (gamma - 1.0));
}

HD auto prim_to_cons(prim_t p) -> cons_t
{
    auto rho = p[index_density];
    auto pre = p[index_pressure];
    auto w = lorentz_factor(p);
    auto h = enthalpy_density(p) / rho;
    auto m = rho * w;
    auto u = cons_t{};
    u[0] = m;
    u[1] = m * (h * p[1]);
    u[2] = m * (h * w - 1.0) - pre;
    return u;
}

HD auto cons_to_prim(cons_t cons, double p=0.0) -> prim_t
{
    auto newton_iter_max = 50;
    auto error_tolerance = 1e-12 * (cons[index_density] + cons[index_energy]);
    auto gm = gamma;
    auto m = cons[index_density];
    auto tau = cons[index_energy];
    auto ss = momentum_squared(cons);
    auto n = 0;
    auto w0 = 0.0;

    while (true)
    {
        auto et = tau + p + m;
        auto b2 = min2(ss / et / et, 1.0 - 1e-10);
        auto w2 = 1.0 / (1.0 - b2);
        auto w = sqrt(w2);
        auto d = m / w;
        auto de = (tau + m * (1.0 - w) + p * (1.0 - w2)) / w2;
        auto dh = d + de + p;
        auto a2 = gm * p / dh;
        auto g = b2 * a2 - 1.0;
        auto f = de * (gm - 1.0) - p;

        if (n == newton_iter_max)
        {
            printf("c2p failed; D=%f tau=%f\n", cons[index_density], cons[index_energy]);
            exit(1);
        }
        if (fabs(f) < error_tolerance) {
            w0 = w;
            break;
        }
        p -= f / g;
        n += 1;
    }
    return {m / w0, w0 * cons[1] / (tau + m + p), p};
}

HD auto prim_and_cons_to_flux(prim_t p, cons_t u, int axis) -> cons_t
{
    double pre = p[index_pressure];
    double vn = beta_component(p, axis);
    auto f = cons_t{};
    f[0] = vn * u[0];
    f[1] = vn * u[1] + pre * (axis == 0);
    f[2] = vn * u[2] + pre * vn;
    return f;
}

HD auto sound_speed_squared(prim_t p) -> double
{
    const double pre = p[index_pressure];
    const double rho_h = enthalpy_density(p);
    return gamma * pre / rho_h;
}

HD auto outer_wavespeeds(prim_t p, int axis) -> dvec_t<2>
{
    double a2 = sound_speed_squared(p);
    double uu = gamma_beta_squared(p);
    double vn = beta_component(p, axis);
    double vv = uu / (1.0 + uu);
    double v2 = vn * vn;
    double k0 = sqrt(a2 * (1.0 - vv) * (1.0 - vv * a2 - v2 * (1.0 - a2)));
    return vec(
        (vn * (1.0 - a2) - k0) / (1.0 - vv * a2),
        (vn * (1.0 - a2) + k0) / (1.0 - vv * a2)
    );
}

HD auto riemann_hlle(prim_t pl, prim_t pr, cons_t ul, cons_t ur) -> cons_t
{
    auto fl = prim_and_cons_to_flux(pl, ul, 0);
    auto fr = prim_and_cons_to_flux(pr, ur, 0);
    auto al = outer_wavespeeds(pl, 0);
    auto ar = outer_wavespeeds(pr, 0);
    auto alm = al[0];
    auto alp = al[1];
    auto arm = ar[0];
    auto arp = ar[1];
    auto am = min3(alm, arm, 0.0);
    auto ap = max3(alp, arp, 0.0);
    return (fl * ap - fr * am - (ul - ur) * ap * am) / (ap - am);
};




struct State
{
    double time;
    int iteration;
    memory_backed_array_t<1, cons_t, ref_counted_ptr_t> conserved;
    memory_backed_array_t<1, prim_t, ref_counted_ptr_t> primitive;
};




/**
 * On clang 15.0.0, the update function runs order 50% slower when the update
 * function is inside a struct, versus inside a namespace.
 */
namespace srhd
{
    static void update(State& state)
    {
        auto N = state.conserved.size();
        auto dx = 1.0 / N;
        auto iv = range(N + 1);
        auto ic = range(N);
        auto dt = dx * 0.3;
        auto interior_faces = index_space(ivec(1), uvec(N - 1));
        auto interior_cells = index_space(ivec(1), uvec(N - 2));

        auto u = state.conserved;
        auto p = state.primitive;

        p = ic.map([p, u] HD (int i)
        {
            return cons_to_prim(u[i], p[i][index_pressure]);
        }).cache();

        auto fhat = iv[interior_faces].map([p, u] HD (int i)
        {
            auto ul = u[i - 1];
            auto ur = u[i];
            auto pl = p[i - 1];
            auto pr = p[i];
            return riemann_hlle(pl, pr, ul, ur);
        }).cache();

        auto du = ic[interior_cells].map([fhat, dt, dx] HD (int i)
        {
            auto fm = fhat[i];
            auto fp = fhat[i + 1];
            return (fp - fm) * (-dt / dx);
        });

        u = (u.at(interior_cells) + du).cache();

        state.time += dt;
        state.iteration += 1;
        state.primitive = p;
        state.conserved = u;
    }
}




int main()
{
    auto initial_primitive = [] HD (double x)
    {
        if (x < 0.5)
            return vec(1.0, 0.0, 1.0);
        else
            return vec(0.1, 0.0, 0.125);
    };

    auto t_final = 0.2;
    auto N = 100000;
    auto dx = 1.0 / N;
    auto ic = range(N);
    auto xc = (ic + 0.5) * dx;
    auto p = xc.map(initial_primitive).cache();
    auto u = p.map([] HD (prim_t p) { return prim_to_cons(p); }).cache();
    auto fold = 50;

    auto state = State{
        0.0, 0, u, p,
    };

    while (state.time < t_final)
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        for (int m = 0; m < fold; ++m)
        {
            srhd::update(state);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        auto Mzps = N * fold * 1e-6 / delta.count();
        printf("[%04d] t=%.3lf Mzps=%.3lf\n", state.iteration, state.time, Mzps);
    }

    p = state.conserved.map([] HD (cons_t u) { return cons_to_prim(u, 0.0); }).cache();
    for (int i = 0; i < N; ++i)
    {
        printf("%+.4f %+.4f %+.4f %+.4f\n", xc[i], p[i][0], p[i][1], p[i][2]);
    }
    return 0;
}
