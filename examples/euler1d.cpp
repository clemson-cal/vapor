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
#define gamma (5.0 / 3.0)
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))

#ifdef __NVCC__
#define CACHE_FLUX false
#else
#define CACHE_FLUX true
#endif




int main()
{
    auto cons_to_prim = [] HD (dvec_t<3> u)
    {
        auto rho = u[0];
        auto mom = u[1];
        auto nrg = u[2];
        auto p0 = rho;
        auto p1 = mom / rho;
        auto p2 = (nrg - 0.5 * mom * mom / rho) * (gamma - 1.0);
        return vec(p0, p1, p2);
    };

    auto prim_to_cons = [] HD (dvec_t<3> p)
    {
        auto rho = p[0];
        auto vel = p[1];
        auto pre = p[2];
        auto u0 = rho;
        auto u1 = rho * vel;
        auto u2 = 0.5 * rho * vel * vel + pre / (gamma - 1.0);
        return vec(u0, u1, u2);
    };

    auto prim_and_cons_to_flux = [] HD (dvec_t<3> p, dvec_t<3> u)
    {
        auto vel = p[1];
        auto pre = p[2];
        auto nrg = u[2];
        auto f0 = vel * u[0];
        auto f1 = vel * u[1] + pre;
        auto f2 = vel * (nrg + pre);
        return vec(f0, f1, f2);
    };

    auto sound_speed_squared = [] HD (dvec_t<3> p)
    {
        auto rho = p[0];
        auto pre = p[2];
        return gamma * pre / rho;
    };

    auto riemann_hlle = [=] HD (dvec_t<3> ul, dvec_t<3> ur)
    {
        auto pl = cons_to_prim(ul);
        auto pr = cons_to_prim(ur);
        auto fl = prim_and_cons_to_flux(pl, ul);
        auto fr = prim_and_cons_to_flux(pr, ur);
        auto csl = sqrt(sound_speed_squared(pl));
        auto csr = sqrt(sound_speed_squared(pr));
        auto alm = pl[1] - csl;
        auto alp = pl[1] + csl;
        auto arm = pr[1] - csr;
        auto arp = pr[1] + csr;
        auto am = min3(alm, arm, 0.0);
        auto ap = max3(alp, arp, 0.0);
        return (fl * ap - fr * am - (ul - ur) * ap * am) / (ap - am);
    };

    auto initial_primitive = [] HD (double x)
    {
        if (x < 0.5)
            return vec(1.0, 0.0, 1.0);
        else
            return vec(0.1, 0.0, 0.125);
    };
    auto exec = default_executor_t();
    auto pool = pool_allocator_t();

    auto t_final = 0.1;
    auto N = 20000;
    auto dx = 1.0 / N;
    auto iv = range(N + 1);
    auto ic = range(N);
    auto xc = (ic + 0.5) * dx;
    auto dt = dx * 0.3;
    auto u = xc.map(initial_primitive).map(prim_to_cons).cache(exec, pool);

    auto interior_faces = index_space(uvec(1), uvec(N - 1));
    auto interior_cells = index_space(uvec(1), uvec(N - 2));
    auto t = 0.0;
    auto n = 0;
    auto fold = 50;

    while (t < t_final)
    {
        auto t1 = std::chrono::high_resolution_clock::now();

        for (int m = 0; m < fold; ++m)
        {
            auto fhat = iv[interior_faces].map([u, riemann_hlle] HD (uint i)
            {
                auto ul = u[i - 1];
                auto ur = u[i];
                return riemann_hlle(ul, ur);
            }).cache(exec, pool);

            auto du = ic[interior_cells].map([fhat, dt, dx] HD (uint i)
            {
                auto fm = fhat[i];
                auto fp = fhat[i + 1];
                return (fp - fm) * (-dt / dx);
            });

            u = (u.at(interior_cells) + du).cache(exec, pool);

            t += dt;
            n += 1;
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        auto Mzps = N * fold * 1e-6 / delta.count();
        printf("[%04d] t=%.3lf Mzps=%.3lf\n", n, t, Mzps);
    }

    // auto p = u.map(cons_to_prim);
    // for (int i = 0; i < N; ++i)
    // {
    //     printf("%+.4f %+.4f %+.4f %+.4f\n", xc[i], p[i][0], p[i][1], p[i][2]);
    // }
    return 0;
}
