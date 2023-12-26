#include <chrono>
#include <cstdio>
#include <cmath>
#include "core_array.hpp"

using namespace vapor;
#define gamma (5.0 / 3.0)
#define min2(a, b) ((a) < (b) ? (a) : (b))
#define max2(a, b) ((a) > (b) ? (a) : (b))
#define min3(a, b, c) min2(a, min2(b, c))
#define max3(a, b, c) max2(a, max2(b, c))
#define cache_flux true
#define vector_components 1
#define index_density 0
#define index_pressure 2
#define index_energy 2
#define num_prim 3
#define num_cons 3




int main()
{
    auto gamma_beta_squared = [=] (dvec_t<num_prim> p)
    {
        if constexpr (vector_components == 1)
        {
            return p[1] * p[1];
        }
        if constexpr (vector_components == 2)
        {
            return p[1] * p[1] + p[2] * p[2];
        }
        if constexpr (vector_components == 3)
        {
            return p[1] * p[1] + p[2] * p[2] + p[3] * p[3];
        }
        return 0.0;
    };

    auto momentum_squared = [=] (dvec_t<num_prim> u)
    {
        if constexpr (vector_components == 1)
        {
            return u[1] * u[1];
        }
        if constexpr (vector_components == 2)
        {
            return u[1] * u[1] + u[2] * u[2];
        }
        if constexpr (vector_components == 3)
        {
            return u[1] * u[1] + u[2] * u[2] + u[3] * u[3];
        }
        return 0.0;
    };

    auto lorentz_factor = [=] (dvec_t<num_prim> p)
    {
        return sqrt(1.0 + gamma_beta_squared(p));
    };

    auto beta_component = [=] (dvec_t<num_prim> p, int axis)
    {
        return p[axis + 1] / lorentz_factor(p);
    };

    auto enthalpy_density = [=] (dvec_t<num_prim> p)
    {
        auto rho = p[index_density];
        auto pre = p[index_pressure];
        return rho + pre * (1.0 + 1.0 / (gamma - 1.0));
    };

    auto prim_to_cons = [=] (dvec_t<num_prim> p)
    {
        auto rho = p[index_density];
        auto pre = p[index_pressure];
        auto w = lorentz_factor(p);
        auto h = enthalpy_density(p) / rho;
        auto m = rho * w;
        auto u = dvec_t<num_cons>{};

        if constexpr (vector_components == 1)
        {
            u[0] = m;
            u[1] = m * h * p[1];
            u[2] = m * (h * w - 1.0) - pre;
        }
        if constexpr (vector_components == 2)
        {
            u[0] = m;
            u[1] = m * h * p[1];
            u[2] = m * h * p[2];
            u[3] = m * (h * w - 1.0) - pre;
        }
        if constexpr (vector_components == 3)
        {
            u[0] = m;
            u[1] = m * h * p[1];
            u[2] = m * h * p[2];
            u[3] = m * h * p[3];
            u[4] = m * (h * w - 1.0) - pre;
        }
        return u;
    };

    auto cons_to_prim = [=] (dvec_t<num_cons> cons, double p=0.0)
    {
        auto newton_iter_max = 500;
        auto error_tolerance = 1e-12 * (cons[index_density] + cons[index_energy]);
        auto gm              = gamma;
        auto m               = cons[index_density];
        auto tau             = cons[index_energy];
        auto ss              = momentum_squared(cons);
        auto n = 0;
        auto w0 = 0.0;

        while (true)
        {
            auto et = tau + p + m;
            auto b2 = min2(ss / et / et, 1.0 - 1e-10);
            auto w2 = 1.0 / (1.0 - b2);
            auto w  = sqrt(w2);
            auto d  = m / w;
            auto de = (tau + m * (1.0 - w) + p * (1.0 - w2)) / w2;
            auto dh = d + de + p;
            auto a2 = gm * p / dh;
            auto g  = b2 * a2 - 1.0;
            auto f  = de * (gm - 1.0) - p;

            p -= f / g;
            n += 1;

            if (n == newton_iter_max)
            {
                printf("c2p failed; D=%f tau=%f\n", cons[index_density], cons[index_energy]);
                exit(1);
            }
            if (fabs(f) < error_tolerance) {
                w0 = w;
                break;
            }
        }

        auto prim = dvec_t<num_prim>{};
        prim[index_density] = m / w0;
        prim[index_pressure] = p;

        if constexpr (vector_components >= 1) prim[1] = w0 * cons[1] / (tau + m + p);
        if constexpr (vector_components >= 2) prim[2] = w0 * cons[2] / (tau + m + p);
        if constexpr (vector_components >= 3) prim[3] = w0 * cons[3] / (tau + m + p);

        return prim;
    };

    auto prim_and_cons_to_flux = [=] (dvec_t<num_prim> p, dvec_t<num_cons> u, int axis)
    {
        double pre = p[index_pressure];
        double vn = beta_component(p, axis);

        auto f = dvec_t<num_cons>{};

        if constexpr (vector_components == 1)
        {
            f[0] = vn * u[0];
            f[1] = vn * u[1] + pre * (axis == 0);
            f[2] = vn * u[2] + pre * vn;
        }
        if constexpr (vector_components == 2)
        {
            f[0] = vn * u[0];
            f[1] = vn * u[1] + pre * (axis == 0);
            f[2] = vn * u[2] + pre * (axis == 1);
            f[3] = vn * u[3] + pre * vn;
        }
        if constexpr (vector_components == 3)
        {
            f[0] = vn * u[0];
            f[1] = vn * u[1] + pre * (axis == 0);
            f[2] = vn * u[2] + pre * (axis == 1);
            f[3] = vn * u[3] + pre * (axis == 2);
            f[4] = vn * u[4] + pre * vn;
        }
        return f;
    };

    auto sound_speed_squared = [=] (dvec_t<num_prim> p)
    {
        const double pre = p[index_pressure];
        const double rho_h = enthalpy_density(p);
        return gamma * pre / rho_h;
    };

    auto outer_wavespeeds = [=] (dvec_t<num_prim> p, int axis)
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
    };

    auto riemann_hlle = [=] HD (
        dvec_t<num_prim> pl, dvec_t<num_prim> pr,
        dvec_t<num_cons> ul, dvec_t<num_cons> ur)
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

    auto initial_primitive = [] HD (double x)
    {
        if (x < 0.5)
            return vec(1.0, 0.0, 1.0);
        else
            return vec(0.1, 0.0, 0.125);
    };
    auto exec = default_executor_t();

    auto t_final = 0.2;
    auto N = 100000;
    auto dx = 1.0 / N;
    auto iv = range(N + 1);
    auto ic = range(N);
    auto xc = (ic + 0.5) * dx;
    auto dt = dx * 0.3;
    auto p = xc.map(initial_primitive).cache(exec);
    auto u = p.map(prim_to_cons).cache(exec);
    auto p2 = u.map(cons_to_prim).cache(exec);
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
            p = ic.map([cons_to_prim, p, u] (auto i)
            {
                return cons_to_prim(u[i], p[i][index_pressure]);
            }).cache(exec);

            auto fhat = iv[interior_faces].map([p, u, riemann_hlle] HD (uint i)
            {
                auto ul = u[i - 1];
                auto ur = u[i];
                auto pl = p[i - 1];
                auto pr = p[i];
                return riemann_hlle(pl, pr, ul, ur);
            }).cache_if<cache_flux>(exec);

            auto du = ic[interior_cells].map([fhat, dt, dx] HD (uint i)
            {
                auto fm = fhat[i];
                auto fp = fhat[i + 1];
                return (fp - fm) * (-dt / dx);
            });

            u = (u.at(interior_cells) + du).cache(exec);

            t += dt;
            n += 1;
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        auto delta = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        auto Mzps = N * fold * 1e-6 / delta.count();
        printf("[%04d] t=%.3lf Mzps=%.3lf\n", n, t, Mzps);
    }

    // p = u.map(cons_to_prim).cache(exec);
    // for (int i = 0; i < N; ++i)
    // {
    //     printf("%+.4f %+.4f %+.4f %+.4f\n", xc[i], p[i][0], p[i][1], p[i][2]);
    // }
    return 0;
}
