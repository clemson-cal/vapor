"""
Solves the 1d Euler equations using a first-order Godunov scheme

When run with Jax, this code performs at roughly 20 Mzps on a single-vapor CPU,
and about 2.6 Gzps on an A100. See the sister code euler1d.cpp for the C++
implementation. That code is typically 6x faster than Jax on the CPU and 4x
faster on the GPU.
"""
from time import perf_counter_ns

jax = True

if jax:
    from jax import jit, numpy as np

else:
    import numpy as np

    def jit(f):
        return f


gamma: float = 5.0 / 3.0


@jit
def cons_to_prim(u):
    rho = u[0]
    mom = u[1]
    nrg = u[2]
    p0 = rho
    p1 = mom / rho
    p2 = (nrg - 0.5 * mom**2 / rho) * (gamma - 1.0)
    return np.stack([p0, p1, p2], axis=0)


@jit
def prim_to_cons(p):
    rho = p[0]
    vel = p[1]
    pre = p[2]
    u0 = rho
    u1 = rho * vel
    u2 = 0.5 * rho * vel**2 + pre / (gamma - 1.0)
    return np.stack([u0, u1, u2], axis=0)


@jit
def prim_and_cons_to_flux(p, u):
    vel = p[1]
    pre = p[2]
    nrg = u[2]
    f0 = vel * u[0]
    f1 = vel * u[1] + pre
    f2 = vel * (nrg + pre)
    return np.stack([f0, f1, f2], axis=0)


@jit
def sound_speed_squared(p):
    rho = p[0]
    pre = p[2]
    return gamma * pre / rho


@jit
def riemann_hlle(ul, ur):
    pl = cons_to_prim(ul)
    pr = cons_to_prim(ur)
    fl = prim_and_cons_to_flux(pl, ul)
    fr = prim_and_cons_to_flux(pr, ur)
    csl = np.sqrt(sound_speed_squared(pl))
    csr = np.sqrt(sound_speed_squared(pr))
    al0 = pl[1] - csl
    al1 = pl[1] + csl
    ar0 = pr[1] - csr
    ar1 = pr[1] + csr
    am = np.minimum(al0, ar0).clip(None, 0.0)
    ap = np.maximum(al1, ar1).clip(0.0, None)
    return (fl * ap - fr * am - (ul - ur) * ap * am) / (ap - am)


@jit
def initial_primitive(x):
    l = np.array([1.0, 0.0, 1.000])
    r = np.array([0.1, 0.0, 0.125])
    return ((x < 0.5)[:, None] * l + (x > 0.5)[:, None] * r).T


def main():
    N = 100000

    xv = np.linspace(0.0, 1.0, N + 1)
    xc = 0.5 * (xv[1:] + xv[:-1])
    dx = (xv[-1] - xv[0]) / N
    dt = dx * 0.3

    u = prim_to_cons(initial_primitive(xc))
    n = 0
    t = 0.0
    t_final = 0.01
    plot = True
    fold = 20

    while t < t_final:
        t0 = perf_counter_ns()
        for i in range(fold):
            fhat = riemann_hlle(u[:, :-1], u[:, 1:])
            du = np.diff(fhat, axis=1) * (-dt / dx)

            if jax:
                u = u.at[:, 1:-1].add(du)
            else:
                u[:, 1:-1] += du

            t += dt
            n += 1
        t1 = perf_counter_ns()
        print(f"[{n:04d}] t={t:.3f} Mzps={1e3 * N * fold / (t1 - t0):.2f}")

    if plot:
        from matplotlib import pyplot as plt

        p = cons_to_prim(u)
        plt.plot(xc, p[0, :])
        plt.show()


main()
