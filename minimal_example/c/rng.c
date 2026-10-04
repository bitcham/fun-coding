#include "rng.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void rng_seed(rng_t *r, uint64_t seed) {
    r->s = seed ? seed : 0x9e3779b97f4a7c15ULL;
}

uint64_t rng_next(rng_t *r) {
    uint64_t z = (r->s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

double rng_unif(rng_t *r) {
    /* 53-bit precision in [0, 1) */
    return (double)(rng_next(r) >> 11) * (1.0 / (double)(1ULL << 53));
}

double rng_normal(rng_t *r, double mu, double sigma) {
    double u1 = rng_unif(r);
    double u2 = rng_unif(r);
    if (u1 < 1e-300) u1 = 1e-300;
    double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return mu + sigma * z;
}

int rng_choice(rng_t *r, const double *p, int n) {
    double u = rng_unif(r);
    double cum = 0.0;
    for (int i = 0; i < n; i++) {
        cum += p[i];
        if (u <= cum) return i;
    }
    return n - 1;
}
