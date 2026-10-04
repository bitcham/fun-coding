#ifndef RNG_H
#define RNG_H

#include <stdint.h>

/* splitmix64 기반의 간단한 PRNG. 결정성을 위해 seed로 초기화. */

typedef struct {
    uint64_t s;
} rng_t;

void rng_seed(rng_t *r, uint64_t seed);
uint64_t rng_next(rng_t *r);
double rng_unif(rng_t *r);            /* [0, 1) */
double rng_normal(rng_t *r, double mu, double sigma);
int rng_choice(rng_t *r, const double *p, int n);  /* multinomial */

#endif
