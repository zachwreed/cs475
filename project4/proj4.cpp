/************************************************
 * Author: Zachary Reed
 * Description: Project 4 Vectorized Array Multiplication/Reduction using SSE
 * Date: 5/11/2020
 * References:
 * 1.) Author: Mike Baily
 *     Title: Vectorized Array Multiplication/Reduction using SSE
 *     Description: Source code example for Project 4
 *     Date Accessed: 5/11/2020
 ************************************************/

#include <xmmintrin.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#ifndef NUMTRIES
	#define NUMTRIES 10
#endif


#ifndef ARRSIZE
	#define ARRSIZE	1024 // 1 KiB
#endif

#define SSE_WIDTH	4
#define RANMAX      100.0
#define RANMIN      0.01

float a[ARRSIZE];
float b[ARRSIZE];

// Returms MulSum for SIMD calculation
float SimdMulSum(float *a, float *b, int len){
	float sum[4] = {0., 0., 0., 0.};
	int limit = (len/SSE_WIDTH) * SSE_WIDTH;
	register float *pa = a;
	register float *pb = b;

	__m128 ss = _mm_loadu_ps( &sum[0] );
	for (int i = 0; i < limit; i += SSE_WIDTH) {
		ss = _mm_add_ps( ss, _mm_mul_ps( _mm_loadu_ps( pa ), _mm_loadu_ps( pb ) ) );
		pa += SSE_WIDTH;
		pb += SSE_WIDTH;
	}
	_mm_storeu_ps( &sum[0], ss );

	for (int i = limit; i < len; i++){
		sum[0] += a[i] * b[i];
	}
	return sum[0] + sum[1] + sum[2] + sum[3];
}

// Returns MulSum for non SIMD calculation
float MulSum(float *a, float *b, int len) {
    float sum = 0.;

    for (int i = 0; i < len; i++){
		sum += a[i] * b[i];
	}

    return sum;
}

// Returns random float in range
float Ranf(unsigned int *seedp, float low, float high) {
    float r = (float) rand_r( seedp ); // 0 - RAND_MAX
    return(low+r*(high-low)/(float)RAND_MAX);
}


// main:
int main (int argc, char *argv[]) {
    unsigned int seed = 2;

    // fill the random-value arrays:
    for (int n = 0; n < ARRSIZE; n++) {       
            a[n] = Ranf(&seed, RANMIN, RANMAX);
            b[n] = Ranf(&seed, RANMIN, RANMAX);
    } 

    double maxPerformanceSIMD = 0.;
    double maxPerformanceNoSIMD = 0.;

    for (int t = 0; t < NUMTRIES; t++) {
        // Get Runtime of SIMD Array Multiplication
        double time0SIMD = omp_get_wtime();
        float sumSIMD = SimdMulSum(a, b, ARRSIZE);
        double time1SIMD = omp_get_wtime();

        // Get Runtime of No-SIMD Array Multiplication
        double time0NoSIMD = omp_get_wtime();
        float sumNoSIMD = MulSum(a, b, ARRSIZE);
        double time1NoSIMD = omp_get_wtime();

        // Calculate MM/s for SIMD
        double megaMulsPerSecSIMD = (double)ARRSIZE / (time1SIMD - time0SIMD) / 1000000.;
        if (megaMulsPerSecSIMD > maxPerformanceSIMD) {
            maxPerformanceSIMD = megaMulsPerSecSIMD;
        }

        // Calculate MM/s for No-SIMD
        double megaMulsPerSecNoSIMD = (double)ARRSIZE / (time1NoSIMD - time0NoSIMD) / 1000000.;
         if (megaMulsPerSecNoSIMD > maxPerformanceNoSIMD) {
            maxPerformanceNoSIMD = megaMulsPerSecNoSIMD;
        }
    }
    printf("%d,%lf, %lf\n", ARRSIZE, maxPerformanceSIMD, maxPerformanceNoSIMD);
    return 0;
}