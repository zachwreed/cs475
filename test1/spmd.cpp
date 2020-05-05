#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>

#define NUM 1000000
float A[NUM], B[NUM], C[NUM];


void DoWork( int me, int total ) {
    int first = NUM * me / total;
    int last = NUM * (me+1)/total - 1;
    for( int i = first; i <= last; i++ ) {
        C[ i ] = A[ i ] * B[ i ];
    }
}

// Single Program Multiple Data (SPMD)
void spmd() {
    int total = omp_get_num_threads( );

    #pragma omp parallel default(none),private(me),shared(total) {
        int me = omp_get_thread_num( );
        DoWork( me, total );
    }
}

// Trapezoid Integration Problem

// Don't do this way
void TrapDont() {
    const double A = 0.;
    const double B = M_PI;
    double dx = ( B - A ) / (float) ( numSubdivisions – 1 ); // delta x
    double sum = ( Function( A ) + Function( B ) ) / 2.; // first and last over 2
    
    omp_set_num_threads( numThreads );
    
    // Will cause issues summing into sum
    #pragma omp parallel for default(none), shared(dx,sum)
    for( int i = 1; i < numSubdivisions - 1; i++ ) {
        double x = A + dx * (float) i;
        double f = Function( x );
        sum += f;
        // Assembly at this line:
        // Load sum
        // Add f    # what if the scheduler switches here?
        // Store sum
    }
    sum *= dx;
}
// DO it this way
void TrapDo() {
    const double A = 0.;
    const double B = M_PI;
    double dx = ( B - A ) / (float) ( numSubdivisions – 1 ); // delta x
    double sum = ( Function( A ) + Function( B ) ) / 2.; // first and last over 2
    
    omp_set_num_threads( numThreads );
    // Solution 1, use reduction (fastest benchmark)
    // Will create separate sum for each thread and add together at barrier
    #pragma omp parallel for default(none), shared(dx), reduction(+:sum)
    for( int i = 1; i < numSubdivisions - 1; i++ ) {
        double x = A + dx * (float) i;
        double f = Function( x );
        sum += f;
    }

    // Solution 2, use atomic
    #pragma omp parallel for default(none), shared(dx)
    for( int i = 1; i < numSubdivisions - 1; i++ ) {
        double x = A + dx * (float) i;
        double f = Function( x );
        #pragma omp atomic // uses built-in hardware instruction
        sum += f;
    }
    
    // Solution 3, use critical
    #pragma omp parallel for default(none), shared(dx)
    for( int i = 1; i < numSubdivisions - 1; i++ ) {
        double x = A + dx * (float) i;
        double f = Function( x );
        #pragma omp critical // disables scheduler interuptions during the below section
        sum += f;
    }
    sum *= dx;
}