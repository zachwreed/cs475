/************************************************
 * Author: Zachary Reed
 * Description: Project 2 Source Code
 * Date: 4/26/2020
 * References:
 * 1.) Author: Mike Baily
 *     Title: Numeric Integration with OpenMP Reduction
 *     Description: Source code example for Project 2
 *     Date Accessed: 4/26/2020
 ************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define XMIN     -1.
#define XMAX      1.
#define YMIN     -1.
#define YMAX      1.

// setting the number of threads:
#ifndef NUMT
    #define NUMT		1
#endif

// set number of nodes for grid [][]
#ifndef NUMNODES
    #define NUMNODES	10000
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
	#define NUMTRIES	100
#endif

// Set N for x^n + y^n + z^n = 1
#ifndef N
    #define N	4
#endif

// iu,iv = 0 .. NUMNODES-1
double Height(int iu, int iv) {

	double x = -1.  +  2.*(double)iu /(double)(NUMNODES-1);	// -1. to +1.
	double y = -1.  +  2.*(double)iv /(double)(NUMNODES-1);	// -1. to +1.

	double xn = pow(fabs(x), (double)N);
	double yn = pow(fabs(y), (double)N);
	double r = 1. - xn - yn;
	
    if (r < 0.) {
	    return 0.;
    }

	double height = pow(1. - xn - yn, 1./(double)N);
	return height;
}

int main(int argc, char *argv[]) {
	double fullTileArea = (((XMAX - XMIN)/(double)(NUMNODES-1)) * ((YMAX - YMIN)/(double)(NUMNODES-1)));

	double sumVolume = 0.;
	double maxPerformance = 0.;      // must be declared outside the NUMTRIES loop

    omp_set_num_threads(NUMT);	// set the number of threads to use in the for-loop:`

	// looking for the maximum performance:
    for (int t = 0; t < NUMTRIES; t++) {
		
		double volume = 0.;
		double time0 = omp_get_wtime();

		#pragma omp parallel for default(none) shared(fullTileArea) reduction(+:volume)
		for (int i = 0; i < NUMNODES*NUMNODES; i++) {
			int iu = i % NUMNODES;
			int iv = i / NUMNODES;
			double adjTileArea = fullTileArea;
			double z = Height(iu, iv);

			// If tile at corner, iu and iv at 0 or max
			if ((iu == NUMNODES-1 || iu == 0) && (iv == NUMNODES-1 || iv == 0)) {
				adjTileArea -= fullTileArea * 0.75;
			}

			// Else if tile at edge, iu or iv at 0 or max
			else if ((iu == NUMNODES-1 || iu == 0) || (iv == NUMNODES-1 || iv == 0)) {
				adjTileArea -= fullTileArea * 0.5;
			}
			volume += 2 * (adjTileArea * z);
		}
		
		double time1 = omp_get_wtime();
		double megaHeightsPerSecond = (double)NUMNODES * NUMNODES / (time1 - time0) / 1000000.;

		if (megaHeightsPerSecond > maxPerformance) {
            maxPerformance = megaHeightsPerSecond;
        }
		sumVolume += volume;
	}

	double avgVolume = sumVolume / NUMTRIES;
	printf("%d, %d, %lf, %lf\n", NUMT, NUMNODES, maxPerformance, avgVolume);

    return 0;
}