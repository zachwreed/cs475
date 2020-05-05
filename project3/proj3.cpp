/************************************************
 * Author: Zachary Reed
 * Description: Project 2 Source Code
 * Date: 5/4/2020
 * References:
 * 1.) Author: Mike Baily
 *     Title: Functional Decomposition
 *     Description: Source code example for Project 3
 *     Date Accessed: 5/4/2020
 ************************************************/

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#ifndef NUMTRIES
	#define NUMTRIES	100
#endif

// Basic time step will be 1 month. Necessary consts:
const float GRAIN_GROWS_PER_MONTH =	  9.0;
const float ONE_DEER_EATS_PER_MONTH = 1.0; // in inches
// PRECIP is in inches.
const float AVG_PRECIP_PER_MONTH =	7.0;	// average
const float AMP_PRECIP_PER_MONTH =	6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise
const float MIDPRECIP =				10.0;
// TEMP is in degrees Fahrenheit (°F).
const float AVG_TEMP =				60.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise
const float MIDTEMP =				40.0;


// Use variables like NextNumDeer, NextNumGrain in each function.
// State Global Variables 
int	    NowYear;		// 2020 - 2025
int	    NowMonth;		// 0 - 11
float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
int	    NowNumDeer;		// number of deer in the current population
unsigned int seed = 0;

// Mutex/Barrier Global Variables
omp_lock_t	Lock;
int		NumInThreadTeam;
int		NumAtBarrier;
int		NumGone;

// Helper Functions
// -----------------------------------------------

// Returns square root
float SQR(float x) {
    return x*x;
}

// Returns random float in range
float Ranf(unsigned int *seedp, float low, float high) {
    float r = (float) rand_r( seedp ); // 0 - RAND_MAX
    return(low+r*(high-low)/(float)RAND_MAX);
}

// Returns random int in range
int Ranf(unsigned int *seedp, int ilow, int ihigh) {
    float low = (float)ilow;
    float high = (float)ihigh + 0.9999f;
    return (int)(Ranf(seedp, low,high));
}

// Sets NowPrecip and NowTemp for next environment state
void setNextEnvironment() {
    float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );
    // Compute NowTemp
    float temp = AVG_TEMP - AMP_TEMP * cos( ang );
    NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );
    // Compute NowPrecip
    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
    NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
    if( NowPrecip < 0. ) {
        NowPrecip = 0.;
    }
}

// Section Functions
// -----------------------------------------------

void GrainDeer() {
    while(NowYear < 2026) {
        // compute next Number of Deer
        int nextNumDeer;

        // If too many deer, decrement 
        if (float(NowNumDeer) > NowHeight) {
            nextNumDeer = NowNumDeer - 1;
        }
        // If more grain than deer, increment
        else if (float(NowNumDeer) < NowHeight) {
            nextNumDeer = NowNumDeer + 1;
        }
        // Else stay the same
        else {
            nextNumDeer = NowNumDeer;
        }
        // Clamp nextMumDeer
        if (nextNumDeer < 0) {
            nextNumDeer = 0;
        }
        
        // DoneComputing barrier:
        #pragma omp barrier

        NowNumDeer = nextNumDeer;

        // DoneAssigning barrier:
        #pragma omp barrier

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

void Grain() {
    while(NowYear < 2026) {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:
        float nextHeight;

        float tempFactor = exp(-SQR((NowTemp-MIDTEMP)/10.));
        float precipFactor = exp(-SQR((NowPrecip-MIDPRECIP)/10.));

        // Configure nextHeight
        nextHeight = NowHeight;
        nextHeight += tempFactor*precipFactor*GRAIN_GROWS_PER_MONTH;
        nextHeight -= (float)NowNumDeer*ONE_DEER_EATS_PER_MONTH;

        // clamp nextHeight
        if (nextHeight < 0) {
            nextHeight = 0;
        }

        // DoneComputing barrier:
        #pragma omp barrier

        NowHeight = nextHeight;
        
        // DoneAssigning barrier:
        #pragma omp barrier

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

void Watcher() {
    while(NowYear < 2026) {
 
        // DoneComputing barrier:
        #pragma omp barrier

        // DoneAssigning barrier:
        #pragma omp barrier

        // Increment Time
        if (NowMonth >= 11) {
            NowMonth = 0;
            NowYear++;
        }
        else {
            NowMonth++;
        }

        // Set Next Environment
        setNextEnvironment();

        // Print Results NowNumDeer, NowHeight, NowTemp, NowPrecip
        printf("%d, %lf, %lf, %lf\n", NowNumDeer, NowHeight, NowTemp, NowPrecip);

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

int main() {

    // Setup initial Environment variables
    // starting date and time:
    NowMonth =    0;
    NowYear  = 2020;

    // starting state (feel free to change this if you want):
    NowNumDeer = 1;
    NowHeight =  1.;

    // Set initial environment state
    setNextEnvironment();

    printf("NowNumDeer, NowHeight, NowTemp, NowPrecip\n");
    printf("%d, %lf, %lf, %lf\n", NowNumDeer, NowHeight, NowTemp, NowPrecip);

    omp_set_num_threads(3);	// same as # of sections
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            GrainDeer();
        }

        #pragma omp section
        {
            Grain();
        }

        #pragma omp section
        {
            Watcher();
        }

        // #pragma omp section
        // {
        //     MyAgent( );	// your own
        // }
    } // implied barrier -- all functions must return in order
	// to allow any of them to get past here
}