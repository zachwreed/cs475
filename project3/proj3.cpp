/************************************************
 * Author: Zachary Reed
 * Description: Project 3 Functional Decomposition
 * Date: 5/6/2020
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
int     NowPrintMonth;
float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
int	    NowNumDeer;		// number of deer in the current population

float NowDeerDiseaseMod;    // Represents percent of population killed
float NowGrainDiseaseMod;   // Represents percent of height in grains killed

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

        // Modify if Disease is Present
        if (NowDeerDiseaseMod > .0) {
            nextNumDeer = nextNumDeer - int(nextNumDeer * NowDeerDiseaseMod);
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

        // Modify if Disease
        if (NowGrainDiseaseMod > .0) {
            nextHeight = nextHeight - (nextHeight * NowGrainDiseaseMod);
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

void Disease() {
    while(NowYear < 2026) {
        // compute next Number of Predators
        int rand = Ranf(&seed, 2, 10);
        float NextDeerDiseaseMod = 0.0;
        float NextGrainDiseaseMod = 0.0;

        // If 20% chance (5, 10) to disease deer
        if (rand % 5 == 0) {
            NextDeerDiseaseMod = 0.3;
        }

        // If 10% chance (7) to disease grain
        else if (rand % 7 == 0) {
            NextGrainDiseaseMod = 0.4;
        }
        
        // DoneComputing barrier:
        #pragma omp barrier
        NowDeerDiseaseMod = NextDeerDiseaseMod;
        NowGrainDiseaseMod = NextGrainDiseaseMod;

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
        NowPrintMonth++;

        // Set Next Environment
        setNextEnvironment();

        // Print Results NowNumDeer, NowDeerDiseaseMod in Percent, NowHeight, NowGrainDiseaseMod in Percent, NowTemp in Celsius, NowPrecip
    printf("%d, %d,%d,%lf,%d,%lf,%lf\n", NowPrintMonth, NowNumDeer, int(NowDeerDiseaseMod * 100), (NowHeight * 2.54),int(NowGrainDiseaseMod * 100),(5./9.)*(NowTemp-32), (NowPrecip * 2.54));

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

int main() {

    // Setup initial Environment variables
    // starting date and time:
    NowMonth = 0;
    NowPrintMonth = 0;
    NowYear  = 2020;

    // starting state (feel free to change this if you want):
    NowNumDeer = 1;
    NowHeight =  1.;
    NowDeerDiseaseMod = 0.;
    NowGrainDiseaseMod = 0.;

    // Set initial environment state
    setNextEnvironment();

    printf("Month,NowNumDeer,NowDeerDiseaseMod,NowHeight,NowGrainDiseaseMod,NowTemp,NowPrecip\n");
    printf("%d, %d,%d,%lf,%d,%lf,%lf\n", NowPrintMonth, NowNumDeer, int(NowDeerDiseaseMod * 100), (NowHeight * 2.54),int(NowGrainDiseaseMod * 100),(5./9.)*(NowTemp-32), (NowPrecip * 2.54));

    omp_set_num_threads(4);	// same as # of sections
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

        #pragma omp section
        {
            Disease();
        }
    } // implied barrier -- all functions must return in order
	// to allow any of them to get past here
}