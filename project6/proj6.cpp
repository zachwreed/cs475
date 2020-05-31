// 1. Program header

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifdef WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif
#include <omp.h>

#include "./CL/cl.h"
#include "./CL/cl_platform.h"

// Define Array Size
#ifndef NUM_ELEMENTS
    #define	NUM_ELEMENTS 64*1024*1024
#endif

#ifndef LOCAL_SIZE
    #define	LOCAL_SIZE 64
#endif
 

#ifndef PART
	#define PART 1
#endif 

#define	NUM_WORK_GROUPS	NUM_ELEMENTS/LOCAL_SIZE

// Filename of OpenCL functions
// {} initializer prevents narrowing 
const char *CL_FILE_NAME = {"proj6.cl"};
const float	TOL = 0.0001f;

// Function Prototypes
void Wait(cl_command_queue);
int	LookAtTheBits(float);


int main(int argc, char *argv[]) {

	char *CL_FUNC1 = (char *)"ArrayMult";
	char *CL_FUNC2 = (char *)"ArrayMultAdd";

	FILE *fp;
	bool isSum = false;
	char *funcName; 

	// see if we can even open the opencl kernel program
	// (no point going on if we can't):
    #ifdef WIN32
        errno_t err = fopen_s(&fp, CL_FILE_NAME, "r");
        if(err != 0)
    #else
        fp = fopen(CL_FILE_NAME, "r");
        if(fp == NULL)
    #endif 
		{
            fprintf(stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME);
            return 1;
        }

	// Sets up variables for the three parts required in the project specifications
	int part = PART;
	switch (part) {
		// D[gid] = A[gid] * B[gid];
		case 1:
			funcName = CL_FUNC1;
			break;

		// D[gid] = (A[gid] * B[gid]) + C[gid];
		case 2:
			funcName = CL_FUNC2;
			break;

		// Summation{A[:]*B[:]};
		case 3:
			funcName = CL_FUNC1;
			isSum = true;
			break;

		default:
			funcName = CL_FUNC1;
			break;
	}

	// returned status from opencl calls
	cl_int status;
	
    // test against CL_SUCCESS
	// get the platform id:

	cl_platform_id platform;
	status = clGetPlatformIDs(1, &platform, NULL);
	
    if(status != CL_SUCCESS) {
		fprintf( stderr, "clGetPlatformIDs failed (2)\n" );
    }
	
	// get the device id:
	cl_device_id device;
	status = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
	if(status != CL_SUCCESS)
		fprintf( stderr, "clGetDeviceIDs failed (2)\n" );

	// 2. allocate the host memory buffers:
	float *hA = new float[ NUM_ELEMENTS ];
	float *hB = new float[ NUM_ELEMENTS ];
	float *hC = new float[ NUM_ELEMENTS ];
	float *hD = new float[ NUM_ELEMENTS ];
	float sum = 0;

	// fill the host memory buffers:
	for(int i = 0; i < NUM_ELEMENTS; i++) {
		hA[i] = hB[i] = hD[i] = (float)sqrt((double)i);
	}

	size_t dataSize = NUM_ELEMENTS * sizeof(float);

	// 3. create an opencl context:
	cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &status );
    if(status != CL_SUCCESS) {
		fprintf(stderr, "clCreateContext failed\n");
    }

	// 4. create an opencl command queue:
	cl_command_queue cmdQueue = clCreateCommandQueue(context, device, 0, &status);
	if(status != CL_SUCCESS) {
		fprintf( stderr, "clCreateCommandQueue failed\n" );
    }

	// 5. allocate the device memory buffers:
	cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY,  dataSize, NULL, &status);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clCreateBuffer failed (1)\n");
    }

	cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY,  dataSize, NULL, &status);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clCreateBuffer failed (2)\n");
    }

	cl_mem dC = clCreateBuffer( context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status );
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clCreateBuffer failed (3)\n");
    }

	// Add dD for second part
	cl_mem dD;
	if(part == 2) {
		dD = clCreateBuffer(context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status );
		if(status != CL_SUCCESS) {
			fprintf(stderr, "clCreateBuffer failed (3)\n");
		}
	}


	// 6. enqueue the 2 commands to write the data from the host buffers to the device buffers:
	status = clEnqueueWriteBuffer(cmdQueue, dA, CL_FALSE, 0, dataSize, hA, 0, NULL, NULL);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueWriteBuffer failed (1)\n");
    }

	status = clEnqueueWriteBuffer(cmdQueue, dB, CL_FALSE, 0, dataSize, hB, 0, NULL, NULL);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueWriteBuffer failed (2)\n");
    }

	// Add dD for second part
	if(part == 2) {
		status = clEnqueueWriteBuffer(cmdQueue, dD, CL_FALSE, 0, dataSize, hD, 0, NULL, NULL);
		if(status != CL_SUCCESS) {
			fprintf(stderr, "clEnqueueWriteBuffer failed (2)\n");
		}
	}


	Wait(cmdQueue);

	// 7. read the kernel code from a file:
	fseek(fp, 0, SEEK_END);
	size_t fileSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	char *clProgramText = new char[fileSize+1];	// leave room for '\0'
	size_t n = fread(clProgramText, 1, fileSize, fp);
	clProgramText[fileSize] = '\0';
	fclose(fp);
	if(n != fileSize) {
		fprintf(stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", fileSize, CL_FILE_NAME, n);
    }
	// create the text for the kernel program:

	char *strings[1];
	strings[0] = clProgramText;
	cl_program program = clCreateProgramWithSource( context, 1, (const char **)strings, NULL, &status );
	
    if(status != CL_SUCCESS) {
		fprintf(stderr, "clCreateProgramWithSource failed\n");
    }
	delete []clProgramText;

	// 8. compile and link the kernel code:
	const char *options = { "" };
	status = clBuildProgram(program, 1, &device, options, NULL, NULL);
	if(status != CL_SUCCESS) {
		size_t size;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
		cl_char *log = new cl_char[size];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log, NULL);
		fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
		delete []log;
	}
	// -------------------------------------------------------------------
	// 9. create the kernel object:
	cl_kernel kernel = clCreateKernel(program, funcName, &status);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clCreateKernel failed\n");
    }

	// 10. setup the arguments to the kernel object:
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg failed (1)\n");
    }

	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg failed (2)\n");
    }

	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg failed (3)\n");
    }

	// Add dD to kernel args
	if (part == 2) {
		status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &dD);
		if(status != CL_SUCCESS) {
			fprintf(stderr, "clSetKernelArg failed (3)\n");
    	}
	}

	// 11. enqueue the kernel object for execution:
	size_t globalWorkSize[3] = {NUM_ELEMENTS, 1, 1};
	size_t localWorkSize[3]  = {LOCAL_SIZE, 1, 1};

	Wait(cmdQueue);
	double time0 = omp_get_wtime();

	time0 = omp_get_wtime();

	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueNDRangeKernel failed: %d\n", status );

	Wait(cmdQueue);

	// 12. read the results buffer back from the device to the host:
	status = clEnqueueReadBuffer(cmdQueue, dC, CL_TRUE, 0, dataSize, hC, 0, NULL, NULL);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueReadBuffer failed\n");
	}

	// Sum results if summation case
	if (isSum) {
		for (int i = 0; i < NUM_ELEMENTS; i++) {
			sum += hC[i];
		}
	}

	double time1 = omp_get_wtime();

	// Output results
	fprintf(stdout, "%d,%d,%lf\n", LOCAL_SIZE, NUM_WORK_GROUPS, (double)NUM_ELEMENTS/(time1-time0)/1000000.);

    #ifdef WIN32
        Sleep( 2000 );
    #endif


	// 13. clean everything up:
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(dA);
	clReleaseMemObject(dB);
	clReleaseMemObject(dC);

	delete []hA;
	delete []hB;
	delete []hC;

	return 0;
}


int LookAtTheBits(float fp) {
	int *ip = (int*)&fp;
	return *ip;
}


// wait until all queued tasks have taken place:
void Wait(cl_command_queue queue) {
      cl_event wait;
      cl_int status;

      status = clEnqueueMarker(queue, &wait);
      if(status != CL_SUCCESS) {
	      fprintf(stderr, "Wait: clEnqueueMarker failed\n");
      }

      status = clWaitForEvents(1, &wait);
      if(status != CL_SUCCESS) {
	      fprintf(stderr, "Wait: clWaitForEvents failed\n");
      }
}