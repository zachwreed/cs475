/************************************************
 * Author: Zachary Reed
 * Description: Project 6 Source Code
 * Date: 6/8/2020
 * References:
 * 1.) Author: Mike Baily
 *     Title: OpenMP: Autocorrelation using CPU OpenMP, CPU SIMD, and GPU {OpenCL or CUDA}
 *     Description: Source code example for Project 7b
 *     Date Accessed: 6/7/2020
 * 2.) Author: Mike Baily
 *     Title: OpenMP: OpenCL Array Multiply, Multiply-Add, and Multiply-Reduce
 *     Description: Source code example for Project 6
 *     Date Accessed: 6/7/2020
 * 3.) Author: Mike Baily
 *     Title: Numeric Integration with OpenMP Reduction
 *     Description: Source code example for Project 2
 *     Date Accessed: 6/7/2020
 ************************************************/

#include <stdio.h>
#include <stdlib.h>     // exit
#include <omp.h>
#include <xmmintrin.h>
#include "./CL/cl.h"
#include "./CL/cl_platform.h"
#include <time.h>

#ifdef WIN32
    #include <windows.h>
#else
    #include <unistd.h>
#endif

const char *CL_FILE_NAME = {"proj7.cl"};
const char *CL_FILE_FUNC = {"AutoCorrelate"};

// SSE width of 4 for SIMD
#define SSE_WIDTH	4

// Define Num Threads for OpenMP Implementation
#ifndef LOCAL_SIZE
        #define LOCAL_SIZE 64
#endif

#ifndef FUNC_TYPE
        #define FUNC_TYPE 1
#endif


void Wait(cl_command_queue);


float SimdMulSum(float *a, float *b, int len){
	float sum[4] = {0., 0., 0., 0.};
	int limit = (len/SSE_WIDTH) * SSE_WIDTH;
	register float *pa = a;
	register float *pb = b;

	register __m128 ss = _mm_loadu_ps( &sum[0] );
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

double SIMDimdAutoCorrelate(float *Sums, float A[], int Size) {

    double time0 = omp_get_wtime();

    // Perform AutoCorrelate
    for(int shift = 0; shift < Size; shift++) {
        Sums[shift] = SimdMulSum(&A[0], &A[0+shift], Size);
    }

    double time1 = omp_get_wtime();

    return time1 - time0;
}

double OpenMPAutoCorrelate(float *Sums, float A[], int Size, int numT) {
    
    #ifndef _OPENMP
        fprintf(stderr, "No OpenMP support!\n" );
        return -1.;
    #endif
    
    omp_set_num_threads(numT);

    double time0 = omp_get_wtime();

    #pragma omp parallel for default(none) shared(Sums, A, Size)
    for(int shift = 0; shift < Size; shift++) {
	    float sum = 0.;
	    for(int i = 0; i < Size; i++) {
	        sum += A[i] * A[i + shift];
	    }
	    Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
    }

    double time1 = omp_get_wtime();

    return time1 - time0;
}

double OpenCLAutoCorrelate(float *hSums, float *hA, int Size) {

    FILE *fp;
    cl_int status;
    cl_platform_id platform;
	cl_device_id device;

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
            return -1.;
        }

    // Get platform ID
	status = clGetPlatformIDs(1, &platform, NULL);
    if(status != CL_SUCCESS) {
		fprintf(stderr, "clGetPlatformIDs failed (2)\n");
        return -1.;
    }

    // Get device ID
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clGetDeviceIDs failed (2)\n");
        return -1.;
    }


    // Create an opencl context:
	cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &status );
    if(status != CL_SUCCESS) {
		fprintf(stderr, "clCreateContext failed\n");
        return -1.;
    }

	// Create an opencl command queue:
	cl_command_queue cmdQueue = clCreateCommandQueue(context, device, 0, &status);
	if(status != CL_SUCCESS) {
		fprintf( stderr, "clCreateCommandQueue failed\n" );
        return -1.;
    }

    // Make 2 Device Arguments
    cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY, 2*Size*sizeof(cl_float), NULL, &status);

    cl_mem dSums = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 1*Size*sizeof(cl_float), NULL, &status);

    // Enqueue the 2 commands to write the data from the host buffers to the device buffers:
	status = clEnqueueWriteBuffer(cmdQueue, dA, CL_FALSE, 0, 2*Size*sizeof(cl_float), hA, 0, NULL, NULL);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueWriteBuffer failed (1)\n");
        return -1.;
    }

	status = clEnqueueWriteBuffer(cmdQueue, dSums, CL_FALSE, 0, 1*Size*sizeof(cl_float), hSums, 0, NULL, NULL);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueWriteBuffer failed (2)\n");
        return -1.;
    }
    
    Wait(cmdQueue);

    // Read Kernel Code from proj7.cl file
	fseek(fp, 0, SEEK_END);
	size_t fileSize = ftell(fp);
	fseek(fp, 0, SEEK_SET);

    char *clProgramText = new char[fileSize+1];	// leave room for '\0'
	size_t n = fread(clProgramText, 1, fileSize, fp);
	clProgramText[fileSize] = '\0';
	fclose(fp);
	if(n != fileSize) {
		fprintf(stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", fileSize, CL_FILE_NAME, n);
        return -1.;
    }

    // create the text for the kernel program:
	char *strings[1];
	strings[0] = clProgramText;
	cl_program program = clCreateProgramWithSource( context, 1, (const char **)strings, NULL, &status);
	
    if(status != CL_SUCCESS) {
		fprintf(stderr, "clCreateProgramWithSource failed\n");
        return -1.;
    }
	delete []clProgramText;

    // Compile and link the kernel code:
    const char *options = { "" };
	status = clBuildProgram(program, 1, &device, options, NULL, NULL);
	if(status != CL_SUCCESS) {
		size_t size;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
		cl_char *log = new cl_char[size];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, size, log, NULL);
		
        fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
		delete []log;
        return -1.;
	}

    // Create Kernel
    cl_kernel kernel = clCreateKernel(program, CL_FILE_FUNC, &status);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clCreateKernel failed\n");
        return -1.;
    }

    // Make Kernel Arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
    if(status != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg failed (1)\n");
        return -1.;
    }

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dSums);
    if(status != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg failed (1)\n");
        return -1.;
    }


    // Make global and local work sizes
    size_t globalWorkSize[3] = {Size, 1, 1};
    size_t localWorkSize[3] = {LOCAL_SIZE, 1, 1};

    Wait(cmdQueue);

    // Start Timer
    double time0 = omp_get_wtime();

    // Call AutoCorrelate
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueNDRangeKernel failed: %d\n", status);
        return -1.;
    }
    Wait(cmdQueue);

    // End Timer
	double time1 = omp_get_wtime();

    // Read hSum back from dSum
    status = clEnqueueReadBuffer(cmdQueue, dSums, CL_TRUE, 0, 1*Size*sizeof(cl_float), hSums, 0, NULL, NULL);
	if(status != CL_SUCCESS) {
			fprintf(stderr, "clEnqueueReadBuffer failed\n");
        return -1.;
	}

    Wait(cmdQueue);


    // Clean everything up:
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(dA);
	clReleaseMemObject(dSums);

    return time1 - time0;
}   

void AutoCorrelate(float *Sums, float *A, int Size) {
    for(int shift = 0; shift < Size; shift++) {
        float sum = 0.;
        for(int i = 0; i < Size; i++) {
            sum += A[i] * A[i + shift];
        }
        Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
    }
}

int main(int argc, char *argv[]) {
    FILE *fp = fopen( "signal.txt", "r" );
    
    if(fp == NULL) {
        fprintf(stderr, "Cannot open file 'signal.txt'\n");
        exit(1);
    }

    int Size;
    fscanf(fp, "%d", &Size);
    float *A = new float[ 2*Size ];
    float *Sums = new float[ 1*Size ];
    
    for(int i = 0; i < Size; i++) {
        fscanf(fp, "%f", &A[i]);
        A[i+Size] = A[i];		// duplicate the array
    }
    fclose(fp);


    int func = FUNC_TYPE;
    double time;

    // Switch on function type
    switch(func) {
        // Non-Parallel OpenMP 1 Thread Implementation
        case 0:
            time = OpenMPAutoCorrelate(Sums, A, Size, 16);
            fprintf(stdout, "OpenMP 16 Thread\n");
        break;

        // Non-Parallel OpenMP 1 Thread Implementation
        case 1:
            time = OpenMPAutoCorrelate(Sums, A, Size, 1);
            fprintf(stdout, "OpenMP 1 Thread\n");
        break;

        // OpenCL Implementation
        case 2:
            time = OpenCLAutoCorrelate(Sums, A, Size);
            fprintf(stdout, "OpenCL 64 Local Size\n");
        break;

        // SIMD Implementation
        case 3:
            time = SIMDimdAutoCorrelate(Sums, A, Size);
            fprintf(stdout, "SIMD SSE Width 4\n");

        break;
    }

    // If error
    if (time < 0.) {
        return 1;
    }
 
    // Print results MegaMultSums/Sec
    fprintf(stdout, "%lf\n", (double)Size*(double)Size/(time)/1000000.);

    // if (func == 1) {
    //     fprintf(stdout, "Sums [1-512]\n");
    //     for(int i = 1; i < 512; i++) {
    //         fprintf(stdout, "%lf\n", Sums[i]);
    //     }
    // }

    delete []A;
    delete []Sums;

    return 0;
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