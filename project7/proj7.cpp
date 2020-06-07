#include <stdio.h>
#include <stdlib.h>     // exit
#include <omp.h>
#include <xmmintrin.h>
#include "./CL/cl.h"
#include "./CL/cl_platform.h"

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
#ifndef NUMT
        #define NUMT 16
#endif	

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

void SIMDimdAutoCorrelate(float *Sums, float A[], int Size) {
    for(int shift = 0; shift < Size; shift++) {
        Sums[shift] = SimdMulSum( &A[0], &A[0+shift], Size );
    }
}

void OpenMPAutoCorrelate(float *Sums, float A[], int Size) {
    omp_set_num_threads( NUMT );
    #pragma omp parallel for default(none) shared(Sums, A, Size)
    for(int shift = 0; shift < Size; shift++) {
	    float sum = 0.;
	    for(int i = 0; i < Size; i++) {
	        sum += A[i] * A[i + shift];
	    }
	    Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
    }
}

void OpenCLAutoCorrelate(float *hSums, float *hA, int Size) {

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
        }

    // Get platform ID
	status = clGetPlatformIDs(1, &platform, NULL);
    if(status != CL_SUCCESS) {
		fprintf(stderr, "clGetPlatformIDs failed (2)\n");
        return;
    }

    // Get device ID
    status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clGetDeviceIDs failed (2)\n");
        return;
    }


    // Create an opencl context:
	cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &status );
    if(status != CL_SUCCESS) {
		fprintf(stderr, "clCreateContext failed\n");
        return;
    }

	// Create an opencl command queue:
	cl_command_queue cmdQueue = clCreateCommandQueue(context, device, 0, &status);
	if(status != CL_SUCCESS) {
		fprintf( stderr, "clCreateCommandQueue failed\n" );
        return;
    }

    // Make 2 Device Arguments
    cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY, 2*Size*sizeof(cl_float), NULL, &status);

    cl_mem dSums = clCreateBuffer(context, CL_MEM_WRITE_ONLY, 1*Size*sizeof(cl_float), NULL, &status);

    // Enqueue the 2 commands to write the data from the host buffers to the device buffers:
	status = clEnqueueWriteBuffer(cmdQueue, dA, CL_FALSE, 0, 2*Size*sizeof(cl_float), hA, 0, NULL, NULL);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueWriteBuffer failed (1)\n");
        return;
    }

	status = clEnqueueWriteBuffer(cmdQueue, dSums, CL_FALSE, 0, 1*Size*sizeof(cl_float), hSums, 0, NULL, NULL);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueWriteBuffer failed (2)\n");
        return;
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
        return;
    }

    // create the text for the kernel program:
	char *strings[1];
	strings[0] = clProgramText;
	cl_program program = clCreateProgramWithSource( context, 1, (const char **)strings, NULL, &status);
	
    if(status != CL_SUCCESS) {
		fprintf(stderr, "clCreateProgramWithSource failed\n");
        return;
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
        return;
	}

    // Create Kernel
    cl_kernel kernel = clCreateKernel(program, CL_FILE_FUNC, &status);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clCreateKernel failed\n");
        return;
    }

    // Make Kernel Arguments
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA);
    if(status != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg failed (1)\n");
        return;
    }

    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &dSums);
    if(status != CL_SUCCESS) {
		fprintf(stderr, "clSetKernelArg failed (1)\n");
        return;
    }


    // Make global and local work sizes
    size_t globalWorkSize[3] = {Size, 1, 1};
    size_t localWorkSize[3] = {LOCAL_SIZE, 1, 1};

    Wait(cmdQueue);

    // Call AutoCorrelate
	status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
	if(status != CL_SUCCESS) {
		fprintf(stderr, "clEnqueueNDRangeKernel failed: %d\n", status);
        return;
    }

    Wait(cmdQueue);


    // Read hSum back from dSum
    status = clEnqueueReadBuffer(cmdQueue, dSums, CL_TRUE, 0, 1*Size*sizeof(cl_float), hSums, 0, NULL, NULL);
	if(status != CL_SUCCESS) {
			fprintf(stderr, "clEnqueueReadBuffer failed\n");
            return;
	}

    Wait(cmdQueue);


    // Clean everything up:
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(dA);
	clReleaseMemObject(dSums);

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
    switch(func) {
        // Non-Parallel Implementation
        case 0:
            AutoCorrelate(Sums, A, Size);
        break;

        // OpenCL Implementation
        case 1:
            OpenCLAutoCorrelate(Sums, A, Size);
        break;

        // SIMD Implementation
        case 2:
            SIMDimdAutoCorrelate(Sums, A, Size);
        break;
    }

    for(int i = 0; i < Size; i++) {
        fprintf(stdout, "Sum [%d] = %lf'\n", i, Sums[i]);
    }

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