/************************************************
 * Author: Zachary Reed
 * Description: Project 7 Source Code
 * Date: 6/8/2020
 * References:
 * 1.) Author: Mike Baily
 *     Title: Autocorrelation using CPU OpenMP, CPU SIMD, and GPU {OpenCL or CUDA}
 *     Description: Source code example for Project 7
 *     Date Accessed: 6/6/2020
 ************************************************/

kernel void AutoCorrelate(global const float *dA, global float *dSums) {
    int Size = get_global_size( 0 );	// the dA size is actually x 2
    int gid  = get_global_id( 0 );
    int shift = gid;

    float sum = 0.;
    for(int i = 0; i < Size; i++) {
        sum += dA[i] * dA[i + shift];
    }
    dSums[shift] = sum;
}