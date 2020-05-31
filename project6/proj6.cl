kernel void ArrayMult(global const float *dA, global const float *dB, global float *dC) {
	// Tells you where you are in the overall 1D dataset 
	int gid = get_global_id(0);

	dC[gid] = dA[gid] * dB[gid];
}

kernel void ArrayMultAdd(global const float *dA, global const float *dB, global float *dC, global float *dD) {
	// Tells you where you are in the overall 1D dataset 
	int gid = get_global_id(0);

	dC[gid] = (dA[gid] * dB[gid]) + dD[gid];
}
