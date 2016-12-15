extern "C"
 __global__ void shiftPoint(int totalPixels, double r, double g, double b, double* shiftedPointResults, double* weights, double *imageArray, double kernelBandwidth)
{
     int pixel = blockIdx.x * blockDim.x + threadIdx.x;

     if (pixel < totalPixels) {
        double old_r = imageArray[pixel*3];
        double old_g = imageArray[pixel*3 + 1];
        double old_b = imageArray[pixel*3 + 2];

        // Euclidean distance
        double distance = 0.0;
        distance += (r - old_r) * (r - old_r);
        distance += (g - old_g) * (g - old_g);
        distance += (b - old_b) * (b - old_b);


        // Gaussian kernel
        double weight = exp(-1 * (distance) / kernelBandwidth);

        shiftedPointResults[pixel*3] = old_r * weight;
        shiftedPointResults[pixel*3+1] = old_g * weight;
        shiftedPointResults[pixel*3+2] = old_b * weight;

        weights[pixel] = weight;

     }
}

__global__ void reduceArray(double* in_data, double* out_data, unsigned int n) {
           extern __shared__ double sdata[];

                unsigned int tid = threadIdx.x;

                unsigned int blockSize = blockDim.x;



                unsigned int i = blockIdx.x*(256*2) + tid;
                                unsigned int gridSize = blockSize*2*gridDim.x;
                                sdata[tid] = 0;

                                while (i < n) { sdata[tid] += in_data[i*gridDim.y + blockIdx.y] + in_data[(i+blockSize)*gridDim.y + blockIdx.y]; i += gridSize; }
                                __syncthreads();

                                if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
                                if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
                                if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
                                if (tid < 32) {
                                        if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
                                        if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
                                        if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
                                        if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
                                        if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
                                        if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
                                }
                                if (tid == 0) out_data[(blockIdx.x)*gridDim.y+blockIdx.y] = sdata[0];
        }