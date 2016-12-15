extern "C"
 __global__ void kmeans(int pixels, int k, double* error, int *imageArray, 
					    int* centroidPixels, int *clustersArray)
 {
     int pixel = blockIdx.x * blockDim.x + threadIdx.x;

     if (pixel < pixels) {
         int r = imageArray[pixel*5];
         int g = imageArray[pixel*5 + 1];
         int b = imageArray[pixel*5 + 2];
         int x = imageArray[pixel*5 + 3];
         int y = imageArray[pixel*5 + 4];

         int bestCluster = 0;

         double min_score = 10;
         int i = 0;
         for (i = 0; i < k; i++) {
             double color_distance = 0;
             double distance = 0;
             double score = 0;

             int c_r = centroidPixels[i*5];
             int c_g = centroidPixels[i*5+1];
             int c_b = centroidPixels[i*5+2];
             int c_x = centroidPixels[i*5+3];
             int c_y = centroidPixels[i*5+4];

             color_distance += (r - c_r)*(r - c_r);
             color_distance += (g - c_g)*(g - c_g);
             color_distance += (b - c_b)*(b - c_b);
             distance += (x - c_x)*(x - c_x);
             distance += (y - c_y)*(y - c_y);

             color_distance = sqrt(color_distance);
             distance = sqrt(distance);

             color_distance /= 443.4050067;
             distance /= sqrt((float)pixels * 2);

             score = color_distance + distance;

             if (score < min_score) {
                 bestCluster = i;
                 min_score = score;
             }
         }

         clustersArray[pixel] = bestCluster;
         error[pixel] = min_score;
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

