extern "C"
 __global__ void kmeans(int pixels, int k, double colorWeight, double distanceWeight,
     double* error, int *imageArray, int* centroidPixels, int *clustersArray)
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

             score = colorWeight*color_distance + distanceWeight*distance;

             if (score < min_score) {
                 bestCluster = i;
                 min_score = score;
             }
         }

         clustersArray[pixel] = bestCluster;
         error[pixel] = min_score;
     }
 }

