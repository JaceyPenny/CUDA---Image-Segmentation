/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <time.h>
#include <float.h>
#include <string>
#include <math.h>
#include <fstream>
#include <iostream>
#include "support.h"
#include "kernel.cu"

int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;
    time_t t;
	
    int width = 0, height = 0;
	
	std::ifstream in("640x480_ladybug.txt");
	std::streambuf *cinbuf = std::cin.rdbuf(); //save old buf
	std::cin.rdbuf(in.rdbuf());
	
	cin >> width >> height;
	
	int* image = new int[width*height*5];
	
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			cin >> image[(y*width + x)*5] >> image[(y*width + x)*5 + 1] >> image[(y*width + x)*5 + 2];
			image[(y*width + x)*5 + 3] = x;
			image[(y*width + x)*5 + 4] = y;
		}
	}
	
	int minClusters = 2, maxClusters = 20;
	
	double tolerance = 1e-4;
	
	int imageSizeInBytes = width*height*5*sizeof(int);
			
	int* deviceImage;
	cudaMalloc(&deviceImage, imageSizeInBytes);
	cudaMemcpy(deviceImage, image, imageSizeInBytes, cudaMemcpyHostToDevice);
	
	int blockSizeX = 256;
	int gridSizeX = (int) ceil((double)(width*height) / blockSizeX);
	
	for (int clusters = minClusters; clusters <= maxClusters; clusters++) {
		int* clusterArray = new int[width*height];
		int* c = new int[clusters*5];
		
		int h = 0;
        for(int i = 0; i < k; i++) {
            c[i*5] = image[h*5];
            c[i*5+1] = image[h*5+1];
            c[i*5+2] = image[h*5+2];
            c[i*5+3] = image[h*5+3];
            c[i*5+4] = image[h*5+4];
            h += width*height / clusters;
        }
		
		double* error = (double*) malloc(width*height*sizeof(double));
		
		int* deviceClusters;
		cudaMalloc(&deviceClusters, width*height*sizeof(int));
		
		int* deviceCentroids;
		cudaMalloc(&deviceCentroids, clusters*5*sizeof(int));
		
		double* errorDevice;
		cudaMalloc(&errorDevice, sizeof(double)*width*height);
		
		int[] c1 = new int[clusters*5];
		cudaMemcpy(deviceCentroids, c, clusters*5*sizeof(int), cudaMemcpyHostToDevice);
		
		int[] counts = new int[clusters];
		
		double old_error, error = DBL_MAX;
		
		int l = 0;
		
		do {
			l++;
			old_error = error;
			error = 0;
			
			for (int i = 0; i < clusters; i++) {
				counts[i] = 0;
				c1[i*5] = 0;
				c1[i*5+1] = 0;
				c1[i*5+2] = 0;
				c1[i*5+3] = 0;
				c1[i*5+4] = 0;
			}
			
			cudaMemcpy(deviceCentroids, c, clusters*5*sizeof(int), cudaMemcpyHostToDevice);
			
			kmeans<<<gridSizeX, blockSizeX>>>(width*height, clusters, errorDevice, deviceImage, deviceCentroids, deviceCentroids);
			
			cudaMemcpy(clusterArray, deviceClusters, sizeof(int)*width*height, cudaMemcpyDeviceToHost);
			
			for (int i = 0; i < width*height; i++) {
				int cluster = clusterArray[i];
				counts[cluster]++;
				c1[cluster*5] += image[i*5];
				c1[cluster*5+1] += image[i*5+1];
				c1[cluster*5+2] += image[i*5+2];
				c1[cluster*5+3] += image[i*5+3];
				c1[cluster*5+4] += image[i*5+4];
			}
			
			for (int i = 0; i < clusters; i++) {
				if (counts[i] > 0) {
					c[i*5] = c1[i*5] / counts[i];
                    c[i*5+1] = c1[i*5+1] / counts[i];
                    c[i*5+2] = c1[i*5+2] / counts[i];
                    c[i*5+3] = c1[i*5+3] / counts[i];
                    c[i*5+4] = c1[i*5+4] / counts[i];
				} else {
					c[i*5] = c1[i*5];
                    c[i*5+1] = c1[i*5+1];
                    c[i*5+2] = c1[i*5+2];
                    c[i*5+3] = c1[i*5+3];
                    c[i*5+4] = c1[i*5+4];
				}
			}
			
			double* errors = new double[width*height];
			cudaMemcpy(errors, errorDevice, sizeof(double)*width*height, cudaMemcpyDeviceToHost);
			for (int i = 0; i < width*height; i++) {
				error += errors[i];
			}
			
		} while (fabs(old_error-error) > tolerance);
		
		
		stopTime(&timer);
		float elapsedTime = elapsedTime(timer);
		cout << "Time for " << clusters << ": " << int(elapsedTime*1000) << "ms     Iterations: " << iterations << endl;
		iterations = 0;
	}

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

	free(image);
	cudaFree(deviceImage);



    return 0;

}

