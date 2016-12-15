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
	
	string input = "64x64_logo.txt";
	
	cout << "Test file: ";
	cin >> input;
	
	std::ifstream in(input.c_str());
	std::streambuf *cinbuf = std::cin.rdbuf(); //save old buf
	std::cin.rdbuf(in.rdbuf());
	
	cin >> width >> height;
	
	int* image = new int[width*height*3];
	
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int pos = y*width+x;
			cin >> image[pos*3] >> image[pos*3 + 1] >> image[pos*3 + 2];
		}
	}
	
	int minBandwidth = 1000, maxBandwidth = 2800;
	
	double tolerance = 1.0;
	
	int imageSizeInBytes = width*height*3*sizeof(int);
			
	int* deviceImage;
	cudaMalloc(&deviceImage, imageSizeInBytes);
	cudaMemcpy(deviceImage, image, imageSizeInBytes, cudaMemcpyHostToDevice);
	
	int blockSizeX = 256;
	int gridSizeX = (int) ceil((double)(width*height) / blockSizeX);
	
	for (int bandwidth = minBandwidth; bandwidth <= maxBandwidth; bandwidth += 200) {
		
		int n = width*height;

		double[] shiftedPoints = new double[n*3];
		bool[] stopMoving = new bool[n];
		for (int i = 0; i < n; i++) stopMoving[i] = false;
		
		double[] shiftedPointResults = new double[n*3];
		for (int i = 0; i < n*3; i++) shiftedPointResults[i] = 0.0;
		double[] weightSumsByBlock = new double[gridSizeX];
		for (int i = 0; i < gridSizeX; i++) weightSumsByBlock[i] = 0.0;
		double[] shiftedPointSumsByBlock = new double[gridSizeX * 3];
		for (int i = 0; i < gridSizeX*3; i++) shiftedPointSumsByBlock[i] = 0.0;

		for (int i = 0; i < n*3; i++) {
		    shiftedPoints[i] = image[i];
		}
		
		double* deviceShiftedPointResults;
		cudaMalloc(&deviceShiftedPointResults, n * 3 * sizeof(double));
		
		double* deviceWeightResults;
		cudaMalloc(&deviceWeightResults, n * sizeof(double));
		
		double* deviceWeightSums;
		cudaMalloc(&deviceWeightSums, gridSizeX * sizeof(double));
		
		double* deviceShiftedPointSums;
		cudaMalloc(&deviceShiftedPointSums, gridSizeX * 3 * sizeof(double));

		double maxShiftDistance;

		int iterations = 0;
		
		dim3 gridDim(gridSizeX, 3, 1);
		dim3 blockDim(blockSizeX, 1, 1);
		
		do {
			maxShiftDistance = 0.0;
			
			for (int i = 0; i < n; i++) {
				if (!stopMoving[i]) {
					double newPointR = 0.0, newPointG = 0.0, newPointB = 0.0;
					
					shiftPoint<<<gridSizeX, blockSizeX>>>(
							n, 
							shiftedPoints[i*3], 
							shiftedPoints[i*3+1], 
							shiftedPoitns[i*3+2], 
							deviceShiftedPointResults, 
							deviceWeightResults, 
							deviceImage, 
							bandwidth
					);
					
					reduceArray<<<gridSizeX, blockSizeX, blockSizeX * sizeof(double)>>>(
						deviceWeightResults,
						deviceWeightSums,
						n
					);
					
					reduceArray<<<gridDim, blockDim, blockSizeX * sizeof(dobule)>>>(
						deviceShiftedPointResults,
						deviceShiftedPointSums,
						n
					);
					
					cudaMemcpy(shiftedPointSumsByBlock, deviceShiftedPointSums, sizeof(double)*gridSizeX*3, cudaMemcpyDeviceToHost);
					cudaMemcpy(weightSumsByBlock, deviceWeightSums, sizeof(double)*gridSizeX, cudaMemcpyDeviceToHost);
					
					double totalWeight = 0.0;
					for (int j = 0; j < gridSizeX; j++) {
						totalWeight += weightSumsByBlock[j];
					}
					
					for (int j = 0; j < gridSizeX; j++) {
						newPointR += shiftedPointSumsByBlock[j*3];
						newPointG += shiftedPointSumsByBlock[j*3 + 1];
						newPointB += shiftedPointSumsByBlock[j*3 + 2];
					}
					
					newPointR /= totalWeight;
					newPointG /= totalWeight;
					newPointB /= totalWeight;
					
					double oldR = shiftedPoints[i*3], oldG = shiftedPoints[i*3+1], oldB = shiftedPoints[i*3+2];
					double shiftDistance = 0.0;
					shiftDistance += (newPointR - oldR) * (newPointR - oldR);
					shiftDistance += (newPointG - oldG) * (newPointG - oldG);
					shiftDistance += (newPointB - oldB) * (newPointB - oldB);
					shiftDistance = sqrt(shiftDistance);
					
					if (shiftDistance <= tolerance) {
						stopMoving[i] = true;
					}
					
					if (shiftDistance > maxShiftDistance) {
						maxShiftDistance = shiftDistance;
					}
					
					shiftedPoints[i*3] = newPoint[0];
					shiftedPoints[i*3+1] = newPoint[1];
					shiftedPoints[i*3+2] = newPoint[2];
					
					iterations++;
				}
			}
			
		} while (maxShiftDistance > tolerance)
		
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

