import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.*;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.Arrays;
import java.util.Scanner;

import static jcuda.driver.JCudaDriver.*;

public class CUDA
{
    public static void main(String[] args)
    {
        Scanner scan = new Scanner(System.in);
        File[] picFiles = new File[0];
        URL imageToDownload;
        try {
            imageToDownload = new URL("http://google.com/");
        } catch (Exception e) {
            imageToDownload = null;
        }

        int optionChosen = -1;

        System.out.println("IMAGE SEGMENTER 1.0C alpha:");
        System.out.println("Choose image from: \n"
                + "(1) Folder\n(2) URL\n(0) Exit");

        while (optionChosen < 0 || optionChosen > 2) {
            System.out.print("Enter the number you want to segment: ");
            String input = scan.nextLine();
            try {
                optionChosen = Integer.parseInt(input);
            } catch (NumberFormatException e) {}
        }

        int number = -1;

        if (optionChosen == 0) {
            System.out.println("Exiting program...");
            return;
        } else if (optionChosen == 1) {
            picFiles = new File(".").listFiles((dir, filename) -> filename.toLowerCase().endsWith(".jpg")
                    || filename.toLowerCase().endsWith(".jpeg")
                    || filename.toLowerCase().endsWith(".png")
                    || filename.toLowerCase().endsWith(".gif")
                    || filename.toLowerCase().endsWith(".bmp"));


            if (picFiles.length > 0) {
                System.out.println("The following pictures were found in this directory:");
                for (int i = 0; i < picFiles.length; i++) {
                    System.out.println("(" + (i+1) + ") " + picFiles[i].getName());
                }

                System.out.println("(0) Exit");
            } else {
                System.out.println("No valid picture files found in directory.");
                return;
            }

            number = -1;
            while (number < 0 || number > picFiles.length) {
                System.out.print("Enter the number you want to segment: ");
                String input = scan.nextLine();
                try {
                    number = Integer.parseInt(input);
                } catch (NumberFormatException e) {}
            }

            if (number == 0) {
                return;
            }
        } else if (optionChosen == 2) {

            boolean valid = false;
            while (!valid) {
                System.out.print("Enter the URL of the image: ");
                try {
                    imageToDownload = new URL(scan.nextLine());
                    valid = true;
                } catch (Exception e) {
                    valid = false;
                }
            }

        }

        int startClusters = 0;
        while (startClusters < 2 || startClusters > 99) {
            System.out.print("How many clusters to start (>2, <100): ");
            String input = scan.nextLine();
            try {
                startClusters = Integer.parseInt(input);
            } catch (NumberFormatException e) {}
        }

        int endClusters = 0;
        while (endClusters < 2 || endClusters > 99) {
            System.out.print("How many clusters to end (>2, <100): ");
            String input = scan.nextLine();
            try {
                endClusters = Integer.parseInt(input);
            } catch (NumberFormatException e) {}
        }

        double distanceWeight = -1;
        while (distanceWeight < 0) {
            System.out.print("How much weight on distance: ");
            String input = scan.nextLine();
            try {
                distanceWeight = Double.parseDouble(input);
            } catch (NumberFormatException e) {}
        }

        double colorWeight = -1;
        while (colorWeight < 0) {
            System.out.print("How much weight on color: ");
            String input = scan.nextLine();
            try {
                colorWeight = Double.parseDouble(input);
            } catch (NumberFormatException e) {}
        }

        if (startClusters > endClusters) {
            int temp = endClusters;
            endClusters = startClusters;
            startClusters = temp;
        }

        try {
            long st = System.currentTimeMillis();
            System.out.print("Downloading and initializing... ");
            String urlPath = imageToDownload.getPath();
            String urlFileName = urlPath.substring( urlPath.lastIndexOf('/')+1, urlPath.length() );

            File outputDirectory = new File("output/" + ((optionChosen == 1) ? picFiles[number-1].getName() : urlFileName) + "/");
            outputDirectory.mkdirs();
            for(File file: outputDirectory.listFiles())
                file.delete();

            // Open image file
            BufferedImage bi;

            if (optionChosen == 1) {
                bi = ImageIO.read(picFiles[number-1]);
            } else {
                try {
                    bi = ImageIO.read(imageToDownload);
                } catch (Exception e) {
                    System.out.println("Invalid URL. Run the program again with a correct URL.");
                    return;
                }

            }

            if (bi == null) {
                System.out.println("ERROR: File input error.");
                return;
            }

            int length = bi.getWidth() * bi.getHeight();
            int[] image = new int[length*5];

            int[] pixel;
            int count = 0;

            // Read image data
            for (int y = 0; y < bi.getHeight(); y++) {
                for (int x = 0; x < bi.getWidth(); x++) {
                    pixel = bi.getRaster().getPixel(x, y, new int[4]);
                    image[count*5  ] = pixel[0];
                    image[count*5+1] = pixel[1];
                    image[count*5+2] = pixel[2];
                    image[count*5+3] = x;
                    image[count*5+4] = y;
                    count++;
                }
            }

            // setup cuda
            JCudaDriver.setExceptionsEnabled(true);

            // Create the PTX file by calling the NVCC
            String ptxFileName;
            try
            {
                ptxFileName = preparePtxFile("KmeansKernel.cu");
            }
            catch (IOException e)
            {
                System.out.println(e.getMessage());
                System.out.println("Exiting...");
                return;
            }

            cuInit(0);
            CUdevice device = new CUdevice();
            cuDeviceGet(device, 0);
            CUcontext context = new CUcontext();
            cuCtxCreate(context, 0, device);

            CUmodule module = new CUmodule();
            cuModuleLoad(module, ptxFileName);

            CUfunction kmeansFunction = new CUfunction();
            cuModuleGetFunction(kmeansFunction, module, "kmeans");

            // Allocated device input data and copy host input to device
            CUdeviceptr imageDevice = new CUdeviceptr();
            cuMemAlloc(imageDevice, image.length * Sizeof.INT);
            cuMemcpyHtoD(imageDevice, Pointer.to(image), image.length * Sizeof.INT);

            int blockSizeX = 256;
            int gridSizeX = (int) Math.ceil((double)(image.length / 5) / blockSizeX);

            long et = System.currentTimeMillis();
            System.out.println(((double)(et-st)/1000.0) + "s");

            for (int k = startClusters; k <= endClusters; k++) {
                long startTime = System.currentTimeMillis();

                int[] clusters = new int[length];
                int[] c = new int[k*5];
                int h = 0;
                for(int i = 0; i < k; i++) {
                    c[i*5] = image[h*5];
                    c[i*5+1] = image[h*5+1];
                    c[i*5+2] = image[h*5+2];
                    c[i*5+3] = image[h*5+3];
                    c[i*5+4] = image[h*5+4];
                    h += length / k;
                }

                //kmeans(image, it, 1e-4, null, distanceWeight, colorWeight);
                executeKmeansKernel(kmeansFunction, imageDevice, image, clusters, c, k, 1e-4, distanceWeight, colorWeight, blockSizeX, gridSizeX);


                int[] output = calculateAveragePixels(image, clusters);

                BufferedImage outputImage = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_INT_RGB);
                // Set outputImage to the averaged colors
                for (int i = 0; i < length; i++) {

                    int rgb = output[i*5];
                    rgb = (rgb << 8) + output[i*5+1];
                    rgb = (rgb << 8) + output[i*5+2];
                    outputImage.setRGB(i%bi.getWidth(), i/bi.getWidth(), rgb);
                }

                String fileName = ((optionChosen == 1) ? picFiles[number-1].getName() : urlFileName) + "/output_k" + ((k < 10) ? ("0" + k) : ("" + k)) + ".bmp";

                File outputFile = new File("output/" + fileName);
                ImageIO.write(outputImage, "BMP", outputFile);

                long runTime = System.currentTimeMillis() - startTime;
                System.out.println("Completed iteration k=" + k + " in " + ((double)runTime/1000.0) + "s");
            }

            System.out.println("Files saved to " + outputDirectory.getAbsolutePath() + "\\");

            cuMemFree(imageDevice);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void executeKmeansKernel(CUfunction kmeansFunction, CUdeviceptr imageDevice, int[] image, int[] clusters, int[] c,
                                            int k, double tolerance, double distanceWeight, double colorWeight,
                                            int blockSizeX, int gridSizeX) {


        CUdeviceptr clustersDevice = new CUdeviceptr();
        cuMemAlloc(clustersDevice, clusters.length * Sizeof.INT);

        // Alloc device output
        CUdeviceptr centroidPixels = new CUdeviceptr();
        cuMemAlloc(centroidPixels, k * 5 * Sizeof.INT);

        /*
        CUdeviceptr countsDevice = new CUdeviceptr();
        cuMemAlloc(countsDevice, k * Sizeof.INT);
        */

        CUdeviceptr errorDevice = new CUdeviceptr();
        cuMemAlloc(errorDevice, Sizeof.DOUBLE * clusters.length);

        /*
        CUdeviceptr c1Device = new CUdeviceptr();
        cuMemAlloc(c1Device, k * 5 * Sizeof.INT);
        */

        int[] c1 = new int[k*5];

        cuMemcpyHtoD(centroidPixels, Pointer.to(c), Sizeof.INT * 5 * k);

        // Set up kernel parameters: pointers to array of pointers



        // begin kMeans algorithm
        int[] counts = new int[k];
        double old_error, error = Double.MAX_VALUE;

        int l = 0;

        do {
            l++;
            old_error = error;
            error = 0;

            Arrays.fill(counts, 0);
            Arrays.fill(c1, 0);
            //cuMemcpyHtoD(c1Device, Pointer.to(c1), k * 5 * Sizeof.INT);
            cuMemcpyHtoD(centroidPixels, Pointer.to(c), k * 5 * Sizeof.INT);

            Pointer kernelParameters = Pointer.to(
                    Pointer.to(new int[] {clusters.length}),
                    Pointer.to(new int[] {k}),
                    Pointer.to(new double[] {colorWeight}),
                    Pointer.to(new double[] {distanceWeight}),
                    Pointer.to(errorDevice),
                    Pointer.to(imageDevice),
                    Pointer.to(centroidPixels),
                    Pointer.to(clustersDevice)
            );

            cuLaunchKernel(kmeansFunction,
                    gridSizeX, 1, 1,
                    blockSizeX, 1, 1,
                    0, null,
                    kernelParameters, null
            );
            cuCtxSynchronize();

            // calculate c1
            cuMemcpyDtoH(Pointer.to(clusters), clustersDevice, Sizeof.INT*clusters.length);


            for (int i = 0; i < clusters.length; i++) {
                int cluster = clusters[i];
                counts[cluster]++;
                c1[cluster*5] += image[i*5];
                c1[cluster*5+1] += image[i*5+1];
                c1[cluster*5+2] += image[i*5+2];
                c1[cluster*5+3] += image[i*5+3];
                c1[cluster*5+4] += image[i*5+4];
            }

            for (int i = 0; i < k; i++) {
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


            double[] errors = new double[clusters.length];
            cuMemcpyDtoH(Pointer.to(errors), errorDevice, Sizeof.DOUBLE*clusters.length);
            error = sumArray(errors);
            System.out.println("" + l + " iterations");

        } while (Math.abs(old_error - error) > tolerance);

        cuMemcpyDtoH(Pointer.to(clusters), clustersDevice, clusters.length * Sizeof.INT);

        cuMemFree(errorDevice);
        cuMemFree(centroidPixels);
        cuMemFree(clustersDevice);
    }

    private static int[] calculateAveragePixels(int[] image, int[] clusters) {
        int[] output = Arrays.copyOf(image, image.length);
        int clusterNumber = 0;
        int pixelsFoundForCluster;
        do {
            pixelsFoundForCluster = 0;
            int r = 0, g = 0, b = 0;
            for (int i = 0; i < clusters.length; i++) {
                if (clusters[i] == clusterNumber) {
                    pixelsFoundForCluster++;
                    r += image[i * 5];
                    g += image[i * 5 + 1];
                    b += image[i * 5 + 2];
                }
            }

            if (pixelsFoundForCluster == 0) {
                break;
            }

            r /= pixelsFoundForCluster;
            g /= pixelsFoundForCluster;
            b /= pixelsFoundForCluster;

            for (int i = 0; i < clusters.length; i++) {
                if (clusters[i] == clusterNumber) {
                    output[i * 5] = r;
                    output[i * 5 + 1] = g;
                    output[i * 5 + 2] = b;
                }
            }

            clusterNumber++;
        } while (pixelsFoundForCluster > 0);

        return output;
    }

    private static String preparePtxFile(String cuFileName) throws IOException
    {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command = "nvcc " + modelString + " -ptx "+ cuFile.getPath()+" -o " + ptxFileName;

        //System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage = new String(toByteArray(process.getErrorStream()));
        String outputMessage = new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                    "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                    "Could not create .ptx file: "+errorMessage);
        }

        //System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    private static byte[] toByteArray(InputStream inputStream) throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }

    private static double sumArray(double[] array) {
        double sum = 0.0;
        for (double d : array) {
            sum += d;
        }

        return sum;
    }
 }
