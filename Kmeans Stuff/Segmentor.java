import java.awt.image.*;
import java.awt.Color;
import java.io.*;
import javax.imageio.*;
import java.util.*;
import java.net.URL;


public class Segmentor {
	
	public static class Pixel {
		public int r, g, b;
		public int x, y;
		public int clusterNumber;
		
		public Pixel(int r, int g, int b) {
			this(r, g, b, -1);
		}
		
		public Pixel(int r, int g, int b, int clusterNumber) {
			this(r, g, b, 0, 0, clusterNumber);
		}
		
		public Pixel(int r, int g, int b, int x, int y) {
			this(r, g, b, x, y, -1);
		}
		
		public Pixel(int r, int g, int b, int x, int y, int clusterNumber) {
			this.r = r;
			this.g = g;
			this.b = b;
			this.x = x;
			this.y = y;
			this.clusterNumber = clusterNumber;
		}
		
		public Pixel(Pixel other) {
			this(other.r, other.g, other.b, other.x, other.y, other.clusterNumber);
		}
		
		public String toString() {
			return "Red: " + r + "\tGreen: " + g + "\tBlue: " + b + " (" + x + ", " + y + ")";
		}
	}
	
	public static class Cluster {
		//public int clusterNumber;
		ArrayList<Pixel> pixels;
		
		public Cluster() {
			//this.clusterNumber = clusterNumber;
			pixels = new ArrayList<Pixel>();
		}
		
		public void addPixel(Pixel p) {
			pixels.add(p);
		}
		
		public void addPixel(int r, int g, int b) {
			pixels.add(new Pixel(r, g, b, -1));
		}
		
		public Pixel averagePixel() {
			int r = 0, g = 0, b = 0, x = 0, y =0;
			
			for (int i = 0; i < pixels.size(); i++) {
				r += pixels.get(i).r;
				g += pixels.get(i).g;
				b += pixels.get(i).b;
				x += pixels.get(i).x;
				y += pixels.get(i).y;
			}
			
			r /= pixels.size();
			g /= pixels.size();
			b /= pixels.size();
			x /= pixels.size();
			y /= pixels.size();
			
			return new Pixel(r, g, b, x, y);
		}
	}
	
	public static void kmeans(Pixel[] data, int k, double t, Pixel[] centroids, double distanceWeight, double colorWeight) {
		//int[] labels = new int[data.length];
		
		int h = 0, i = 0, j = 0;
		int[] counts = new int[k];
		double old_error, error = Double.MAX_VALUE;
		Pixel[] c = (centroids != null) ? centroids : new Pixel[k];
		Pixel[] c1 = new Pixel[k];
		
		for (i = 0; i < k; i++) {			
			c[i] = new Pixel(data[h]);
			h += data.length / k;
		}
		
		int iteration = 0;
		
		do {
			old_error = error;
			error = 0;
			
			for (i = 0; i < k; i++)
				counts[i] = 0;
			for (i = 0; i < k; i++)
				c1[i] = null;
			
			for (h = 0; h < data.length; h++) {
				double min_score = Double.MAX_VALUE;
				for (i = 0; i < k; i++) {
					double color_distance = 0;
					double distance = 0;
					double score = 0;
					
					color_distance += Math.pow(data[h].r - c[i].r, 2);
					color_distance += Math.pow(data[h].g - c[i].g, 2);
					color_distance += Math.pow(data[h].b - c[i].b, 2);
					
					distance += Math.pow(data[h].x - c[i].x, 2);
					distance += Math.pow(data[h].y - c[i].y, 2);
					
					// square root distances
					color_distance = Math.pow(color_distance, 0.5);
					distance = Math.pow(distance, 0.5);
					
					// normalize distances
					color_distance /= 443.4050067;	// max distance between colors
					distance /= Math.pow(data.length * 2, 0.5);
					
					score = colorWeight*color_distance + distanceWeight*distance;
					
					if (score < min_score) {
						data[h].clusterNumber = i;
						min_score = score;
					}
				}
				
				int cluster = data[h].clusterNumber;
				if (c1[cluster] == null) {
					c1[cluster] = new Pixel(data[h]);
				} else {
					c1[cluster].r += data[h].r;
					c1[cluster].g += data[h].g;
					c1[cluster].b += data[h].b;
					c1[cluster].x += data[h].x;
					c1[cluster].y += data[h].y;
				}
				counts[data[h].clusterNumber]++;
				
				error += min_score;
			}
			try {
				for (i = 0; i < k; i++) {
					if (c1[i] == null)
						c1[i] = new Pixel(0, 0, 0, 0, 0, i);
					
					c[i].r = (counts[i] > 0) ? c1[i].r / counts[i] : c1[i].r;
					c[i].g = (counts[i] > 0) ? c1[i].g / counts[i] : c1[i].g;
					c[i].b = (counts[i] > 0) ? c1[i].b / counts[i] : c1[i].b;
					c[i].x = (counts[i] > 0) ? c1[i].x / counts[i] : c1[i].x;
					c[i].y = (counts[i] > 0) ? c1[i].y / counts[i] : c1[i].y;
				}
			} catch (NullPointerException e) {
				e.printStackTrace();
			}
			
		} while (Math.abs(old_error - error) > t);
	}
	
	public static void main(String[] args) {
		Scanner scan = new Scanner(System.in);
		File[] picFiles = new File[0];
		URL imageToDownload;
		try {
			imageToDownload = new URL("http://google.com/");
		} catch (Exception e) {
			imageToDownload = null;
		}
		
		int optionChosen = -1;
		
		System.out.println("IMAGE SEGMENTER 1.0 alpha:");
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
			picFiles = new File(".").listFiles(new FilenameFilter() { 
					public boolean accept(File dir, String filename) { 
						return filename.toLowerCase().endsWith(".jpg")
							|| filename.toLowerCase().endsWith(".jpeg")
							|| filename.toLowerCase().endsWith(".png")
							|| filename.toLowerCase().endsWith(".gif")
							|| filename.toLowerCase().endsWith(".bmp"); 
					}
			    } );
			
			
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
			Pixel[] image = new Pixel[length];
			
			int[] pixel = new int[3];
			int count = 0;
			
			// Read image data
			for (int y = 0; y < bi.getHeight(); y++) {
				for (int x = 0; x < bi.getWidth(); x++) {
					pixel = bi.getRaster().getPixel(x, y, new int[4]);
					image[count++] = new Pixel(pixel[0], pixel[1], pixel[2], x, y, -1);
				}
			}
			
			for (int it = startClusters; it <= endClusters; it++) {
				long startTime = System.currentTimeMillis();
				
				kmeans(image, it, 1e-4, null, distanceWeight, colorWeight);
				
				Cluster[] clusters = new Cluster[it];
				
				for (int i = 0; i < image.length; i++) {
					if (clusters[image[i].clusterNumber] == null)
						clusters[image[i].clusterNumber] = new Cluster();
					
					clusters[image[i].clusterNumber].addPixel(image[i]);
				}
				
				Pixel[] averages = new Pixel[it];
				for (int i = 0; i < it; i++) {
					if (clusters[i] != null)
						averages[i] = clusters[i].averagePixel();
					else 
						averages[i] = new Pixel(0, 0, 0);
				}
								
				BufferedImage outputImage = new BufferedImage(bi.getWidth(), bi.getHeight(), BufferedImage.TYPE_INT_RGB);
				// Set outputImage to the averaged colors
				for (int i = 0; i < image.length; i++) {
					Pixel clusterAverage = averages[image[i].clusterNumber];
					int rgb = clusterAverage.r;
					rgb = (rgb << 8) + clusterAverage.g;
					rgb = (rgb << 8) + clusterAverage.b;
					outputImage.setRGB(i%bi.getWidth(), i/bi.getWidth(), rgb);
				}
				
				String fileName = ((optionChosen == 1) ? picFiles[number-1].getName() : urlFileName) + "/output_k" + ((it < 10) ? ("0" + it) : ("" + it)) + ".bmp";
				
				File outputFile = new File("output/" + fileName);
				ImageIO.write(outputImage, "BMP", outputFile);
				
				long runTime = System.currentTimeMillis() - startTime;
				System.out.println("Completed iteration k=" + it + " in " + ((double)runTime/1000.0) + "s");
			}
			
			System.out.println("Files saved to " + outputDirectory.getAbsolutePath() + "/");
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}