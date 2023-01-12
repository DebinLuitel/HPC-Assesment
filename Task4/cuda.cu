//2059150
//Debin Luitel

#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"

// Compile with:  nvcc cudablur.cu lodepng.cpp -o cudablur
// Execute with: ./cudablur


//This function is a __global__ kernel which performs a box blur on an image stored in device memory. It takes the image to be blurred and the width and height of the image as arguments.
//The kernel uses a 3x3 kernel to blur the image and stores the blurred image back in the device memory. 
__global__ void box_blur(unsigned char * device_image_output, unsigned char * device_image_input, unsigned int width, unsigned int height)
{
	int r = 0;
	int g = 0;
	int b = 0;
	int a = 0;
	int x, y;
	int count = 0;

	int idx = blockDim.x * blockIdx.x + threadIdx.x; //calculating the index of the pixel
	int pixel = idx * 4;

	for (x = (pixel - 4); x <= (pixel + 4); x += 4) //calculating the average of the pixel and its neighbours
	{
		if ((x > 0) && x < (height * width * 4) && ((x - 4) / (4 * width) == pixel / (4 * width))) 
		{
			for (y = (x - (4 * width)); y <=  (x + (4 * width)); y += (4 * width)) //calculating the average of the pixel and its neighbours
			{
				if (y > 0 && y < (height * width * 4)) 
				{
					r += device_image_input[y];
					g += device_image_input[1 + y];
					b += device_image_input[2 + y]; 
					count++;
				}
			}
		}
	}
	
	a = device_image_input[3 + pixel];

	device_image_output[pixel] = r / count; //storing the blurred image in the device memory
	device_image_output[1 + pixel] = g / count; 
	device_image_output[2 + pixel] = b / count;//storing the blurred image in the device memory
	device_image_output[3 + pixel] = a; 
}

// reads an image file, applies a blur effect, and writes the resulting image to a new file.
int main(int argc, char **argv)
{
	unsigned int error;//declaring error variable
	unsigned char *image;//declaring image variable
	unsigned int width;//declaring width variable
	unsigned int height;//declaring height variable
	const char *input_filename = "2059150.png";//input file name
	const char *output_filename = "2059150_blur.png";//output file name

	error = lodepng_decode32_file(&image, &width, &height, input_filename);//decoding the input file
	if (error) {
		printf("Error %u: %s\n", error, lodepng_error_text(error));//printing the error message
	}

	int array_size = width * height * 4;
	int array_bytes = array_size * sizeof(unsigned char);

	unsigned char host_image_input[array_size * 4];//declaring host memory pointers
	unsigned char host_image_output[array_size * 4];

	for (int i = 0; i < array_size; i++) { //copying the image to the host memory
		host_image_input[i] = image[i]; 
	}

	// declaring device memory pointers
	unsigned char * d_in; //declaring device memory pointers
	unsigned char * d_out;

//Allocating device memory, copying the host image input array to device memory,
// launching the kernel function, copying the processed result from device array to the host array and encoding the output to a file.

	cudaMalloc((void**) &d_in, array_bytes);
	cudaMalloc((void**) &d_out, array_bytes);

	// copying the host image input array to device memory
	cudaMemcpy(d_in, host_image_input, array_bytes, cudaMemcpyHostToDevice);

	// launching the kernel function
	box_blur<<<height, width>>>(d_out, d_in, width, height);

	// copying the processed result from device array to the host array
	cudaMemcpy(host_image_output, d_out, array_bytes, cudaMemcpyDeviceToHost);
	
	error = lodepng_encode32_file(output_filename, host_image_output, width, height);
	if (error) {
		printf("error %u: %s\n", error, lodepng_error_text(error));
	}

	printf("\n Blurred image is saved on the same directory with name 2059150_blur.png\n\n");//printing the output file name

	// deallocating the device memory
	cudaFree(d_in);//freeing the device memory
	cudaFree(d_out);//freeing the device memory

	return 0;//returning 0
}