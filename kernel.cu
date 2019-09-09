#include <iostream>
#include <cmath>
#include <cstdlib>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void generate_gaussian_kernel(float* kernel);
__global__ void gaussian_blur(unsigned char* dev_img, unsigned char* dev_processed_img, int x_dim, int y_dim, float* kernel, int kernel_dim);

int main()
{
	cudaError_t cuda_error = cudaSuccess;

	// Load input file
	char* filename = "forest_background.bmp";
	int width, height, components_per_pixel;
	unsigned char* img = stbi_load(filename, &width, &height, &components_per_pixel, 1);
	if (!img) std::cerr << "Image load failed" << std::endl;
	else std::cout 
		<< "Image dimensions: " << width << " x " << height << std::endl		
		<< "Components per pixel: " << components_per_pixel << std::endl;
	size_t img_size = width * height * sizeof(unsigned char);

	int kernel_width = 3;
	float kernel[] =
	{
		1.0f, 1.0f, 1.0f,
		1.0f, 2.0f, 1.0f,
		1.0f, 1.0f, 1.0f
	};
	//generate_gaussian_kernel(kernel);
	size_t kernel_size = kernel_width * kernel_width * sizeof(float);

	// Allocate device input image
	unsigned char* dev_img = NULL;
	cuda_error = cudaMalloc((void**)& dev_img, img_size);
	cuda_error = cudaMemcpy(dev_img, img, img_size, cudaMemcpyHostToDevice);

	// Allocate device Gassian kernel
	float* dev_kernel = NULL;
	cuda_error = cudaMalloc((void**)& dev_kernel, kernel_size);
	cuda_error = cudaMemcpy(dev_kernel, kernel, sizeof(kernel), cudaMemcpyHostToDevice);
		
	// Allocate device output image
	unsigned char* dev_processed_img = NULL;
	cuda_error = cudaMalloc((void**)& dev_processed_img, img_size);
	cuda_error = cudaMemset(dev_processed_img, 0, img_size);	

	// Allocate host output image
	unsigned char* processed_img = (unsigned char*)malloc(img_size);
	
	dim3 blocks(256, 256);
	gaussian_blur <<<blocks, 1>>> (dev_img, dev_processed_img, width, height, dev_kernel, kernel_width * kernel_width);

	cuda_error = cudaMemcpy(processed_img, dev_processed_img, img_size, cudaMemcpyDeviceToHost);
	if (cuda_error != cudaSuccess)
		std::cerr << "Failed to copy image to device:" << std::endl << cudaGetErrorString(cuda_error) << std::endl;
		
	int img_write_succeeded = stbi_write_bmp("output.bmp", width, height, 1, processed_img);
	if (!img_write_succeeded) std::cerr << "Image write failed" << std::endl;

	free(img);
	free(kernel);
	cudaFree(dev_img);
	cudaFree(dev_kernel);
	cudaFree(dev_processed_img);	
}

void generate_gaussian_kernel(float* kernel)
{
	/*int width = radius * 2 + 1;
	const float PI = 3.14f;

	float* gaussian_kernel = (float*)malloc(width * width * sizeof(float));

	for (int x = 0; x < width; ++x)
	{
		for (int y = 0; y < width; ++y)
		{
			gaussian_kernel[y * width + x] =
				(1 / (2 * PI * sigma * sigma)) * exp(-1 * (x * x + y * y) / (2 * sigma * sigma));			
		}		
	}*/

	kernel[0] = 0.04f;
	kernel[1] = 0.16f;
	kernel[2] = 0.23f;	
	kernel[3] = 0.16f;
	kernel[4] = 0.04f;
}

__global__ void gaussian_blur(unsigned char* dev_img, unsigned char* dev_processed_img, int width, int height, float* kernel, int kernel_dim)
{
	int x = blockIdx.x;
	int y = blockIdx.y;
	int thread_index = x + width * y;

	if (thread_index > width * height) return;

	//unsigned char horizontal_sum = 0;
	//unsigned char vertical_sum = 0;
		
	for (int i = 0; i < kernel_dim; i++)
	{
		for (int j = 0; j < kernel_dim; j++)
		{
			int conv_x = x - (kernel_dim / 2) + i;
			int conv_y = y - (kernel_dim / 2) + j;
			int conv_index = conv_x + width * conv_x;
			if (conv_index >= 0 && conv_index <= width * height)
				dev_processed_img[thread_index] += dev_img[conv_index] * kernel[i + kernel_dim * j];
			else
				dev_processed_img[thread_index] += 255;
		}
	}

	/*for (int i = 0; i < kernel_dim; i++)
	{
		int conv_index = thread_index - (kernel_dim / 2) + i;

		if (conv_index < x_dim * y + 1 || conv_index > x_dim * (y + 1))
			horizontal_sum += 0;
		else
			horizontal_sum += dev_img[conv_index] * kernel[i];
	}

	for (int i = 0; i < kernel_dim; i++)
	{
		int conv_index = thread_index - (kernel_dim / 2) + x_dim * i;

		if (conv_index < 0 || conv_index > x_dim * y_dim)
			vertical_sum += 0;
		else
			vertical_sum += dev_img[conv_index] * kernel[i];
	}

	dev_processed_img[thread_index] += horizontal_sum + vertical_sum;*/
}
