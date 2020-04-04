#include <iostream>
#include <string>
#include <cassert>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

cv::Mat imageRGBA;
cv::Mat imageGrey;

uchar4        *d_rgbaImage__;
unsigned char *d_greyImage__;

size_t numRows() { return imageRGBA.rows; }
size_t numCols() { return imageRGBA.cols; }

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}

void preProcess(uchar4 **inputImage, unsigned char **greyImage,
                uchar4 **d_rgbaImage, unsigned char **d_greyImage,
                const std::string &filename) {
  //make sure the context initializes ok
  checkCudaErrors(cudaFree(0));

  cv::Mat image;
  image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
  }

  cv::cvtColor(image, imageRGBA, CV_BGR2RGBA);

  //allocate memory for the output
  imageGrey.create(image.rows, image.cols, CV_8UC1);

  //This shouldn't ever happen given the way the images are created
  //at least based upon my limited understanding of OpenCV, but better to check
  if (!imageRGBA.isContinuous() || !imageGrey.isContinuous()) {
    std::cerr << "Images aren't continuous!! Exiting." << std::endl;
    exit(1);
  }

  *inputImage = (uchar4 *)imageRGBA.ptr<unsigned char>(0);
  *greyImage  = imageGrey.ptr<unsigned char>(0);

  const size_t numPixels = numRows() * numCols();
  //allocate memory on the device for both input and output
  checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
  checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
  checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char))); //make sure no memory is left laying around

  //copy input array to the GPU
  checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

  d_rgbaImage__ = *d_rgbaImage;
  d_greyImage__ = *d_greyImage;
}

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,unsigned char* const greyImage,int numRows, int numCols){
	int threadId = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
  	if (threadId <  numRows * numCols){
    		const unsigned char R = rgbaImage[threadId].x;
    		const unsigned char G = rgbaImage[threadId].y;
    		const unsigned char B = rgbaImage[threadId].z;
    		greyImage[threadId] = .299f * R + .587f * G + .114f * B;
  	}
}

void postProcess(const std::string& output_file, unsigned char* data_ptr) {
  	cv::Mat output(numRows(), numCols(), CV_8UC1, (void*)data_ptr);
  	//output the image
  	cv::imwrite(output_file.c_str(), output);
}

void cleanup(){
  	//cleanup
  	cudaFree(d_rgbaImage__);
  	cudaFree(d_greyImage__);
}

int main(int argc,char* argv[]){

	//load input file
	std::string input_file = argv[1];
	//define output file
	std::string output_file = argv[2];

	uchar4 *h_rgbaImage, *d_rgbaImage;
	unsigned char *h_greyImage, *d_greyImage;

	//load the image and give us our input and output pointers
  	preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

	int thread = 16;
  	int grid = (numRows()*numCols() + thread - 1)/ (thread * thread);
  	const dim3 blockSize(thread, thread);
  	const dim3 gridSize(grid);
  	rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows(), numCols());


	cudaDeviceSynchronize(); //checkCudaErrors(cudaGetLastError());

	size_t numPixels = numRows()*numCols();
  	checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

	//check results and output the grey image
  	postProcess(output_file, h_greyImage);

	cleanup();
}

