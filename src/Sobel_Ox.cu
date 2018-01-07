#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#define PI 3.14159265;

using namespace std;
using namespace cv	;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__
void image_Sobel(uchar3 *d_inputImage, unsigned char *d_grad, unsigned char *d_gradx, unsigned char *d_grady, float *d_filter_x, float *d_filter_y, int d_filterWidth, int numRows, int numCols)
{

    int absolute_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int absolute_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int Id = absolute_x + absolute_y*numCols;

    const  int sizeofinput = numRows*numCols;
 //   __shared__ uchar3 sh_arr[256];
 //   sh_arr[Id] = d_inputImage[Id];
 //   __syncthreads();
    float Xcolor_x=0,Xcolor_y=0 , Xcolor_z=0, Ycolor_x=0, Ycolor_y=0, Ycolor_z=0;
    float Xfilter_value;
    float Yfilter_value;

    for(int fx=0;fx<d_filterWidth;fx++)
    {
        int px = fx + (absolute_x-d_filterWidth/2);

        for(int fy=0;fy<d_filterWidth;fy++)
        {

            int py = fy + (absolute_y-d_filterWidth/2);

            px = min(max(0,px),numCols-1);
            py = min(max(0,py),numRows-1);

            uchar3 rgb_pixel = d_inputImage[px + py*numCols];

            float Xfilter_value = d_filter_x[fx + fy*d_filterWidth];
            float Yfilter_value = d_filter_y[fx + fy*d_filterWidth];

            Xcolor_x+= Xfilter_value*static_cast<float>(rgb_pixel.x);
            Xcolor_y+= Xfilter_value*static_cast<float>(rgb_pixel.y);
            Xcolor_z+= Xfilter_value*static_cast<float>(rgb_pixel.z);

            Ycolor_x+= Yfilter_value*static_cast<float>(rgb_pixel.x);
            Ycolor_y+= Yfilter_value*static_cast<float>(rgb_pixel.y);
            Ycolor_z+= Yfilter_value*static_cast<float>(rgb_pixel.z);
        }
    }
    d_gradx[Id] = max(max(Xcolor_x,Xcolor_y),Xcolor_z);
    d_grady[Id] = max(max(Ycolor_x,Ycolor_y),Ycolor_z);

    d_grad[Id] = (d_gradx[Id] + d_grady[Id])/2;

}


__global__
void calctheta(unsigned char * theta,unsigned char *d_gradx, unsigned char *d_grady, int numRows, int numCols)
{

    int absolute_x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int absolute_y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int Id = absolute_x + absolute_y*numCols;

    float abs_angle = atan2f (d_grady[Id],d_gradx[Id])*180/PI;

    if(abs_angle<0)
    {
        theta[Id] = 180 + abs_angle;
    }
    else
    {
        theta[Id] = abs_angle;
    }
 //   printf("%f \t",static_cast<float>(theta[Id]));
}

__global__
void binning()
{


}



int main()
{


    float h_filter_x[9]={-1,0,1,-2,0,2,-1,0,1};
    float h_filter_y[9]={1,2,1,0,0,0,-1,-2,-1};
    int h_filterWidth=3;


    //Read the image

    Mat input,input_gray;

    VideoCapture cap(0);

    if(!cap.isOpened())
    {
    	cout << "Error opening video stream or file" << endl;
    	return -1;
    }

    if( cap.isOpened())
    {
    	cout << "In capture ..." << endl;
    	for(;;)
    	{


    		cap >> input;

    		if(waitKey(1) >= 0) break;

    		if(!input.data) return -1;

    		cout<<"total: "<<input.total();

    		//Initialize stuff on CPU

    		unsigned char *h_gradx = (unsigned char*)malloc(input.total());
    		unsigned char *h_grady = (unsigned char*)malloc(input.total());
    		unsigned char *h_grad = (unsigned char*)malloc(input.total());
    		unsigned char *h_theta = (unsigned char*)malloc(input.total());

    		//Initialize and Copy stuff on GPU

    		uchar3 *d_inputImage = new uchar3[input.total()];
    		float *d_filter_x;
    		float *d_filter_y;


    		unsigned char *d_gradx, *d_grady, *d_grad, *d_theta;

    		gpuErrchk(cudaMalloc((void**)&d_gradx, input.total()*sizeof(unsigned char)));
    		gpuErrchk(cudaMalloc((void**)&d_grady, input.total()*sizeof(unsigned char)));
    		gpuErrchk(cudaMalloc((void**)&d_grad, input.total()*sizeof(unsigned char)));
    		gpuErrchk(cudaMalloc((void**)&d_theta, input.total()*sizeof(unsigned char)));

    		gpuErrchk(cudaMalloc((void**)&d_filter_y,h_filterWidth*h_filterWidth*sizeof(float)));
    		gpuErrchk(cudaMalloc((void**)&d_filter_x,h_filterWidth*h_filterWidth*sizeof(float)));

    		gpuErrchk(cudaMemcpy(d_filter_x, h_filter_x, h_filterWidth*h_filterWidth*sizeof(float), cudaMemcpyHostToDevice));
    		gpuErrchk(cudaMemcpy(d_filter_y, h_filter_y, h_filterWidth*h_filterWidth*sizeof(float), cudaMemcpyHostToDevice));

    		gpuErrchk(cudaMalloc((void**)&d_inputImage,input.total()*sizeof(uchar3)));
    		gpuErrchk(cudaMemcpy(d_inputImage, input.data,input.total()*sizeof(uchar3), cudaMemcpyHostToDevice));


    		int numrows = input.rows;
    		int numcols = input.cols;

    		const dim3 blockSize(16, 16, 1);
    		const dim3 gridSize((numcols/blockSize.x),(numrows/blockSize.y),1);

    		cout<<"\ngridSize::"<<gridSize.x<<" "<<gridSize.y;
    		cout<<"\nBlockSize::"<<blockSize.x<<" "<<blockSize.y<<endl;

    		image_Sobel<<<gridSize, blockSize>>>(d_inputImage, d_grad, d_gradx, d_grady, d_filter_y, d_filter_x, h_filterWidth, numrows, numcols);
    		cudaThreadSynchronize();

    		uchar3 *h_output = (uchar3*)malloc(input.total()*sizeof(uchar3));

    		cudaMemcpy(h_output, d_inputImage, input.total()*sizeof(uchar3), cudaMemcpyDeviceToHost);
    		cudaMemcpy(h_gradx, d_gradx, input.total()*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    		cudaMemcpy(h_grady, d_grady, input.total()*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    		cudaMemcpy(h_grad, d_grad, input.total()*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    		calctheta<<<gridSize, blockSize>>>(d_theta, d_gradx, d_grady, numrows, numcols);
    		cudaThreadSynchronize();

    		Mat img(numrows,numcols,CV_8UC1,h_grad,cv::Mat::AUTO_STEP);

    		imshow("rgb",img);
    		cout<<"image dims: "<<img.size();
    		cudaDeviceSynchronize();






    	}
    }
    return 0;
}
