#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "caffe/layers/opflow_warp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
	
template <typename Dtype>
void OpflowWarpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top){
	    CHECK_EQ(bottom[0]->count(),bottom[1]->count());
		CHECK_EQ(bottom[1]->count(),bottom[2]->count());
		
		OpflowWarpParameter opflow_warp_param = this->layer_param_.opflow_warp_param();
		is_forward_ = opflow_warp_param.is_forward();
		
		count = bottom[0]->count();
		height = bottom[0]->height();
		width = bottom[0]->width();
		
		Dtype* opflowX = bottom[1]->mutable_cpu_data();
		Dtype* opflowY = bottom[2]->mutable_cpu_data();
		const Dtype* opflowLabel = bottom[3]->cpu_data();
		
		if(opflowLabel[0]==Dtype(1))
		{
			caffe_scal(count,Dtype(-1),opflowX);
			caffe_scal(count,Dtype(-1),opflowY);
		}
		if(!is_forward_)
		{
			caffe_scal(count,Dtype(-1),opflowX);
			caffe_scal(count,Dtype(-1),opflowY);
		}
		
		opImgX.create(height,width,CV_32FC1);
		int pixCount = 0;
		for (int i = 0;i < height;i++)
		for (int j = 0;j < width;j++)
		{
			opImgX.at<float>(i,j) = opflowX[pixCount];
			pixCount++;
		}
		
	//cv::Mat opImgY;
	pixCount = 0;
	opImgY.create(height, width, CV_32FC1);	
	for (int i = 0;i < height;i++)
		for (int j = 0;j < width;j++)
		{
			opImgY.at<float>(i,j) = opflowY[pixCount];
			pixCount++;
		}
	meshX.create(height, width,CV_32FC1);
    meshY.create(height, width,CV_32FC1);
    cv::Range xgv = cv::Range(0,meshX.cols - 1);
    cv::Range ygv = cv::Range(0,meshX.rows - 1);
    std::vector<float> t_x, t_y;
    for (int i = xgv.start; i <= xgv.end; i++) t_x.push_back(i);
    for (int j = ygv.start; j <= ygv.end; j++) t_y.push_back(j);
    cv::repeat(cv::Mat(t_x).t(), t_y.size(), 1, meshX);
    cv::repeat(cv::Mat(t_y), 1, t_x.size(), meshY);
}
	
template <typename Dtype>
void OpflowWarpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top)
{
		top[0]->ReshapeLike(*bottom[0]);									
}

template <typename Dtype>
void OpflowWarpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                        const vector<Blob<Dtype>*>& top)
{
	Dtype* salmapCur = bottom[0]->mutable_cpu_data();
	Dtype* salmapWarp = top[0]->mutable_cpu_data();
	
	cv::Mat srcMask;
	srcMask.create(height, width, CV_32FC1);	
	int pixCount = 0;
	for (int i = 0;i < height;i++)
		for (int j = 0;j < width;j++)
		{
			srcMask.at<float>(i,j) = salmapCur[pixCount];
			pixCount++;
		}
	
	cv::Mat offsetX = meshX + opImgX;
    cv::Mat offsetY = meshY + opImgY;
	
	cv::Mat resMask;
    resMask.create(height, width, CV_32FC1);
	cv::remap(srcMask,resMask,offsetX,offsetY,cv::INTER_LINEAR,0,cv::Scalar::all(0));
	pixCount = 0;
	for (int i = 0;i < height;i++)
		for (int j = 0;j < width;j++)
		{
			salmapWarp[pixCount] = resMask.at<float>(i,j);
			pixCount++;
		}
}

template <typename Dtype>
void OpflowWarpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom)
{
	const Dtype* salmapWarp_diff = top[0]->cpu_diff();
	Dtype* salmapCur_diff = bottom[0]->mutable_cpu_diff();
	Dtype* opflowX_diff = bottom[1]->mutable_cpu_diff();
	Dtype* opflowY_diff = bottom[2]->mutable_cpu_diff();
	
	Dtype* opflowLabel_diff = bottom[3]->mutable_cpu_diff();
	
	//inverse the optical flow information
	opImgX = opImgX * (-1);
	opImgY = opImgY * (-1);
	
	cv::Mat offsetX = meshX + opImgX;
    cv::Mat offsetY = meshY + opImgY;
	
	cv::Mat topDiff;
	topDiff.create(height, width, CV_32FC1);	
	int pixCount = 0;
	for (int i = 0;i < height;i++)
		for (int j = 0;j < width;j++)
		{
			topDiff.at<float>(i,j) = salmapWarp_diff[pixCount];
			pixCount++;
		}
	cv::Mat bottomDiff;
    bottomDiff.create(height, width, CV_32FC1);
	cv::remap(topDiff,bottomDiff,offsetX,offsetY,cv::INTER_LINEAR,0,cv::Scalar::all(0));
	pixCount = 0;
	for (int i = 0;i < height;i++)
		for (int j = 0;j < width;j++)
		{
			salmapCur_diff[pixCount] = bottomDiff.at<float>(i,j);
			pixCount++;
		}
		
	caffe_set(count,Dtype(0),opflowX_diff);
	caffe_set(count,Dtype(0),opflowY_diff);
	caffe_set(Dtype(1),Dtype(1),opflowLabel_diff);
}

INSTANTIATE_CLASS(OpflowWarpLayer);
REGISTER_LAYER_CLASS(OpflowWarp);
}