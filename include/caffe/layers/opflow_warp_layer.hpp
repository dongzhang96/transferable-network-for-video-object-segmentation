#ifndef CAFFE_OPFLOW_WARP_LAYER_HPP_
#define CAFFE_OPFLOW_WARP_LAYER_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class OpflowWarpLayer : public Layer<Dtype> {
	public:
	    explicit OpflowWarpLayer(const LayerParameter& param)
	      : Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		   const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		                     const vector<Blob<Dtype>*>& top);
		virtual inline const char* type() const {return "OpflowWarp";}
		virtual inline int ExactNumBottomBlobs() const {return 4;}
		virtual inline int ExactNumTopBlobs() const {return 1;}
	
	protected:
	    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		   const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
		   const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& bottom);
		
		bool is_forward_;
		cv::Mat opImgX;
		cv::Mat opImgY;
		cv::Mat meshX;
		cv::Mat meshY;
		
		int count;
		int height;
		int width;
};

}

#endif