#include <vector>

#include "caffe/layers/linear_sum_layer.hpp"

namespace caffe {

template <typename Dtype>
void LinearSumLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num_axes(), bottom[1]->num_axes());
  CHECK_EQ(bottom[0]->num_axes(), bottom[2]->num_axes());
  CHECK_EQ(bottom[0]->num_axes(), bottom[3]->num_axes());
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i == 1) {
      CHECK_EQ(bottom[1]->shape(i), 1);
	  CHECK_EQ(bottom[2]->shape(i), 1);
    }
    else {
      CHECK_EQ(bottom[0]->shape(i), bottom[1]->shape(i));
	  CHECK_EQ(bottom[0]->shape(i), bottom[2]->shape(i));
    }
  }
  //check shape for label
  for (int i = 0; i < bottom[0]->num_axes(); ++i) {
    if (i == 0) {
      CHECK_EQ(bottom[0]->shape(i), bottom[3]->shape(i));
    }
    else {
      CHECK_EQ(bottom[3]->shape(i), 1);
    }
  }
  H_ = bottom[0]->shape(1);
  N_ = bottom[0]->count(2);

  LOG(INFO) << "Num of coefficient is " << N_;
  LOG(INFO) << "Output dimension is " << H_;
}

template <typename Dtype>
void LinearSumLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //map
  vector<int> top_shape;
  top_shape.push_back(bottom[0]->shape(0));
  top_shape.push_back(H_);
  top_shape.push_back(1);
  top_shape.push_back(1);
  top[0]->Reshape(top_shape);
  
}

template <typename Dtype>
void LinearSumLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  int num = bottom[0]->shape(0);
  const Dtype* X = bottom[0]->cpu_data();
  //weight map from other class, foreground
  const Dtype* W_other = bottom[1]->cpu_data();
  //weight map from the same class, background
  const Dtype* W_same = bottom[2]->cpu_data();
  //flag, Is The Same Class, 0---other class; 1--same class
  const Dtype* Flag_class = bottom[3]->cpu_data();
  
  Dtype* Y = top[0]->mutable_cpu_data();
  for (int i = 0; i < num; ++i) {
	  const int flag_value = static_cast<int>(Flag_class[0]);	  
	  if ((flag_value == 1) && ((i+1) % 3) == 0) {
		  caffe_cpu_gemv(CblasNoTrans, H_, N_, (Dtype)1, X, W_same, (Dtype)0, Y);		  
	  }
	  else {
		  caffe_cpu_gemv(CblasNoTrans, H_, N_, (Dtype)1, X, W_other, (Dtype)0, Y);
	  }
	  W_other += bottom[1]->offset(1);
	  W_same += bottom[2]->offset(1);
	  Flag_class += bottom[3]->offset(1);
	  Y += top[0]->offset(1);
	  X += bottom[0]->offset(1);
  }
}

template <typename Dtype>
void LinearSumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int num = bottom[0]->shape(0);
  const Dtype* X = bottom[0]->cpu_data();
  const Dtype* W_other = bottom[1]->cpu_data();
  const Dtype* W_same = bottom[2]->cpu_data();
  const Dtype* Flag_class = bottom[3]->cpu_data();
  const Dtype* Y_diff = top[0]->cpu_diff();
  Dtype* X_diff = bottom[0]->mutable_cpu_diff();
  Dtype* W_other_diff = bottom[1]->mutable_cpu_diff();
  Dtype* W_same_diff = bottom[2]->mutable_cpu_diff();
  
  //num=batchnum
  int numCh = bottom[1]->num();
  int countCh = bottom[1]->count();
  int dimCh = countCh / numCh;
  
  for (int i = 0; i < num; ++i) {
	  const int flag_value = static_cast<int>(Flag_class[0]);
	  	  //selectively update X_diff and W_diff: W_same_diff or W_other_diff
	  if ((flag_value == 1) && ((i+1) % 3) == 0) {
		  if (propagate_down[0]) {
			  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, H_, N_, 1,
			      (Dtype)1, Y_diff, W_same, (Dtype)0, X_diff);
		  }
		  if (propagate_down[2]) {
			  caffe_cpu_gemv(CblasTrans, H_, N_, (Dtype)1, X, Y_diff, 
			      (Dtype)0, W_same_diff);
		  }
		  caffe_set(dimCh, Dtype(0), W_other_diff);
	  }
	  else {
		  if (propagate_down[0]) {
			  caffe_cpu_gemm(CblasNoTrans, CblasNoTrans, H_, N_, 1,
			      (Dtype)1, Y_diff, W_other, (Dtype)0, X_diff);
		  }	  
		  if (propagate_down[1]) {
			  caffe_cpu_gemv(CblasTrans, H_, N_, (Dtype)1, X, Y_diff, 
			      (Dtype)0, W_other_diff);
		  }
		  caffe_set(dimCh, Dtype(0), W_same_diff);
	  }	  
	  if (propagate_down[3]) {
		  LOG(FATAL) << " Layer cannot backpropagate to label inputs.";
	  }	  
    
    X += bottom[0]->offset(1);
    X_diff += bottom[0]->offset(1);
    W_other += bottom[1]->offset(1);
    W_other_diff += bottom[1]->offset(1);
	W_same += bottom[2]->offset(1);
	W_same_diff += bottom[2]->offset(1);
	Flag_class += bottom[3]->offset(1);
    Y_diff += top[0]->offset(1);
  }
}

//#ifdef CPU_ONLY
//STUB_GPU(LinearSumLayer);
//#endif

INSTANTIATE_CLASS(LinearSumLayer);
REGISTER_LAYER_CLASS(LinearSum);

}  // namespace caffe