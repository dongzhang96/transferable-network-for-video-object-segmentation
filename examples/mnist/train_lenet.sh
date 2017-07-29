#!/usr/bin/env sh
GLOG_logtostderr=0 GLOG_log_dir=./Log/ \

/disk1/yangle/ICCVExt/software/caffe-master/.build_release/tools/caffe train \
 --solver=/disk1/yangle/ICCVExt/software/caffe-master/examples/mnist/lenet_solver.prototxt \
 -gpu 3 
  2>&1 | tee /disk1/yangle/ICCVExt/software/caffe-master/examples/mnist/train_doc.txt
