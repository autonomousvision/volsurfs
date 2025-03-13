#pragma once

// pytorch
#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

// eigen
#include <Eigen/Dense>

// c++
#include <iostream>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> EigenMatrixXfRowMajor;

// converts a RowMajor eigen matrix of size HW into a tensor of size HW
inline torch::Tensor eigen2tensor(const EigenMatrixXfRowMajor &eigen_mat)
{

    torch::Tensor wrapped_mat = torch::from_blob(const_cast<float *>(eigen_mat.data()), /*sizes=*/{eigen_mat.rows(), eigen_mat.cols()}, at::kFloat);
    torch::Tensor tensor = wrapped_mat.clone(); // we have to take ownership of the data, otherwise the eigen_mat might go out of scope and then we will point to undefined data

    return tensor;
}

// // converts a RowMajor eigen matrix of size HW cv::Mat of size XY
// inline cv::Mat eigen2mat(const EigenMatrixXfRowMajor &eigen_mat, const int rows, const int cols)
// {

//     CHECK(eigen_mat.rows() == rows * cols) << "We need a row in the eigen mat for each pixel in the image of the cv mat. However nr of rows in the eigen mat is " << eigen_mat.rows() << " while rows*cols of the cv mat is " << rows * cols;

//     int cv_mat_type = -1;
//     if (eigen_mat.cols() == 1)
//     {
//         cv_mat_type = CV_32FC1;
//     }
//     else if (eigen_mat.cols() == 2)
//     {
//         cv_mat_type = CV_32FC2;
//     }
//     else if (eigen_mat.cols() == 3)
//     {
//         cv_mat_type = CV_32FC3;
//     }
//     else if (eigen_mat.cols() == 4)
//     {
//         cv_mat_type = CV_32FC4;
//     }

//     cv::Mat cv_mat(rows, cols, cv_mat_type, (void *)eigen_mat.data());

//     return cv_mat.clone();
// }

// // converts tensor of shape hw into a RowMajor eigen matrix of size HW
// inline EigenMatrixXfRowMajor tensor2eigen(const torch::Tensor &tensor_in)
// {

//     CHECK(tensor_in.dim() == 2) << "The tensor should be a 2D one with shape HW, however it has dim: " << tensor_in.dim();
//     CHECK(tensor_in.scalar_type() == at::kFloat) << "Tensor should be float. Didn't have time to write templates for this functions";

//     torch::Tensor tensor = tensor_in.to(at::kCPU);

//     int rows = tensor.size(0);
//     int cols = tensor.size(1);

//     EigenMatrixXfRowMajor eigen_mat(rows, cols);
//     eigen_mat = Eigen::Map<EigenMatrixXfRowMajor>(tensor.data_ptr<float>(), rows, cols);

//     // make a deep copy of it because map does not actually take ownership
//     EigenMatrixXfRowMajor eigen_mat_copy;
//     eigen_mat_copy = eigen_mat;

//     return eigen_mat_copy;
// }

// // converts a std::vector<float> to a one dimensional tensor with whatever size the vector has
// inline torch::Tensor vec2tensor(const std::vector<float> &vec)
// {

//     torch::Tensor wrapped_mat = torch::from_blob(const_cast<float *>(vec.data()), /*sizes=*/{(int)vec.size()}, at::kFloat);
//     torch::Tensor tensor = wrapped_mat.clone(); // we have to take ownership of the data, otherwise the eigen_mat might go out of scope and then we will point to undefined data

//     return tensor;
// }

// empties cache used by cuda
inline void cuda_clear_cache()
{
    c10::cuda::CUDACachingAllocator::emptyCache();
}