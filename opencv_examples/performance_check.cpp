#include <iostream>
#include "opencv2/opencv.hpp"
using namespace cv;

#define REP_COUNT 500
int main(int argc, char* argv[]) {
  cv::Mat h_img1 = cv::imread("images/inbloom.jpg", 0);

  {
    // Measure initial time ticks
    int64 work_begin = getTickCount();

    cv::Mat h_result1, h_result2, h_result3, h_result4, h_result5;
    for (int i = 0; i < REP_COUNT; i++) {
      cv::threshold(h_img1, h_result1, 128.0, 255.0, cv::THRESH_BINARY);
      cv::threshold(h_img1, h_result2, 128.0, 255.0, cv::THRESH_BINARY_INV);
      cv::threshold(h_img1, h_result3, 128.0, 255.0, cv::THRESH_TRUNC);
      cv::threshold(h_img1, h_result4, 128.0, 255.0, cv::THRESH_TOZERO);
      cv::threshold(h_img1, h_result5, 128.0, 255.0, cv::THRESH_TOZERO_INV);
    }

    // Measure difference in time ticks
    int64 delta = getTickCount() - work_begin;
    double freq = getTickFrequency();
    // Measure frames per second
    double work_fps = freq / delta;
    std::cout << "Performance of Thresholding on CPU: " << std::endl;
    std::cout << "Time: " << (1 / work_fps) << std::endl;
    std::cout << "FPS: " << work_fps << std::endl;
  }

  {
    cv::cuda::GpuMat d_result1, d_result2, d_result3, d_result4, d_result5,
        d_img1;
    // Measure initial time ticks
    int64 work_begin = getTickCount();
    d_img1.upload(h_img1);

    for (int i = 0; i < REP_COUNT; i++) {
      cv::cuda::threshold(d_img1, d_result1, 128.0, 255.0, cv::THRESH_BINARY);
      cv::cuda::threshold(d_img1, d_result2, 128.0, 255.0,
                          cv::THRESH_BINARY_INV);
      cv::cuda::threshold(d_img1, d_result3, 128.0, 255.0, cv::THRESH_TRUNC);
      cv::cuda::threshold(d_img1, d_result4, 128.0, 255.0, cv::THRESH_TOZERO);
      cv::cuda::threshold(d_img1, d_result5, 128.0, 255.0,
                          cv::THRESH_TOZERO_INV);
    }

    cv::Mat h_result1, h_result2, h_result3, h_result4, h_result5;
    d_result1.download(h_result1);
    d_result2.download(h_result2);
    d_result3.download(h_result3);
    d_result4.download(h_result4);
    d_result5.download(h_result5);
    // Measure difference in time ticks
    int64 delta = getTickCount() - work_begin;
    double freq = getTickFrequency();
    // Measure frames per second
    double work_fps = freq / delta;
    std::cout << "Performance of Thresholding on GPU: " << std::endl;
    std::cout << "Time: " << (1 / work_fps) << std::endl;
    std::cout << "FPS: " << work_fps << std::endl;
  }
  return 0;
}
