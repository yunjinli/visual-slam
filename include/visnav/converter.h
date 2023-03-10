#include <Eigen/Core>
#include <Eigen/StdVector>
#include <bitset>
#include <opencv2/opencv.hpp>
#include <vector>

namespace visnav {
/// @brief Change the cv::Mat to std::vector<cv::Mat> for
/// @param Descriptors the descriptor in cv::Mat form
/// @return Return the cv::Mat descripors in std::vector form, each descriptor
/// is in the form of a 1x32 shape 8-bit element.
std::vector<cv::Mat> to_descriptor_vector(const cv::Mat& Descriptors) {
  std::vector<cv::Mat> vDesc;
  vDesc.reserve(Descriptors.rows);
  for (int j = 0; j < Descriptors.rows; j++)
    vDesc.push_back(Descriptors.row(j));

  return vDesc;
}
/// @brief Convert a single std::bitset<256> descriptor to cv::Mat format
/// @param descriptor Single std::bitset<256> descriptor
/// @return cv::Mat format descriptor
cv::Mat to_opencv_descriptor(const std::bitset<256>& descriptor) {
  cv::Mat cv_descriptor = cv::Mat::zeros(1, 32, CV_8U);
  unsigned char* p = cv_descriptor.ptr<unsigned char>();
  for (size_t i = 0; i < 256; i++) {
    if (descriptor.test(i)) {
      *p |= 1 << (7 - (i % 8));
    }
    if (i % 8 == 7) p++;
  }
  return cv_descriptor;
}
/// @brief Convert all descriptors in a image with std::bitset<256> descriptor
/// to cv::Mat format
/// @param descriptors All descriptors in a image
/// @return Converted descriptors in cv::Mat format
std::vector<cv::Mat> to_opencv_descriptors(
    const std::vector<std::bitset<256>>& descriptors) {
  std::vector<cv::Mat> vDesc;
  for (const auto& descriptor : descriptors) {
    cv::Mat cv_descriptor = to_opencv_descriptor(descriptor);
    vDesc.push_back(cv_descriptor);
  }
  return vDesc;
}
/// @brief Convert single cv::Mat descriptor to std::bitset<256> format
/// @param descriptor cv::Mat descriptor
/// @return std::bitset<256> descriptor
std::bitset<256> to_bitset_descriptor(const cv::Mat& descriptor) {
  std::bitset<256> descriptor_bitset;
  const unsigned char* p = descriptor.ptr<unsigned char>();
  for (size_t i = 0; i < 256; i++) {
    std::bitset<8> temp(*p);
    if (temp.test(7 - (i % 8))) {
      descriptor_bitset.set(i);
    }
    if (i % 8 == 7) p++;
  }
  return descriptor_bitset;
}

/// @brief Convert all descriptors in a image with cv::Mat format to
/// std::bitset<256>
/// @param descriptors All descriptors in a image
/// @return All descriptors in std::bitset<256> format
std::vector<std::bitset<256>> to_bitset_descriptors(
    const cv::Mat& descriptors) {
  std::vector<std::bitset<256>> vDesc;
  vDesc.reserve(descriptors.rows);
  for (int j = 0; j < descriptors.rows; j++)
    vDesc.push_back(to_bitset_descriptor(descriptors.row(j)));

  return vDesc;
}

void to_eigen_keypoints(
    const std::vector<cv::KeyPoint>& cv_corners,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        corners) {
  corners.clear();
  for (const auto& corner : cv_corners) {
    corners.emplace_back(corner.pt.x, corner.pt.y);
  }
}

void to_std_pair_matches(std::vector<cv::DMatch>& cv_matches,
                         std::vector<std::pair<int, int>>& matches) {
  matches.clear();
  for (const auto& match : cv_matches) {
    matches.push_back(std::make_pair(match.queryIdx, match.trainIdx));
  }
}

}  // namespace visnav
