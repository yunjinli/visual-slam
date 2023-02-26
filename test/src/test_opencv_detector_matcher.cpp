#include <gtest/gtest.h>
#include <io/dataset_io.h>

#include <opencv2/opencv.hpp>

slam::DatasetIoInterfacePtr dataset_io;
std::string path = "../data/euro_data/MH_01_easy";

TEST(OpencvTestSuite, DetectMatches) {
  cv::Ptr<cv::FeatureDetector> feature_detector;
  cv::Ptr<cv::DescriptorExtractor> feature_descriptor;
  cv::Ptr<cv::DescriptorMatcher> feature_matcher;
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  std::vector<cv::DMatch> matches;
  feature_detector = cv::ORB::create();
  feature_descriptor = cv::ORB::create();
  feature_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  dataset_io = slam::DatasetIoFactory::getDatasetIo("euroc");
  dataset_io->read(path);
  cv::Mat image1;
  cv::Mat image2;
  cv::Mat img_match;
  for (const auto& t : dataset_io->get_data()->get_image_timestamps()) {
    std::string img1_path =
        path + "/mav0/cam0/data/" + std::to_string(t) + ".png";
    image1 = cv::imread(img1_path, 1);
    std::string img2_path =
        path + "/mav0/cam1/data/" + std::to_string(t) + ".png";
    image2 = cv::imread(img2_path, 1);

    feature_detector->detect(image1, keypoints1);
    feature_detector->detect(image2, keypoints2);
    // Second, we transform keypoints into descriptors
    feature_descriptor->compute(image1, keypoints1, descriptors1);
    feature_descriptor->compute(image2, keypoints2, descriptors2);

    // BFMatcher matcher ( NORM_HAMMING );
    feature_matcher->match(descriptors1, descriptors2, matches);

    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, img_match);
    cv::imshow("All matches", img_match);
    int keyboard = cv::waitKey(30);
    if (keyboard == 'q' || keyboard == 27) break;
  }
}

TEST(OpencvTestSuite, TestPrintDescriptorBit) {
  cv::Ptr<cv::FeatureDetector> feature_detector;
  cv::Ptr<cv::DescriptorExtractor> feature_descriptor;
  cv::Ptr<cv::DescriptorMatcher> feature_matcher;
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  std::vector<cv::DMatch> matches;
  feature_detector = cv::ORB::create();
  feature_descriptor = cv::ORB::create();
  feature_matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  dataset_io = slam::DatasetIoFactory::getDatasetIo("euroc");
  dataset_io->read(path);
  cv::Mat image1;
  cv::Mat image2;
  cv::Mat img_match;
  for (const auto& t : dataset_io->get_data()->get_image_timestamps()) {
    std::string img1_path =
        path + "/mav0/cam0/data/" + std::to_string(t) + ".png";
    image1 = cv::imread(img1_path, 1);
    std::string img2_path =
        path + "/mav0/cam1/data/" + std::to_string(t) + ".png";
    image2 = cv::imread(img2_path, 1);

    feature_detector->detect(image1, keypoints1);
    feature_detector->detect(image2, keypoints2);
    // Second, we transform keypoints into descriptors
    feature_descriptor->compute(image1, keypoints1, descriptors1);
    feature_descriptor->compute(image2, keypoints2, descriptors2);

    // BFMatcher matcher ( NORM_HAMMING );
    feature_matcher->match(descriptors1, descriptors2, matches);
    int hamming_dist =
        cv::norm(descriptors1.row(0), descriptors2.row(0), cv::NORM_HAMMING);
    std::cout << hamming_dist << std::endl;
  }
}