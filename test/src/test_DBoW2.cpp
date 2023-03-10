#include <gtest/gtest.h>
#include <visnav/ORBVocabulary.h>
#include <visnav/converter.h>

#include <bitset>
#include <opencv2/opencv.hpp>
using namespace visnav;

TEST(OpencvTestSuite, TestDBow2) {
  ORBVocabulary* mpVocabulary = new ORBVocabulary();
  std::cout << mpVocabulary->getDepthLevels() << std::endl;
  std::cout << mpVocabulary->getBranchingFactor() << std::endl;
  std::cout << mpVocabulary->getScoringType() << std::endl;
}

std::vector<cv::Mat> toDescriptorVector(const cv::Mat& Descriptors) {
  std::vector<cv::Mat> vDesc;
  vDesc.reserve(Descriptors.rows);
  for (int j = 0; j < Descriptors.rows; j++)
    vDesc.push_back(Descriptors.row(j));

  return vDesc;
}

TEST(OpencvTestSuite, TestDBow2DB) {
  std::string strVocFile = "../../Vocabulary/ORBvoc.txt";
  // Load ORB Vocabulary

  ORBVocabulary* mpVocabulary = new ORBVocabulary();
  bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
  if (!bVocLoad) {
    std::cerr << "Wrong path to vocabulary. " << std::endl;
    std::cerr << "Falied to open at: " << strVocFile << std::endl;
    exit(-1);
  }
  std::cout << "Vocabulary loaded!" << std::endl << std::endl;

  cv::Ptr<cv::FeatureDetector> feature_detector;
  cv::Ptr<cv::DescriptorExtractor> feature_descriptor;
  cv::Ptr<cv::DescriptorMatcher> feature_matcher;
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  std::vector<cv::DMatch> matches;
  feature_detector = cv::ORB::create();
  feature_descriptor = cv::ORB::create();

  uint64 t1 = 1403715282312143104;
  // uint64 t2 = 1403715412662142976;
  uint64 t2 = 1403715287362142976;
  while (true) {
    cv::Mat image1;
    cv::Mat image2;
    std::string img1_path =
        "../../data/V1_01_easy/mav0/cam0/data/" + std::to_string(t1) + ".png";
    image1 = cv::imread(img1_path, 1);
    std::string img2_path =
        "../../data/V1_01_easy/mav0/cam1/data/" + std::to_string(t2) + ".png";
    image2 = cv::imread(img2_path, 1);
    cv::imshow("image1", image1);
    cv::imshow("image2", image2);
    feature_detector->detect(image1, keypoints1);
    feature_detector->detect(image2, keypoints2);
    // Second, we transform keypoints into descriptors
    feature_descriptor->compute(image1, keypoints1, descriptors1);
    feature_descriptor->compute(image2, keypoints2, descriptors2);
    std::vector<cv::Mat> vCurrentDesc1 = toDescriptorVector(descriptors1);
    std::vector<cv::Mat> vCurrentDesc2 = toDescriptorVector(descriptors2);
    for (size_t i = 0; i < vCurrentDesc1[0].cols; i++) {
      uchar temp = vCurrentDesc1[0].at<uchar>(0, i);
      std::cout << temp << std::endl;
      std::bitset<8> temp_in_bit(temp);
      std::cout << temp_in_bit << std::endl;
    }

    DBoW2::BowVector mBowVec1;
    DBoW2::FeatureVector mFeatVec1;
    DBoW2::BowVector mBowVec2;
    DBoW2::FeatureVector mFeatVec2;
    mpVocabulary->transform(vCurrentDesc1, mBowVec1, mFeatVec1, 4);
    mpVocabulary->transform(vCurrentDesc2, mBowVec2, mFeatVec2, 4);
    std::cout << mpVocabulary->score(mBowVec1, mBowVec2) << std::endl;
    int keyboard = cv::waitKey(30);
    if (keyboard == 'q' || keyboard == 27) break;
  }
}
