#include <gtest/gtest.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>
#include <tbb/concurrent_unordered_map.h>
#include <visnav/bow_voc.h>
#include <visnav/calibration.h>
#include <visnav/common_types.h>
// #include <visnav/gui_helper.h>
#include <visnav/ORBVocabulary.h>
#include <visnav/keypoints.h>
#include <visnav/loop_closure_utils.h>
// #include <visnav/map_utils.h>
#include <visnav/converter.h>
#include <visnav/matching_utils.h>
#include <visnav/serialization.h>
// #include <visnav/tracks.h>
// #include <visnav/vo_utils.h>

// #include <CLI/CLI.hpp>
// #include <algorithm>
// #include <atomic>
// #include <chrono>
#include <iostream>
// #include <queue>

#include <sophus/se3.hpp>
#include <sstream>
// #include <thread>
using namespace visnav;
int NUM_CAMS = 2;
std::string dataset_path = "../../data/V1_01_easy/mav0/";
std::string cam_calib = "../../opt_calib.json";
std::string voc_path = "../../data/ORBvoc.cereal";
tbb::concurrent_unordered_map<FrameCamId, std::string> images;
Calibration calib_cam;
std::vector<Timestamp> timestamps;
std::shared_ptr<BowVocabulary> bow_voc;
std::shared_ptr<BowDatabase> bow_db;
void load_data(const std::string& dataset_path, const std::string& calib_path) {
  const std::string timestams_path = dataset_path + "/cam0/data.csv";

  {
    std::ifstream times(timestams_path);

    int id = 0;

    while (times) {
      std::string line;
      std::getline(times, line);

      //      if (line.size() < 20 || line[0] == '#' || id > 2700) continue;
      if (line.size() < 20 || line[0] == '#') continue;
      {
        std::string timestamp_str = line.substr(0, 19);
        std::istringstream ss(timestamp_str);
        Timestamp timestamp;
        ss >> timestamp;
        timestamps.push_back(timestamp);
      }

      std::string img_name = line.substr(20, line.size() - 21);

      for (int i = 0; i < NUM_CAMS; i++) {
        FrameCamId fcid(id, i);

        std::stringstream ss;
        ss << dataset_path << "/cam" << i << "/data/" << img_name;

        images[fcid] = ss.str();
      }

      id++;
    }

    std::cerr << "Loaded " << id << " image pairs" << std::endl;
  }

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib_cam);
      std::cout << "Loaded camera from " << calib_path << " with models ";
      for (const auto& cam : calib_cam.intrinsics) {
        std::cout << cam->name() << " ";
      }
      std::cout << std::endl;
    } else {
      std::cerr << "could not load camera calibration " << calib_path
                << std::endl;
      std::abort();
    }
  }
}
TEST(LoopClosureTestSuite, BoWScore) {
  if (!voc_path.empty()) {
    bow_voc.reset(new BowVocabulary(voc_path));
    bow_db.reset(new BowDatabase);
  }
  load_data(dataset_path, cam_calib);
  FrameCamId fcid1(2121, 0), fcid2(2100, 0);
  // std::string image1_path = images.at(fcid1);
  // std::string image2_path = images.at(fcid2);
  uint64 t1 = 1403715282312143104;
  // uint64 t2 = 1403715412662142976;
  uint64 t2 = 1403715287362142976;
  std::string image1_path =
      "../../data/V1_01_easy/mav0/cam0/data/" + std::to_string(t1) + ".png";
  std::string image2_path =
      "../../data/V1_01_easy/mav0/cam1/data/" + std::to_string(t2) + ".png";
  pangolin::ManagedImage<uint8_t> img1 = pangolin::LoadImage(image1_path);
  pangolin::ManagedImage<uint8_t> img2 = pangolin::LoadImage(image2_path);
  KeypointsData kd1, kd2;
  detectKeypointsAndDescriptors(img1, kd1, 1500, true);
  detectKeypointsAndDescriptors(img2, kd2, 1500, true);
  BowVector v1, v2;
  bow_voc->transform(kd1.corner_descriptors, v1);
  bow_voc->transform(kd2.corner_descriptors, v2);
  double score = compute_bow_score(v1, v2);
  std::cout << "The similarity score is: " << score << std::endl;
  std::cout << "The frames are: " << image1_path << " and " << image2_path
            << std::endl;

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

  std::vector<cv::Mat> vCurrentDesc1 =
      to_opencv_descriptors(kd1.corner_descriptors);
  std::vector<cv::Mat> vCurrentDesc2 =
      to_opencv_descriptors(kd2.corner_descriptors);
  DBoW2::BowVector mBowVec1;
  DBoW2::FeatureVector mFeatVec1;
  DBoW2::BowVector mBowVec2;
  DBoW2::FeatureVector mFeatVec2;
  mpVocabulary->transform(vCurrentDesc1, mBowVec1, mFeatVec1, 4);
  mpVocabulary->transform(vCurrentDesc2, mBowVec2, mFeatVec2, 4);
  std::cout << mpVocabulary->score(mBowVec1, mBowVec2) << std::endl;
  // Check if the conversion is correct

  std::bitset<256> desc_back_bitset = to_bitset_descriptor(vCurrentDesc1[0]);
  for (size_t i = 0; i < 256; i++) {
    if (desc_back_bitset.test(i)) {
      ASSERT_TRUE(kd1.corner_descriptors[0].test(i));
    } else {
      ASSERT_FALSE(kd1.corner_descriptors[0].test(i));
    }
  }

  cv::Ptr<cv::FeatureDetector> feature_detector;
  cv::Ptr<cv::DescriptorExtractor> feature_descriptor;
  cv::Ptr<cv::DescriptorMatcher> feature_matcher;
  std::vector<cv::KeyPoint> keypoints1, keypoints2;
  cv::Mat descriptors1, descriptors2;
  feature_detector = cv::ORB::create();
  feature_descriptor = cv::ORB::create();

  cv::Mat image1;
  cv::Mat image2;

  image1 = cv::imread(image1_path, 1);
  image2 = cv::imread(image2_path, 1);
  feature_detector->detect(image1, keypoints1);
  feature_detector->detect(image2, keypoints2);
  // Second, we transform keypoints into descriptors
  feature_descriptor->compute(image1, keypoints1, descriptors1);
  feature_descriptor->compute(image2, keypoints2, descriptors2);

  std::vector<std::bitset<256>> corner_descriptors1 =
      to_bitset_descriptors(descriptors1);
  std::vector<std::bitset<256>> corner_descriptors2 =
      to_bitset_descriptors(descriptors2);

  bow_voc->transform(corner_descriptors1, v1);
  bow_voc->transform(corner_descriptors2, v2);

  score = compute_bow_score(v1, v2);
  std::cout << "The similarity score is: " << score << std::endl;
}
