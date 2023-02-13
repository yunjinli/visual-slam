#include <io/dataset_io.h>

#include <opencv2/opencv.hpp>

using namespace slam;

DatasetIoInterfacePtr dataset_io;
std::string path = "../data/euro_data/MH_01_easy";

int main(int argc, char** argv) {
  dataset_io = DatasetIoFactory::getDatasetIo("euroc");
  dataset_io->read(path);
  cv::Mat image1;
  cv::Mat image2;
  cv::Ptr<cv::FeatureDetector> feature_detector = cv::ORB::create();
  std::vector<cv::KeyPoint> keypoints1;
  for (const auto& t : dataset_io->get_data()->get_image_timestamps()) {
    std::string img1_path =
        path + "/mav0/cam0/data/" + std::to_string(t) + ".png";
    image1 = cv::imread(img1_path, 1);
    cv::Mat img1_keypoints;
    feature_detector->detect(image1, keypoints1);
    cv::drawKeypoints(image1, keypoints1, img1_keypoints);

    cv::namedWindow("Cam0", cv::WINDOW_AUTOSIZE);
    cv::imshow("Cam0", image1);

    cv::namedWindow("Cam0-Corners", cv::WINDOW_AUTOSIZE);
    cv::imshow("Cam0-Corners", img1_keypoints);
    int keyboard = cv::waitKey(30);
    if (keyboard == 'q' || keyboard == 27) return -1;
  }
}