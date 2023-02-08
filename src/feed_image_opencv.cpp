#include <stdio.h>
#include <visnav/common_types.h>

#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using namespace visnav;
const std::string dataset_path = "../data/euro_data/MH_01_easy";
std::vector<Timestamp> timestamps_frame;

int main() {
  const std::string timestams_path = dataset_path + "/mav0/cam0/data.csv";
  {
    std::ifstream times(timestams_path);

    int id = 0;

    while (times) {
      std::string line;
      std::getline(times, line);

      if (line.size() < 20 || line[0] == '#') continue;

      {
        std::string timestamp_str = line.substr(0, 19);
        std::istringstream ss(timestamp_str);
        Timestamp timestamp;
        ss >> timestamp;
        timestamps_frame.push_back(timestamp);
      }
      id++;
    }
  }

  Mat image1;
  Mat image2;
  for (const auto& t : timestamps_frame) {
    std::string img1_path =
        dataset_path + "/mav0/cam0/data/" + std::to_string(t) + ".png";
    std::string img2_path =
        dataset_path + "/mav0/cam1/data/" + std::to_string(t) + ".png";
    image1 = imread(img1_path, 1);
    image2 = imread(img2_path, 1);
    if (!image1.data) {
      printf("No image data \n");
      return -1;
    }
    namedWindow("Cam0", WINDOW_AUTOSIZE);
    imshow("Cam0", image1);
    namedWindow("Cam1", WINDOW_AUTOSIZE);
    imshow("Cam1", image2);
    int keyboard = waitKey(30);
    if (keyboard == 'q' || keyboard == 27) return -1;
  }

  return 0;
}