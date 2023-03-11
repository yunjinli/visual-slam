#include <io/dataset_io.h>

#include <fstream>

namespace visnav {
class EurocDataset : public Dataset {
  size_t num_cams;

  std::string path;

  std::vector<int64_t> image_timestamps;
  std::unordered_map<int64_t, std::string> image_path;

  std::vector<int64_t> gt_timestamps;  // ordered gt timestamps
  std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>
      gt_pose_data;

 public:
  ~EurocDataset(){};

  size_t get_num_cams() const { return num_cams; }

  std::vector<int64_t> &get_image_timestamps() { return image_timestamps; }

  const std::vector<int64_t> &get_gt_timestamps() const {
    return gt_timestamps;
  }
  const std::vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>>
      &get_gt_pose_data() const {
    return gt_pose_data;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  friend class EurocIO;
};

class EurocIO : public DatasetIoInterface {
 public:
  EurocIO() {}

  void read(const std::string &path) {
    std::ifstream os(path, std::ios::binary);
    if (!os.is_open()) {
      std::cerr << "No dataset found in " << path << std::endl;
    }
    data.reset(new EurocDataset);

    data->num_cams = 2;
    data->path = path;

    read_image_timestamps(path + "/cam0/");

    std::ifstream gt_states(path + "/state_groundtruth_estimate0/data.csv",
                            std::ios::binary);
    std::ifstream gt_poses(path + "/gt/data.csv", std::ios::binary);
    if (gt_states.is_open()) {
      read_gt_data_state(path + "/state_groundtruth_estimate0/");
    } else if (gt_poses.is_open()) {
      read_gt_data_pose(path + "/gt/");
    }
  }

  void reset() { data.reset(); }

  DatasetPtr get_data() { return data; }

 private:
  void read_image_timestamps(const std::string &path) {
    std::ifstream f(path + "data.csv");
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;
      std::stringstream ss(line);
      char tmp;
      int64_t t_ns;
      std::string path;
      ss >> t_ns >> tmp >> path;

      data->image_timestamps.emplace_back(t_ns);
      data->image_path[t_ns] = path;
    }
  }

  void read_gt_data_state(const std::string &path) {
    data->gt_timestamps.clear();
    data->gt_pose_data.clear();

    std::ifstream f(path + "data.csv");
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;

      std::stringstream ss(line);

      char tmp;
      uint64_t timestamp;
      Eigen::Quaterniond q;
      Eigen::Vector3d pos, vel, accel_bias, gyro_bias;

      ss >> timestamp >> tmp >> pos[0] >> tmp >> pos[1] >> tmp >> pos[2] >>
          tmp >> q.w() >> tmp >> q.x() >> tmp >> q.y() >> tmp >> q.z() >> tmp >>
          vel[0] >> tmp >> vel[1] >> tmp >> vel[2] >> tmp >> accel_bias[0] >>
          tmp >> accel_bias[1] >> tmp >> accel_bias[2] >> tmp >> gyro_bias[0] >>
          tmp >> gyro_bias[1] >> tmp >> gyro_bias[2];

      data->gt_timestamps.emplace_back(timestamp);
      data->gt_pose_data.emplace_back(q, pos);
    }
  }

  void read_gt_data_pose(const std::string &path) {
    data->gt_timestamps.clear();
    data->gt_pose_data.clear();

    std::ifstream f(path + "data.csv");
    std::string line;
    while (std::getline(f, line)) {
      if (line[0] == '#') continue;

      std::stringstream ss(line);

      char tmp;
      uint64_t timestamp;
      Eigen::Quaterniond q;
      Eigen::Vector3d pos;

      ss >> timestamp >> tmp >> pos[0] >> tmp >> pos[1] >> tmp >> pos[2] >>
          tmp >> q.w() >> tmp >> q.x() >> tmp >> q.y() >> tmp >> q.z();

      data->gt_timestamps.emplace_back(timestamp);
      data->gt_pose_data.emplace_back(q, pos);
    }
  }

  std::shared_ptr<EurocDataset> data;
};
}  // namespace visnav