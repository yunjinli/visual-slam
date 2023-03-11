#include <memory>
#include <sophus/se3.hpp>
#include <unordered_map>
#include <vector>

#pragma once

namespace visnav {

class Dataset {
 public:
  virtual ~Dataset(){};

  virtual size_t get_num_cams() const = 0;

  virtual std::vector<int64_t> &get_image_timestamps() = 0;

  virtual const std::vector<int64_t> &get_gt_timestamps() const = 0;
  virtual const std::vector<Sophus::SE3d,
                            Eigen::aligned_allocator<Sophus::SE3d>>
      &get_gt_pose_data() const = 0;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef std::shared_ptr<Dataset> DatasetPtr;

class DatasetIoInterface {
 public:
  virtual void read(const std::string &path) = 0;
  virtual void reset() = 0;
  virtual DatasetPtr get_data() = 0;

  virtual ~DatasetIoInterface(){};
};

typedef std::shared_ptr<DatasetIoInterface> DatasetIoInterfacePtr;

class DatasetIoFactory {
 public:
  static DatasetIoInterfacePtr getDatasetIo(const std::string &dataset_type);
};

}  // namespace visnav
