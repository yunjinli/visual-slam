#include <gtest/gtest.h>
#include <visnav/imu_dataset.h>

using namespace visnav;

TEST(ImuTestSuite, LoadImuData) {
  std::string dataset_type = "euroc";
  std::string dataset_path = "../../data/euro_data/MH_01_easy";
  DatasetIoInterfacePtr dataset_io =
      DatasetIoFactory::getDatasetIo(dataset_type);
  dataset_io->read(dataset_path);
  Eigen::Vector3d test_acc_data =
      dataset_io->get_data()->get_accel_data()[0].data;
  Timestamp test_acc_ts =
      dataset_io->get_data()->get_accel_data()[0].timestamp_ns;
  Timestamp test_gyro_ts =
      dataset_io->get_data()->get_gyro_data()[0].timestamp_ns;
  ASSERT_EQ(test_acc_ts, test_gyro_ts);
  ASSERT_EQ(test_acc_ts, 1403636579758555392);
  Eigen::Vector3d test_gyro_data =
      dataset_io->get_data()->get_gyro_data()[0].data;
  //  std::cout << test_acc_data[0] << "\n";
  ASSERT_EQ(test_acc_data[0], 8.1476917083333333);
  ASSERT_EQ(test_acc_data[1], -0.37592158333333331);
  ASSERT_EQ(test_acc_data[2], -2.4026292499999999);

  ASSERT_EQ(test_gyro_data[0], -0.099134701513277898);
  ASSERT_EQ(test_gyro_data[1], 0.14730578886832138);
  ASSERT_EQ(test_gyro_data[2], 0.02722713633111154);
  ASSERT_EQ(dataset_io->get_data()->get_accel_data().size(), 36821 - 1);
}
