/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <gtest/gtest.h>

#include "visnav/map_utils.h"

using namespace visnav;

const std::string calib_path = "../../test/ex4_test_data/calib.json";

const std::string calib_path_optimized =
    "../../test/ex4_test_data/calib_optimized.json";

const std::string map_triangulate_path =
    "../../test/ex4_test_data/map_landmark_triangulation.cereal";

const std::string map_init_path =
    "../../test/ex4_test_data/map_initialization_pre_opt.cereal";

const std::string map_localize_path =
    "../../test/ex4_test_data/map_localize_camera.cereal";

const std::string map_localize_path_noisy =
    "../../test/ex4_test_data/map_localize_camera_noisy.cereal";

const std::string map_optimized_path =
    "../../test/ex4_test_data/map_localize_camera_optimized.cereal";

const std::string map_optimized_path_noisy =
    "../../test/ex4_test_data/map_localize_camera_optimized_noisy.cereal";

const std::string map_optimized_path_wo_huber =
    "../../test/ex4_test_data/map_localize_camera_optimized_wo_huber.cereal";

const std::string map_optimized_path_huber_param =
    "../../test/ex4_test_data/map_localize_camera_optimized_huber_param.cereal";

const std::string map_optimized_path_fixed_cameras =
    "../../test/ex4_test_data/"
    "map_localize_camera_optimized_fixed_cameras.cereal";

const std::string map_optimized_path_intrinsics =
    "../../test/ex4_test_data/map_localize_camera_optimized_intrinsics.cereal";

void load_calib(const std::string& path, Calibration& calib_cam) {
  std::ifstream os(path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(calib_cam);
  } else {
    ASSERT_TRUE(false) << "could not load camera ";
  }
}

void save_calib(const std::string& path, const Calibration& calib_cam) {
  std::ofstream os(path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONOutputArchive archive(os);
    archive(calib_cam);
  } else {
    ASSERT_TRUE(false) << "could not save camera ";
  }
}

void test_calibrations_equal(const Calibration& calib_ref,
                             const Calibration& calib) {
  ASSERT_EQ(calib_ref.T_i_c.size(), calib.T_i_c.size());
  ASSERT_EQ(calib_ref.intrinsics.size(), calib.intrinsics.size());

  for (uint32_t i = 0; i < calib_ref.T_i_c.size(); i++) {
    const Sophus::SE3d diff = calib_ref.T_i_c[i].inverse() * calib.T_i_c[i];
    EXPECT_LE(diff.log().norm(), 1e-6)
        << "Pose of camera " << i << " inaccurate.";
  }
  for (uint32_t i = 0; i < calib_ref.intrinsics.size(); i++) {
    EXPECT_EQ(calib_ref.intrinsics[i]->name(), calib.intrinsics[i]->name());
    EXPECT_EQ(calib_ref.intrinsics[i]->height(), calib.intrinsics[i]->height());
    EXPECT_EQ(calib_ref.intrinsics[i]->width(), calib.intrinsics[i]->width());
    const auto& intr_ref = calib_ref.intrinsics[i]->getParam();
    const auto& intr = calib.intrinsics[i]->getParam();
    EXPECT_LE((intr_ref - intr).norm(), 1e-5)
        << "Intrinsics of `" << calib_ref.intrinsics[i]->name()
        << "` camera number " << i
        << " inaccurate:\n`intrinsics_ref` = " << std::setprecision(5)
        << intr_ref.transpose() << "\n`intrinsics_opt` = " << intr.transpose();
  }
}

void test_cameras_equal(const Cameras& cameras_ref, const Cameras& cameras,
                        double threshold = 1e-6) {
  ASSERT_EQ(cameras_ref.size(), cameras.size());
  for (const auto& kv : cameras_ref) {
    ASSERT_TRUE(cameras.count(kv.first) > 0)
        << "Expected camera " << kv.first.frame_id << "_" << kv.first.cam_id
        << " in map.";
    const Sophus::SE3d diff =
        kv.second.T_w_c.inverse() * cameras.at(kv.first).T_w_c;
    EXPECT_LE(diff.log().norm(), threshold)
        << "Pose of camera " << kv.first.frame_id << "_" << kv.first.cam_id
        << " inaccurate.";
  }
}

void test_landmark_equal(const Landmark& lm_ref, const Landmark& lm,
                         const TrackId lm_id) {
  const double lm_dist = (lm_ref.p - lm.p).norm();
  EXPECT_LE(lm_dist, 0.05) << "Position of lm " << lm_id << " inaccurate";
  EXPECT_EQ(lm_ref.obs, lm.obs)
      << "List of observations for lm " << lm_id << " doesn't match.";
}

void test_landmarks_equal(const Landmarks& landmarks_ref,
                          const Landmarks& landmarks) {
  ASSERT_EQ(landmarks_ref.size(), landmarks.size());
  for (const auto& kv : landmarks_ref) {
    ASSERT_TRUE(landmarks.count(kv.first) > 0);
    test_landmark_equal(kv.second, landmarks.at(kv.first), kv.first);
  }
}

TEST(Ex4TestSuite, Triangulate) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras;
  Landmarks landmarks, landmarks_ref;

  load_calib(calib_path, calib_cam);

  load_map_file(map_triangulate_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks_ref);

  const FrameCamId fcid0(0, 1);
  const FrameCamId fcid1(1, 1);

  // We want this test to fail if the conversion of triangulated points into
  // world-frame doesn't use the rotation. I.e. there needs to be rotation on
  // the cam0 pose. Hence we change the world-frame to world-frame-2, i.e.
  // rotate all landmarks and all camera positions by T_w2_w1.
  Eigen::Vector3d tangent_rot_w2_w1;
  tangent_rot_w2_w1 << 0.1, -0.5, 0.3;
  tangent_rot_w2_w1.normalize();
  Eigen::Vector3d tangent_t_w2_w1;
  tangent_t_w2_w1 << 1.2, -0.7, 0.1;
  Sophus::SE3d T_w2_w1(Sophus::SO3d::exp(tangent_rot_w2_w1), tangent_t_w2_w1);

  for (auto& kv : landmarks_ref) {
    kv.second.p = T_w2_w1 * kv.second.p;
  }
  for (auto& kv : cameras) {
    kv.second.T_w_c = T_w2_w1 * kv.second.T_w_c;
  }

  // copy landmarks and remove landmarks shared by the two cameras
  landmarks = landmarks_ref;
  for (const auto& kv : landmarks_ref) {
    if (kv.second.obs.count(fcid0) > 0 && kv.second.obs.count(fcid1) > 0) {
      landmarks.erase(kv.first);
    }
  }

  add_new_landmarks_between_cams(fcid0, fcid1, calib_cam, feature_corners,
                                 feature_tracks, cameras, landmarks);

  // The followign test fails if the implementation doesn't filter cameras that
  // are not yet in the map, i.e. `lm.obs` will contain larger set of cameras
  // for some landmarks `lm`. This assert checks it explicitly, however later
  // `test_landmarks_equal` also ensures that the lm.obs == lm_ref.obs.
  for (const auto& kv : landmarks) {
    for (const auto& FrameCamIdFeatureId : kv.second.obs) {
      ASSERT_TRUE(cameras.count(FrameCamIdFeatureId.first) > 0)
          << "Cameras not yet in the map were added for landmark " << kv.first;
    }
  }

  test_landmarks_equal(landmarks_ref, landmarks);
}

TEST(Ex4TestSuite, InitializeScene) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras, cameras_ref;
  Landmarks landmarks, landmarks_ref;

  load_calib(calib_path, calib_cam);

  load_map_file(map_init_path, feature_corners, feature_matches, feature_tracks,
                outlier_tracks, cameras_ref, landmarks_ref);

  const FrameCamId fcid0(0, 0);
  const FrameCamId fcid1(0, 1);

  initialize_scene_from_stereo_pair(fcid0, fcid1, calib_cam, feature_corners,
                                    feature_tracks, cameras, landmarks);

  test_cameras_equal(cameras_ref, cameras);
  test_landmarks_equal(landmarks_ref, landmarks);
}

TEST(Ex4TestSuite, LocalizeCamera) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras;
  Landmarks landmarks;

  load_calib(calib_path, calib_cam);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  // We want the test to fail if the adapter pose is not set before non-linear
  // opt. So we need a test case where identity is not a good enough
  // initialization. Solution: change world-frame to world-frame-2, i.e. rotate
  // all landmarks and all camera positions by T_w2_w1.

  Eigen::Vector3d tangent_rot_w2_w1;
  tangent_rot_w2_w1 << 0.1, -0.5, 0.3;
  tangent_rot_w2_w1.normalize();
  tangent_rot_w2_w1 *= 2.5;
  Eigen::Vector3d tangent_t_w2_w1;
  tangent_t_w2_w1 << 1.2, -0.7, 0.1;
  Sophus::SE3d T_w2_w1(Sophus::SO3d::exp(tangent_rot_w2_w1).matrix(),
                       tangent_t_w2_w1);

  for (auto& kv : landmarks) {
    kv.second.p = T_w2_w1 * kv.second.p;
  }
  for (auto& kv : cameras) {
    kv.second.T_w_c = T_w2_w1 * kv.second.T_w_c;
  }

  const FrameCamId fcid(3, 0);

  std::vector<TrackId> shared_tracks;
  GetSharedTracks(fcid, feature_tracks, landmarks, shared_tracks);

  for (int i = 0; i < 2; ++i) {
    // do these tests twice; the second time around we remove the camera from
    // the map, since some people implement localize_camera such that it fails
    // for non-exisitng cameras (which is the whole point, but ok...)
    if (i == 1) {
      for (auto& kv : landmarks) {
        kv.second.obs.erase(fcid);
        kv.second.outlier_obs.erase(fcid);
      }
    }

    {
      const double reprojection_error_pnp_inlier_threshold_pixel = 3.0;
      std::vector<TrackId> inlier_track_ids;
      Sophus::SE3d T_w_c;
      localize_camera(fcid, shared_tracks, calib_cam, feature_corners,
                      feature_tracks, landmarks,
                      reprojection_error_pnp_inlier_threshold_pixel, T_w_c,
                      inlier_track_ids);

      std::set<TrackId> shared_set(shared_tracks.begin(), shared_tracks.end());
      std::set<TrackId> inlier_set(inlier_track_ids.begin(),
                                   inlier_track_ids.end());

      // ensure the inliers are a subset of the shared_track_ids
      for (auto tid : inlier_set) {
        EXPECT_TRUE(shared_set.count(tid))
            << "inlier_track_ids should be a subset of shared_track_ids";
      }

      // check expected inlier count and some known outliers
      EXPECT_EQ(151, shared_tracks.size());
      EXPECT_GE(145, inlier_track_ids.size());
      EXPECT_LE(120, inlier_track_ids.size());
      // some true outlier matches:
      EXPECT_FALSE(inlier_set.count(2114) > 0);
      EXPECT_FALSE(inlier_set.count(1260) > 0);
      // some badly localized landmarks:
      EXPECT_FALSE(inlier_set.count(1866) > 0);
      EXPECT_FALSE(inlier_set.count(1158) > 0);
      EXPECT_FALSE(inlier_set.count(674) > 0);

      const Sophus::SE3d diff = cameras.at(fcid).T_w_c.inverse() * T_w_c;
      EXPECT_LE(diff.log().norm(), 0.02)
          << "Pose of camera " << fcid.frame_id << "_" << fcid.cam_id
          << " inaccurate.";
    }
  }
}

TEST(Ex4TestSuite, BundleAdjustment) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras, cameras_ref;
  Landmarks landmarks, landmarks_ref;

  load_calib(calib_path, calib_cam);

  load_map_file(map_optimized_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras_ref, landmarks_ref);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = false;
  ba_options.use_huber = true;
  ba_options.huber_parameter = 1.0;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = 0;
  std::set<FrameCamId> fixed_cameras = {{0, 0}, {0, 1}};

  // Note: This test passes even with ceres::QuaternionParameterization
  // (together with ProductParameterization), even though that is a bit strange
  // (ceres::EigenQuaternionParameterization would make more sense). It seems to
  // even work without any local parameterization... probably because the BA
  // problem is not big enough to show a difference.

  bundle_adjustment(feature_corners, ba_options, fixed_cameras, calib_cam,
                    cameras, landmarks);

  test_cameras_equal(cameras_ref, cameras);
  test_landmarks_equal(landmarks_ref, landmarks);
}

TEST(Ex4TestSuite, BundleAdjustmentNoisy) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras, cameras_ref;
  Landmarks landmarks, landmarks_ref;

  load_calib(calib_path, calib_cam);

  load_map_file(map_optimized_path_noisy, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras_ref, landmarks_ref);

  load_map_file(map_localize_path_noisy, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = false;
  ba_options.use_huber = true;
  ba_options.huber_parameter = 1.0;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = 0;
  std::set<FrameCamId> fixed_cameras = {{0, 0}, {0, 1}};

  // This test was added because with we want to make the test fail when
  // ceres::QuaternionParameterization is used, or when no local
  // parameterization is used. But it even fails for
  // ceres::ProductParameterization( new
  // ceres::EigenQuaternionParameterization(), new
  // ceres::IdentityParameterization(3)), while it should probably work. It
  // seems now the result highly depends on local parameterization (problem
  // might have outliers, and therefore not stable).
  // Let's keep an eye on it to see if that causes any problems.

  bundle_adjustment(feature_corners, ba_options, fixed_cameras, calib_cam,
                    cameras, landmarks);

  test_cameras_equal(cameras_ref, cameras, 1e-5);
  // test_landmarks_equal(landmarks_ref, landmarks);
}

TEST(Ex4TestSuite, BundleAdjustmentWOHuber) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras, cameras_ref;
  Landmarks landmarks, landmarks_ref;

  load_calib(calib_path, calib_cam);

  load_map_file(map_optimized_path_wo_huber, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras_ref, landmarks_ref);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = false;
  // Without huber loss
  ba_options.use_huber = false;
  ba_options.huber_parameter = 1.0;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = 0;
  std::set<FrameCamId> fixed_cameras = {{0, 0}, {0, 1}};

  bundle_adjustment(feature_corners, ba_options, fixed_cameras, calib_cam,
                    cameras, landmarks);

  test_cameras_equal(cameras_ref, cameras);
  test_landmarks_equal(landmarks_ref, landmarks);
}

TEST(Ex4TestSuite, BundleAdjustmentHuberParam) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras, cameras_ref;
  Landmarks landmarks, landmarks_ref;

  load_calib(calib_path, calib_cam);

  load_map_file(map_optimized_path_huber_param, feature_corners,
                feature_matches, feature_tracks, outlier_tracks, cameras_ref,
                landmarks_ref);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = false;
  ba_options.use_huber = true;
  // Change huber param
  ba_options.huber_parameter = 2.0;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = 0;
  std::set<FrameCamId> fixed_cameras = {{0, 0}, {0, 1}};

  bundle_adjustment(feature_corners, ba_options, fixed_cameras, calib_cam,
                    cameras, landmarks);

  test_cameras_equal(cameras_ref, cameras);
  test_landmarks_equal(landmarks_ref, landmarks);
}

TEST(Ex4TestSuite, BundleAdjustmentFixedCameras) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras, cameras_ref;
  Landmarks landmarks, landmarks_ref;

  load_calib(calib_path, calib_cam);

  load_map_file(map_optimized_path_fixed_cameras, feature_corners,
                feature_matches, feature_tracks, outlier_tracks, cameras_ref,
                landmarks_ref);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = false;
  ba_options.use_huber = true;
  ba_options.huber_parameter = 1.0;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = 0;
  // Change fixed cameras
  std::set<FrameCamId> fixed_cameras = {{0, 0}, {1, 0}};

  bundle_adjustment(feature_corners, ba_options, fixed_cameras, calib_cam,
                    cameras, landmarks);

  test_cameras_equal(cameras_ref, cameras);
  test_landmarks_equal(landmarks_ref, landmarks);
}

TEST(Ex4TestSuite, BundleAdjustmentIntrinsics) {
  Calibration calib_cam, calib_cam_ref;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras, cameras_ref;
  Landmarks landmarks, landmarks_ref;

  load_calib(calib_path, calib_cam);
  load_calib(calib_path_optimized, calib_cam_ref);

  load_map_file(map_optimized_path_intrinsics, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras_ref, landmarks_ref);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  BundleAdjustmentOptions ba_options;
  // Optimize intrinsics too
  ba_options.optimize_intrinsics = true;
  ba_options.use_huber = true;
  ba_options.huber_parameter = 1.0;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = 0;
  std::set<FrameCamId> fixed_cameras = {{0, 0}, {0, 1}};

  bundle_adjustment(feature_corners, ba_options, fixed_cameras, calib_cam,
                    cameras, landmarks);

  test_calibrations_equal(calib_cam, calib_cam_ref);
  //  test_cameras_equal(cameras_ref, cameras);
  //  test_landmarks_equal(landmarks_ref, landmarks);
}
