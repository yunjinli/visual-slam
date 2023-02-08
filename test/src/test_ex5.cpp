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

#include <fstream>

#include "visnav/map_utils.h"
#include "visnav/vo_utils.h"

using namespace visnav;

const int MATCH_THRESHOLD = 70;
const double DIST_2_BEST = 1.2;

const std::string calib_path = "../../test/ex4_test_data/calib.json";

const std::string map_localize_path =
    "../../test/ex4_test_data/map_localize_camera.cereal";

const std::string added_landmarks_path =
    "../../test/ex5_test_data/added_landmarks.cereal";

const std::string landmark_match_data_path =
    "../../test/ex5_test_data/landmark_match_data.cereal";

const std::string matches_path = "../../test/ex5_test_data/matches.cereal";
const std::string projections_path =
    "../../test/ex5_test_data/projections.cereal";

void load_calib(const std::string& calib_path, Calibration& calib_cam) {
  std::ifstream os(calib_path, std::ios::binary);

  if (os.is_open()) {
    cereal::JSONInputArchive archive(os);
    archive(calib_cam);
  } else {
    ASSERT_TRUE(false) << "could not load camera ";
  }
}

void test_pose_equal(const Sophus::SE3d& pose_ref, const Sophus::SE3d& pose,
                     const double epsilon = 1e-6) {
  const Sophus::SE3d diff = pose_ref.inverse() * pose;
  EXPECT_LE(diff.log().norm(), epsilon);
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

TEST(Ex5TestSuite, ProjectLandmarks) {
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

  const FrameCamId fcid(3, 0);
  Sophus::SE3d current_pose = cameras[fcid].T_w_c;
  current_pose.translation() += Eigen::Vector3d(0.05, -0.03, 0.06);
  current_pose.so3() *= Sophus::SO3d::exp(Eigen::Vector3d(-0.02, 0.01, -0.03));

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> pp,
      pp_loaded;
  std::vector<TrackId> ptid, ptid_loaded;

  {
    std::ifstream os(projections_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(pp_loaded);
    archive(ptid_loaded);
  }

  // We want the test to fail if points are not using camera pose correctly to
  // transform into camera frame before doing z check.

  // It fails if we don't use current_pose at all.
  // It fails if camera_pose is used (T_w_c)
  // It fails if only rotational part of camera_pose is used
  // It fails if only translational part of camera_pose is used

  const double cam_z_threshold = 0.1;
  project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                    cam_z_threshold, pp, ptid);

  ASSERT_EQ(pp_loaded.size(), pp.size());
  ASSERT_EQ(ptid_loaded.size(), ptid.size());

  std::map<TrackId, Eigen::Vector2d> loaded_map;

  for (size_t i = 0; i < ptid_loaded.size(); i++) {
    loaded_map[ptid_loaded[i]] = pp_loaded[i];
  }

  for (size_t i = 0; i < pp_loaded.size(); i++) {
    Eigen::Vector2d ref = loaded_map[ptid[i]];
    EXPECT_TRUE(ref.isApprox(pp[i])) << "ref: " << ref << ", pp: " << pp[i];
  }
}

TEST(Ex5TestSuite, FindMatchesLandmarks) {
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

  MatchData md_loaded;

  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  // Note: This was saved as std::set<std::pair<FeatureId, FeatureId>>
  // so we need to keep loading it like this, even if md.matches now contains
  // std::vector<std::pair<FeatureId, TrackId>>
  std::set<std::pair<FeatureId, FeatureId>> match_set_loaded(
      md_loaded.matches.begin(), md_loaded.matches.end());

  const FrameCamId fcid(3, 0);
  const Sophus::SE3d current_pose = Sophus::SE3d(
      cameras[fcid].T_w_c.so3() *
          Sophus::SO3d::exp(Eigen::Vector3d(-0.02, 0.01, -0.03)),
      cameras[fcid].T_w_c.translation() + Eigen::Vector3d(0.05, -0.03, 0.06));

  const double cam_z_threshold = 0.1;
  const double match_max_dist_2d = 20.0;

  // first check with the camera already being in the map (it should also work
  // in that case)
  {
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;
    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    KeypointsData kdl = feature_corners[fcid];
    LandmarkMatchData md;

    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           MATCH_THRESHOLD, DIST_2_BEST, md);

    std::set<std::pair<FeatureId, FeatureId>> match_set(md.matches.begin(),
                                                        md.matches.end());

    EXPECT_EQ(match_set_loaded.size(), match_set.size());

    for (const auto& pair : match_set_loaded) {
      EXPECT_TRUE(match_set.count(pair) > 0)
          << "Match (" << pair.first << "," << pair.second
          << ") from md_loaded.matches is missing in md.matches.";
    }

    for (const auto& pair : match_set) {
      EXPECT_TRUE(match_set_loaded.count(pair) > 0)
          << "Match (" << pair.first << "," << pair.second
          << ") in md.matches is too much (not present in md_loaded.matches).";
    }
  }

  // now remove the camera from the map and check again without the camera in
  // the map (the set of matches we expect is smaller in that case)
  {
    // the camera, observations, and expected matches that we know we don't get
    // in that case
    cameras.erase(fcid);
    for (auto& kv : landmarks) {
      kv.second.obs.erase(fcid);
    }
    for (auto p : {std::pair<FeatureId, FeatureId>({32, 1732}),
                   {36, 1732},
                   {65, 1787},
                   {89, 571},
                   {193, 227}}) {
      match_set_loaded.erase(p);
    }

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;
    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    KeypointsData kdl = feature_corners[fcid];
    LandmarkMatchData md;

    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           MATCH_THRESHOLD, DIST_2_BEST, md);

    std::set<std::pair<FeatureId, FeatureId>> match_set(md.matches.begin(),
                                                        md.matches.end());

    EXPECT_EQ(match_set_loaded.size(), match_set.size());

    for (const auto& pair : match_set_loaded) {
      EXPECT_TRUE(match_set.count(pair) > 0)
          << "Match (" << pair.first << "," << pair.second
          << ") from md_loaded.matches is missing in md.matches.";
    }

    for (const auto& pair : match_set) {
      EXPECT_TRUE(match_set_loaded.count(pair) > 0)
          << "Match (" << pair.first << "," << pair.second
          << ") in md.matches is too much (not present in md_loaded.matches).";
    }
  }
}

TEST(Ex5TestSuite, LocalizeCamera) {
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
  Sophus::SE3d T_w2_w1(Sophus::SO3d::exp(tangent_rot_w2_w1), tangent_t_w2_w1);

  for (auto& kv : landmarks) {
    kv.second.p = T_w2_w1 * kv.second.p;
  }
  for (auto& kv : cameras) {
    kv.second.T_w_c = T_w2_w1 * kv.second.T_w_c;
  }

  const FrameCamId fcid(3, 0);
  Sophus::SE3d current_pose = cameras[fcid].T_w_c;
  current_pose.translation() += Eigen::Vector3d(0.05, -0.03, 0.06);
  current_pose.so3() *= Sophus::SO3d::exp(Eigen::Vector3d(-0.02, 0.01, -0.03));

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      projected_points;
  std::vector<TrackId> projected_track_ids;

  const double cam_z_threshold = 0.1;
  project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                    cam_z_threshold, projected_points, projected_track_ids);

  KeypointsData kdl = feature_corners[fcid];

  double match_max_dist_2d = 50.0;

  LandmarkMatchData md;
  find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                         projected_track_ids, match_max_dist_2d,
                         MATCH_THRESHOLD, DIST_2_BEST, md);

  // std::cerr << "matches " << md.matches.size() << std::endl;

  const double reprojection_error_pnp_inlier_threshold_pixel = 3.0;

  localize_camera(Sophus::SE3d(), calib_cam.intrinsics[0], kdl, landmarks,
                  reprojection_error_pnp_inlier_threshold_pixel, md);

  // std::cerr << "inliers " << md.inliers.size() << std::endl;
  // std::cerr << "diff " << (cameras.at(fcid).T_w_c.inverse() *
  // md.T_w_c).log().norm() << std::endl;

  test_pose_equal(cameras.at(fcid).T_w_c, md.T_w_c, 2e-2);
}

TEST(Ex5TestSuite, AddNewLandmarks) {
  Calibration calib_cam;
  Corners feature_corners;
  Matches feature_matches;
  FeatureTracks feature_tracks;
  FeatureTracks outlier_tracks;
  Cameras cameras, cameras_ref;
  Landmarks landmarks, landmarks_ref;

  load_calib(calib_path, calib_cam);

  load_map_file(map_localize_path, feature_corners, feature_matches,
                feature_tracks, outlier_tracks, cameras, landmarks);

  // Cameras:
  // 0_0 0_1 1_0 1_1 2_0 2_1 3_0
  const FrameCamId fcidl(3, 0);
  const FrameCamId fcidr(3, 1);

  // Remove all observations for those 2 frames, so we can test that they are
  // added again
  std::set<TrackId> lm_empty;
  for (auto& kv_lm : landmarks) {
    kv_lm.second.obs.erase(fcidl);
    kv_lm.second.obs.erase(fcidr);

    if (kv_lm.second.obs.empty()) {
      lm_empty.emplace(kv_lm.first);
    }
  }

  for (const TrackId tid : lm_empty) {
    landmarks.erase(tid);
  }

  const KeypointsData& kdl = feature_corners[fcidl];
  const KeypointsData& kdr = feature_corners[fcidr];
  const MatchData& md_stereo = feature_matches[std::make_pair(fcidl, fcidr)];

  LandmarkMatchData md_loaded;
  {
    std::ifstream os(landmark_match_data_path, std::ios::binary);
    if (os.is_open()) {
      cereal::BinaryInputArchive archive(os);
      archive(md_loaded);
    }
  }

  int landmarks_prev_size = int(landmarks.size());
  // set distinguishable landmark id for new landmarks
  const TrackId offset = 100000;
  TrackId next_landmark_id = offset + landmarks_prev_size;

  add_new_landmarks(fcidl, fcidr, kdl, kdr, calib_cam, md_stereo, md_loaded,
                    landmarks, next_landmark_id);

  {
    std::ifstream os(added_landmarks_path, std::ios::binary);
    if (os.is_open()) {
      cereal::BinaryInputArchive archive(os);
      archive(landmarks_ref);
    }
  }

  std::vector<std::pair<TrackId, Landmark>> new_landmarks, new_landmarks_ref;
  Landmarks old_landmarks, old_landmarks_ref;

  for (const auto& kv_lm : landmarks) {
    if (kv_lm.first >= offset + landmarks_prev_size) {
      // landmark with new id
      new_landmarks.emplace_back(kv_lm);
    } else {
      old_landmarks[kv_lm.first] = kv_lm.second;
    }
  }

  for (const auto& kv_lm : landmarks_ref) {
    if (kv_lm.first >= offset + landmarks_prev_size) {
      // landmark with new id
      new_landmarks_ref.emplace_back(kv_lm);
    } else {
      old_landmarks_ref[kv_lm.first] = kv_lm.second;
    }
  }

  test_landmarks_equal(old_landmarks_ref, old_landmarks);

  // Don't test these (empirically determined) inaccurate landmarks to make sure
  // implementations using both opengv::triangulation::triangulate() and
  // opengv::triangulation::triangulate2() pass the test.
  std::set<TrackId> skip_landmarks = {100257, 100250, 100246};

  // The track_ids of new added landmarks might differ if they are added in a
  // differen order. However, we can sort newly added landmarks by their
  // observation list and compare, since the observations should be
  // deterministic.
  auto order = [](auto& a, auto& b) { return a.second.obs < b.second.obs; };
  std::sort(new_landmarks.begin(), new_landmarks.end(), order);
  std::sort(new_landmarks_ref.begin(), new_landmarks_ref.end(), order);

  ASSERT_EQ(new_landmarks_ref.size(), new_landmarks.size());
  for (int64_t i = 0; i < int64_t(new_landmarks.size()); i++) {
    if (skip_landmarks.count(new_landmarks_ref[i].first)) continue;
    test_landmark_equal(new_landmarks_ref[i].second, new_landmarks[i].second,
                        new_landmarks_ref[i].first);
  }

  // Note: it catches the mistake when one rewrites observation of left camera
  // by right camera's feature_id, and not adding right camera into obs.
  // It also catches the mistake when one does not transform new triangulated
  // landmarks into world frame. (or uses only translation or only rotation
  // for transform)
}

TEST(Ex5TestSuite, RemoveOldKFs) {
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

  Landmarks landmarks_loaded, old_landmarks;
  landmarks_loaded = landmarks;

  std::set<FrameId> kf_frames;
  for (const auto& kv : cameras) {
    kf_frames.emplace(kv.first.frame_id);
  }

  const FrameCamId fcid(kf_frames.size(), 0);

  // std::cerr << "kf_frames.size() " << kf_frames.size() << std::endl;

  int max_num_kfs = 2;
  remove_old_keyframes(fcid, max_num_kfs, cameras, landmarks, old_landmarks,
                       kf_frames);

  // Check that we have right number of kfs.
  EXPECT_EQ(max_num_kfs, int(kf_frames.size()));

  // All cameras are in kf_frames
  for (const auto& kv : cameras) {
    EXPECT_TRUE(kf_frames.find(kv.first.frame_id) != kf_frames.end());
  }

  EXPECT_EQ(old_landmarks.size() + landmarks.size(), landmarks_loaded.size());

  // Check that there are no observations from removed cameras.
  for (const auto& kv : landmarks) {
    for (const auto& obs_kv : kv.second.obs) {
      EXPECT_TRUE(kf_frames.find(obs_kv.first.frame_id) != kf_frames.end());
    }
  }
}
