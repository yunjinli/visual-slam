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

#pragma once

#include <fstream>
#include <thread>

#include <ceres/ceres.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

#include <visnav/reprojection.h>
#include <visnav/local_parameterization_se3.hpp>
#include <visnav/imu/preintegration.h>

#include <visnav/tracks.h>

namespace visnav {

// save map with all features and matches
void save_map_file(const std::string& map_path, const Corners& feature_corners,
                   const Matches& feature_matches,
                   const FeatureTracks& feature_tracks,
                   const FeatureTracks& outlier_tracks, const Cameras& cameras,
                   const Landmarks& landmarks) {
  {
    std::ofstream os(map_path, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Saved map as " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to save map as " << map_path << std::endl;
    }
  }
}

// load map with all features and matches
void load_map_file(const std::string& map_path, Corners& feature_corners,
                   Matches& feature_matches, FeatureTracks& feature_tracks,
                   FeatureTracks& outlier_tracks, Cameras& cameras,
                   Landmarks& landmarks) {
  {
    std::ifstream is(map_path, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);
      archive(feature_corners);
      archive(feature_matches);
      archive(feature_tracks);
      archive(outlier_tracks);
      archive(cameras);
      archive(landmarks);

      size_t num_obs = 0;
      for (const auto& kv : landmarks) {
        num_obs += kv.second.obs.size();
      }
      std::cout << "Loaded map from " << map_path << " (" << cameras.size()
                << " cameras, " << landmarks.size() << " landmarks, " << num_obs
                << " observations)" << std::endl;
    } else {
      std::cout << "Failed to load map from " << map_path << std::endl;
    }
  }
}

// Create new landmarks from shared feature tracks if they don't already exist.
// The two cameras must be in the map already.
// Returns the number of newly created landmarks.
int add_new_landmarks_between_cams(const FrameCamId& fcid0,
                                   const FrameCamId& fcid1,
                                   const Calibration& calib_cam,
                                   const Corners& feature_corners,
                                   const FeatureTracks& feature_tracks,
                                   const Cameras& cameras,
                                   Landmarks& landmarks) {
  // shared_track_ids will contain all track ids shared between the two images,
  // including existing landmarks
  std::vector<TrackId> shared_track_ids;

  // find shared feature tracks
  const std::set<FrameCamId> fcids = {fcid0, fcid1};
  if (!GetTracksInImages(fcids, feature_tracks, shared_track_ids)) {
    return 0;
  }

  // at the end of the function this will contain all newly added track ids
  std::vector<TrackId> new_track_ids;

  // TODO SHEET 4: Triangulate all new features and add to the map
  for (const auto& t_id : shared_track_ids) {
    if (landmarks.count(t_id) > 0) {
      // Landmark already exists
    } else {
      const auto& feature_track = feature_tracks.at(
          t_id);  // FeatureTrack = std::map<FrameCamId, FeatureId>;
      const auto& feature_id_fcid0 = feature_track.at(fcid0);
      const auto& feature_id_fcid1 = feature_track.at(fcid1);

      const auto& p2d_fcid0 =
          feature_corners.at(fcid0).corners[feature_id_fcid0];
      const auto& p2d_fcid1 =
          feature_corners.at(fcid1).corners[feature_id_fcid1];

      opengv::bearingVectors_t bearingVectors1;
      opengv::bearingVectors_t bearingVectors2;

      bearingVectors1.push_back(
          calib_cam.intrinsics[fcid0.cam_id]->unproject(p2d_fcid0));
      bearingVectors2.push_back(
          calib_cam.intrinsics[fcid1.cam_id]->unproject(p2d_fcid1));
      const auto& pose_fcid0 = cameras.at(fcid0).T_w_c;
      const auto& pose_fcid1 = cameras.at(fcid1).T_w_c;
      opengv::relative_pose::CentralRelativeAdapter adapter(
          bearingVectors1, bearingVectors2,
          (pose_fcid0.inverse() * pose_fcid1).translation(),
          (pose_fcid0.inverse() * pose_fcid1).rotationMatrix());
      //    opengv::point_t point = (pose_fcid1.inverse() * pose_fcid0) *
      //                            opengv::triangulation::triangulate(adapter,
      //                            0);
      opengv::point_t point =
          pose_fcid0 * opengv::triangulation::triangulate(adapter, 0);
      Landmark l;
      l.p = point;

      for (const auto& camera : cameras) {
        if (feature_track.count(camera.first) > 0) {  // fcid exist
          l.obs.emplace(
              std::make_pair(camera.first, feature_track.at(camera.first)));
        }
      }
      landmarks.emplace(std::make_pair(t_id, l));
      //      if (landmarks.count(t_id) > 0) {
      //        Landmark& l = landmarks.at(t_id);
      //        l.p = point;
      //        //      for (const auto& kv : feature_track) {
      //        //        l.obs.emplace(kv);
      //        //      }
      //        l.obs.clear();

      //        for (const auto& camera : cameras) {
      //          if (feature_track.count(camera.first) > 0) {  // fcid exist
      //            l.obs.emplace(
      //                std::make_pair(camera.first,
      //                feature_track.at(camera.first)));
      //          }
      //        }
      //      } else {
      //        Landmark l;
      //        l.p = point;

      //        for (const auto& camera : cameras) {
      //          if (feature_track.count(camera.first) > 0) {  // fcid exist
      //            l.obs.emplace(
      //                std::make_pair(camera.first,
      //                feature_track.at(camera.first)));
      //          }
      //        }
      //        landmarks.emplace(std::make_pair(t_id, l));
      //      }

      new_track_ids.push_back(t_id);
    }
  }

  return new_track_ids.size();
}

// Initialize the scene from a stereo pair, using the known transformation from
// camera calibration. This adds the inital two cameras and triangulates shared
// landmarks.
// Note: in principle we could also initialize a map from another images pair
// using the transformation from the pairwise matching with the 5-point
// algorithm. However, using a stereo pair has the advantage that the map is
// initialized with metric scale.
bool initialize_scene_from_stereo_pair(const FrameCamId& fcid0,
                                       const FrameCamId& fcid1,
                                       const Calibration& calib_cam,
                                       const Corners& feature_corners,
                                       const FeatureTracks& feature_tracks,
                                       Cameras& cameras, Landmarks& landmarks) {
  // check that the two image ids refer to a stereo pair
  if (!(fcid0.frame_id == fcid1.frame_id && fcid0.cam_id != fcid1.cam_id)) {
    std::cerr << "Images " << fcid0 << " and " << fcid1
              << " don't form a stereo pair. Cannot initialize." << std::endl;
    return false;
  }

  // TODO SHEET 4: Initialize scene (add initial cameras and landmarks)
  Camera camera0;
  Camera camera1;
  camera0.T_w_c = calib_cam.T_i_c[0];
  camera1.T_w_c = calib_cam.T_i_c[1];

  cameras.emplace(std::make_pair(fcid0, camera0));
  cameras.emplace(std::make_pair(fcid1, camera1));
  add_new_landmarks_between_cams(fcid0, fcid1, calib_cam, feature_corners,
                                 feature_tracks, cameras, landmarks);

  return true;
}

// Localize a new camera in the map given a set of observed landmarks. We use
// pnp and ransac to localize the camera in the presence of outlier tracks.
// After finding an inlier set with pnp, we do non-linear refinement using all
// inliers and also update the set of inliers using the refined pose.
//
// shared_track_ids already contains those tracks which the new image shares
// with the landmarks (but some might be outliers).
//
// We return the refined pose and the set of track ids for all inliers.
//
// The inlier threshold is given in pixels. See also the opengv documentation on
// how to convert this to a ransac threshold:
// http://laurentkneip.github.io/opengv/page_how_to_use.html#sec_threshold
void localize_camera(
    const FrameCamId& fcid, const std::vector<TrackId>& shared_track_ids,
    const Calibration& calib_cam, const Corners& feature_corners,
    const FeatureTracks& feature_tracks, const Landmarks& landmarks,
    const double reprojection_error_pnp_inlier_threshold_pixel,
    Sophus::SE3d& T_w_c, std::vector<TrackId>& inlier_track_ids) {
  inlier_track_ids.clear();

  // TODO SHEET 4: Localize a new image in a given map
  opengv::points_t points;
  opengv::bearingVectors_t bearingVectors;

  for (const auto& t_id : shared_track_ids) {
    points.push_back(landmarks.at(t_id).p);
    bearingVectors.push_back(calib_cam.intrinsics[fcid.cam_id]->unproject(
        feature_corners.at(fcid).corners[feature_tracks.at(t_id).at(fcid)]));
  }
  // create the central adapter
  opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors, points);
  // create a Ransac object
  opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      ransac;
  // create an AbsolutePoseSacProblem
  // (algorithm is selectable: KNEIP, GAO, or EPNP)
  std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter, opengv::sac_problems::absolute_pose::
                           AbsolutePoseSacProblem::KNEIP));
  // run ransac
  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ =
      1.0 - cos(atan(reprojection_error_pnp_inlier_threshold_pixel / 500.0));
  ransac.computeModel();
  // get the result
  opengv::transformation_t best_transformation = ransac.model_coefficients_;

  // Refinement using non-linear optimization
  adapter.sett(best_transformation.block(0, 3, 3, 1));
  adapter.setR(best_transformation.block(0, 0, 3, 3));
  opengv::transformation_t nonlinear_transformation =
      opengv::absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);
  T_w_c = Sophus::SE3<double>(nonlinear_transformation.block(0, 0, 3, 3),
                              nonlinear_transformation.block(0, 3, 3, 1));
  ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                          ransac.threshold_, ransac.inliers_);

  for (const auto& kv : ransac.inliers_) {
    inlier_track_ids.push_back(shared_track_ids[kv]);
  }
}

struct BundleAdjustmentOptions {
  /// 0: silent, 1: ceres brief report (one line), 2: ceres full report
  int verbosity_level = 1;

  /// update intrinsics or keep fixed
  bool optimize_intrinsics = false;

  /// use huber robust norm or squared norm
  bool use_huber = true;

  /// parameter for huber loss (in pixel)
  double huber_parameter = 1.0;

  /// maximum number of solver iterations
  int max_num_iterations = 20;
};

// Run bundle adjustment to optimize cameras, points, and optionally intrinsics
void bundle_adjustment(
    const Corners& feature_corners, const BundleAdjustmentOptions& options,
    const std::set<FrameCamId>& fixed_cameras, Calibration& calib_cam,
    Cameras& cameras, Landmarks& landmarks,
    Eigen::aligned_map<Timestamp, PoseVelState<double>>& states,
    Eigen::aligned_map<Timestamp, IntegratedImuMeasurement<double>>&
        imu_measurements,
    bool use_imu) {
  ceres::Problem problem;

  // TODO SHEET 4: Setup optimization problem
  /* -----Quick Reminder-----
   * 1. feature_corners.at(fcid).at(feature_id) would give us the 2d vector ->
   * detected corner in the image
   * 2. Landmarks = std::unordered_map<TrackId, Landmark>;
   */
  /*
   * -----Quick Thought about the implementation-----
   * 1. Since we have local parametrization, so first add those needed parameter
   * block into our problem
   * 2. Iterate through all the landmarks, use all of its inliers observation to
   * calculate residual.
   */
  // Iterate through cameras to add T_w_c of each camera in to the parameter
  // block using local parametrization
  for (auto& kv : cameras) {
    problem.AddParameterBlock(kv.second.T_w_c.data(),
                              Sophus::SE3d::num_parameters,
                              new Sophus::test::LocalParameterizationSE3);
    // Set the fixed frame paramter constant
    if (fixed_cameras.find(kv.first) != fixed_cameras.end()) {
      problem.SetParameterBlockConstant(kv.second.T_w_c.data());
    }
  }

  for (auto& kv : landmarks) {
    const auto& t_id = kv.first;
    auto& lm = kv.second;
    auto& p_3d = lm.p;
    for (const auto& ob : lm.obs) {
      const auto& fcid = ob.first;
      const auto& f_id = ob.second;
      const auto& p_2d = feature_corners.at(fcid).corners[f_id];
      BundleAdjustmentReprojectionCostFunctor* c =
          new BundleAdjustmentReprojectionCostFunctor(
              p_2d, calib_cam.intrinsics[fcid.cam_id]->name());
      ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<
          BundleAdjustmentReprojectionCostFunctor, 2,
          Sophus::SE3d::num_parameters, 3, 8>(c);

      if (options.use_huber) {
        problem.AddResidualBlock(cost_function,
                                 new ceres::HuberLoss(options.huber_parameter),
                                 cameras.at(fcid).T_w_c.data(), p_3d.data(),
                                 calib_cam.intrinsics[fcid.cam_id]->data());
      } else {
        problem.AddResidualBlock(cost_function, NULL,
                                 cameras.at(fcid).T_w_c.data(), p_3d.data(),
                                 calib_cam.intrinsics[fcid.cam_id]->data());
      }
    }
  }

  if (use_imu) {
    // Add the parameter block first
    if (states.size() == 3) {
      for (auto& state : states) {
        problem.AddParameterBlock(state.second.T_w_i.data(),
                                  Sophus::SE3d::num_parameters,
                                  new Sophus::test::LocalParameterizationSE3);
      }
      // Build the two residuals for the frames
      int iter_counter = 0;
      auto iter = states.rbegin();

      while (iter_counter < 3) {
        const IntegratedImuMeasurement<double>& imu_meas =
            imu_measurements[iter->first];
        visnav::PoseVelState<double>& state1 = states[iter->first];
        ++iter;
        visnav::PoseVelState<double>& state0 = states[iter->first];

        // Build parameter blocks for frame optimization
        //    problem.AddParameterBlock(state0.T_w_i.data(),
        //    Sophus::SE3d::num_parameters,
        //                              new
        //                              Sophus::test::LocalParameterizationSE3);
        //    problem.AddParameterBlock(state1.T_w_i.data(),
        //    Sophus::SE3d::num_parameters,
        //                              new
        //                              Sophus::test::LocalParameterizationSE3);
        // might have to add state[1] twice, because it appears in two residuals

        BundleAdjustmentImuCostFunctor* imu_c =
            new BundleAdjustmentImuCostFunctor(imu_meas.getDeltaState(),
                                               visnav::constants::g,
                                               state0.t_ns, state1.t_ns);
        ceres::CostFunction* imu_cost_function =
            new ceres::AutoDiffCostFunction<
                BundleAdjustmentImuCostFunctor, 9,
                Sophus::SE3d::num_parameters,  // state0.T_w_i
                Sophus::SE3d::num_parameters,  // state1.T_w_i
                3,                             // state0.vel_w_i
                3                              // state1.vel_w_i
                >(imu_c);

        Eigen::Matrix<double, 3, 1> g = visnav::constants::g;
        if (options.use_huber) {
          problem.AddResidualBlock(
              imu_cost_function, new ceres::HuberLoss(options.huber_parameter),
              state0.T_w_i.data(), state1.T_w_i.data(), state0.vel_w_i.data(),
              state1.vel_w_i.data());
        } else {
          problem.AddResidualBlock(imu_cost_function, NULL, state0.T_w_i.data(),
                                   state1.T_w_i.data(), state0.vel_w_i.data(),
                                   state1.vel_w_i.data());
        }
        iter_counter++;
      }
    }
  } else {
    // Do nothing here
  }

  if (!options.optimize_intrinsics) {
    // Keep the intrinsics fixed
    problem.SetParameterBlockConstant(calib_cam.intrinsics[0]->data());
    problem.SetParameterBlockConstant(calib_cam.intrinsics[1]->data());
  } else {
    // Do nothing
  }

  // Solve
  ceres::Solver::Options ceres_options;
  ceres_options.max_num_iterations = options.max_num_iterations;
  ceres_options.linear_solver_type = ceres::SPARSE_SCHUR;
  ceres_options.num_threads = std::thread::hardware_concurrency();
  ceres::Solver::Summary summary;
  Solve(ceres_options, &problem, &summary);
  switch (options.verbosity_level) {
    // 0: silent
    case 1:
      std::cout << summary.BriefReport() << std::endl;
      break;
    case 2:
      std::cout << summary.FullReport() << std::endl;
      break;
  }
}

}  // namespace visnav
