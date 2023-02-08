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

#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  // TODO SHEET 5: project landmarks to the image plane using the current
  // locations of the cameras. Put 2d coordinates of the projected points into
  // projected_points and the corresponding id of the landmark into
  // projected_track_ids.
  for (auto& kv : landmarks) {
    const TrackId& id = kv.first;
    const Landmark& l = kv.second;
    Eigen::Vector3d p_c = current_pose.inverse() * l.p;
    //    std::cout << "w = " << cam->width() << " h = " << cam->height() <<
    //    "\n";
    if (p_c[2] < cam_z_threshold) {
      // Don't add because it's behind the camera
    } else {
      Eigen::Vector2d p_2d = cam->project(p_c);
      if (p_2d[0] > cam->width() || p_2d[1] > cam->height() || p_2d[0] < 0 ||
          p_2d[1] < 0) {
        // Don't add because it's out of the image
      } else {
        projected_points.push_back(p_2d);
        projected_track_ids.push_back(id);
      }
    }
  }
}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_threshold,
    const double feature_match_dist_2_best, LandmarkMatchData& md) {
  md.matches.clear();

  // TODO SHEET 5: Find the matches between projected landmarks and detected
  // keypoints in the current frame. For every detected keypoint search for
  // matches inside a circle with radius match_max_dist_2d around the point
  // location. For every landmark the distance is the minimal distance between
  // the descriptor of the current point and descriptors of all observations of
  // the landmarks. The feature_match_threshold and feature_match_dist_2_best
  // should be used to filter outliers the same way as in exercise 3. You should
  // fill md.matches with <featureId,trackId> pairs for the successful matches
  // that pass all tests.
  int current_keypoint_idx = 0;
  for (auto& corner : kdl.corners) {
    int projected_p_idx = 0;
    std::vector<std::pair<TrackId, int>> landmark_distances;
    for (auto& projected_p : projected_points) {
      double dist_2d = (corner - projected_p).norm();
      if (dist_2d < match_max_dist_2d) {
        // The projected point is within the circle
        const TrackId& t_id = projected_track_ids[projected_p_idx];
        int minimal_dist = 256;
        for (const auto& ob : landmarks.at(t_id).obs) {
          const FrameCamId& fcid = ob.first;
          const FeatureId& f_id = ob.second;
          int feature_match_dist =
              (feature_corners.at(fcid).corner_descriptors[f_id] ^
               kdl.corner_descriptors[current_keypoint_idx])
                  .count();
          if (feature_match_dist < minimal_dist) {
            minimal_dist = feature_match_dist;
          }
        }
        landmark_distances.push_back(std::make_pair(t_id, minimal_dist));
      }
      projected_p_idx++;
    }
    std::partial_sort(
        landmark_distances.begin(),
        landmark_distances.begin() +
            std::min((int)landmark_distances.size(), 2),
        landmark_distances.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });

    if (landmark_distances.size() == 0) {
      // do nothing
    } else {
      if (landmark_distances[0].second >= feature_match_threshold) {
        // discard
      } else {
        if (landmark_distances.size() < 2) {
          if (256 < landmark_distances[0].second * feature_match_dist_2_best) {
            // discard
          } else {
            md.matches.push_back(std::make_pair(current_keypoint_idx,
                                                landmark_distances[0].first));
          }
        } else {
          if (landmark_distances[1].second <
              landmark_distances[0].second * feature_match_dist_2_best) {
            // discard
          } else {
            md.matches.push_back(std::make_pair(current_keypoint_idx,
                                                landmark_distances[0].first));
          }
        }
      }
    }

    current_keypoint_idx++;
  }
}

void localize_camera(const Sophus::SE3d& current_pose,
                     const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     LandmarkMatchData& md) {
  md.inliers.clear();

  // default to previous pose if not enough inliers
  md.T_w_c = current_pose;

  if (md.matches.size() < 4) {
    return;
  }

  // TODO SHEET 5: Find the pose (md.T_w_c) and the inliers (md.inliers) using
  // the landmark to keypoints matches and PnP. This should be similar to the
  // localize_camera in exercise 4 but in this exercise we don't explicitly have
  // tracks.
  opengv::points_t points;
  opengv::bearingVectors_t bearingVectors;

  for (const auto& kv : md.matches) {
    points.push_back(landmarks.at(kv.second).p);
    bearingVectors.push_back(cam->unproject(kdl.corners[kv.first]));
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
  md.T_w_c = Sophus::SE3<double>(nonlinear_transformation.block(0, 0, 3, 3),
                                 nonlinear_transformation.block(0, 3, 3, 1));
  ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                          ransac.threshold_, ransac.inliers_);

  for (const auto& kv : ransac.inliers_) {
    md.inliers.push_back(md.matches[kv]);
  }
}

void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, const MatchData& md_stereo,
                       const LandmarkMatchData& md, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 5: Add new landmarks and observations. Here md_stereo contains
  // stereo matches for the current frame and md contains feature to landmark
  // matches for the left camera (camera 0). For all inlier feature to landmark
  // matches add the observations to the existing landmarks. If the left
  // camera's feature appears also in md_stereo.inliers, then add both
  // observations. For all inlier stereo observations that were not added to the
  // existing landmarks, triangulate and add new landmarks. Here
  // next_landmark_id is a running index of the landmarks, so after adding a new
  // landmark you should always increase next_landmark_id by 1.
  for (auto& kv : md.inliers) {  // In this frame
    const FeatureId& f_id = kv.first;
    const TrackId& t_id = kv.second;
    if (landmarks.count(t_id) > 0)  // landmark exists
    {
      landmarks.at(t_id).obs.emplace(std::make_pair(fcidl, f_id));
      //      if (pairExist(md_stereo.inliers,
      //                    f_id))  // if feature id also exist in stereo pair
      //      {
      //        landmarks.at(t_id).obs.emplace(
      //            std::make_pair(fcidr, md_stereo.inliers.at(f_id).second));
      //      }
      // Check if feature id also exist in stereo pair
      for (auto& inlier_pair : md_stereo.inliers) {
        if (inlier_pair.first == f_id) {
          landmarks.at(t_id).obs.emplace(
              std::make_pair(fcidr, inlier_pair.second));
          break;
        }
      }
    } else {
    }
  }
  for (auto& kv : md_stereo.inliers) {
    const FeatureId& f_idl = kv.first;
    const FeatureId& f_idr = kv.second;
    int counter = 0;
    for (auto& fid_tid : md.inliers) {
      const FeatureId& fid = fid_tid.first;
      const TrackId& tid = fid_tid.second;
      if (fid == f_idl) {
        // Already exist
        counter++;
        break;
      } else {
      }
    }
    if (counter == 0) {
      // DO TRIANGULATION
      opengv::bearingVectors_t bearingVectors1;
      opengv::bearingVectors_t bearingVectors2;

      bearingVectors1.push_back(
          calib_cam.intrinsics[fcidl.cam_id]->unproject(kdl.corners.at(f_idl)));
      bearingVectors2.push_back(
          calib_cam.intrinsics[fcidr.cam_id]->unproject(kdr.corners.at(f_idr)));

      opengv::relative_pose::CentralRelativeAdapter adapter(
          bearingVectors1, bearingVectors2, t_0_1, R_0_1);
      opengv::point_t point =
          md.T_w_c * opengv::triangulation::triangulate(adapter, 0);
      Landmark l;
      l.p = point;
      l.obs.emplace(std::make_pair(fcidl, f_idl));
      l.obs.emplace(std::make_pair(fcidr, f_idr));
      landmarks.emplace(std::make_pair(next_landmark_id++, l));
    }
  }
}

bool remove_old_keyframes(const FrameCamId fcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames, Camera& removed_camera,
                          FrameId& removed_fid) {
  kf_frames.emplace(fcidl.frame_id);
  bool removed = false;
  // TODO SHEET 5: Remove old cameras and observations if the number of keyframe
  // pairs (left and right image is a pair) is larger than max_num_kfs. The ids
  // of all the keyframes that are currently in the optimization should be
  // stored in kf_frames. Removed keyframes should be removed from cameras and
  // landmarks with no left observations should be moved to old_landmarks.
  while (kf_frames.size() > max_num_kfs) {
    auto kf_ptr = kf_frames.begin();
    auto kf_id = *kf_frames.begin();
    kf_frames.erase(kf_ptr);
    std::set<FrameCamId> removed_cameras = {{kf_id, 0}, {kf_id, 1}};
    FrameCamId rfcid(kf_id, 0);
    removed_camera = cameras.at(rfcid);
    removed_fid = kf_id;
    for (auto& fcid : removed_cameras) {
      cameras.erase(fcid);
    }
    for (auto& tid_lm : landmarks) {
      for (auto& fcid : removed_cameras) {
        tid_lm.second.obs.erase(fcid);
      }
    }
    removed = true;
  }
  std::vector<TrackId> removed_lm_id;
  for (auto& tid_lm : landmarks) {
    if (tid_lm.second.obs.size() == 0) {
      TrackId old_tid = tid_lm.first;
      Landmark old_lm = tid_lm.second;
      old_landmarks.emplace(std::make_pair(old_tid, old_lm));
      removed_lm_id.push_back(tid_lm.first);
      //      landmarks.erase(tid_lm.first);
    }
  }
  for (int i = 0; i < removed_lm_id.size(); i++) {
    landmarks.erase(removed_lm_id[i]);
  }
  return removed;
}
}  // namespace visnav
