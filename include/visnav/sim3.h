/*
 * Created on Fri Mar 10 2023
 * The project is an ORB-SLAM inspired software which aims to implement all its
 * modules by myself. For more information, please refer to
 * https://github.com/yunjinli/visual-slam The MIT License (MIT) Copyright (c)
 * 2023 Yun-Jin Li (Jim)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <pangolin/image/managed_image.h>
#include <visnav/calibration.h>
#include <visnav/common_types.h>
#include <visnav/keypoints.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

namespace visnav {
using Point = Eigen::Vector3d;
using Points = std::vector<Eigen::Vector3d>;
using M33 = Eigen::Matrix3d;

/// @brief Compute the centroid of a given set of points
/// @param[in] points vector of points
/// @param[out] centroid the output centroid of `points`
void compute_centroid(const Points& points, Point& centroid) {
  centroid = Eigen::Vector3d::Zero();
  for (const auto& p : points) {
    centroid = centroid + p;
  }
  centroid = centroid / points.size();
}
/// @brief Compute the covariance matrix of a given 2 sets of points
/// @param[in] points_cam1 first set of points
/// @param[in] points_cam2 second set of points
/// @param[in] centroid1 centroid of `points_cam1`
/// @param[in] centroid2 centroid of `points_cam1`
/// @param[out] H The ouput covariance matrix
void compute_covariance(const Points& points_cam1, const Points& points_cam2,
                        const Point& centroid1, const Point& centroid2,
                        M33& H) {
  // Note that `P1` is all the points substracting
  // the `centroid1`
  int N = points_cam1.size();
  Eigen::Matrix<double, 3, Eigen::Dynamic> P1;
  // Note that `P2` is all the points substracting
  // the `centroid2`
  Eigen::Matrix<double, 3, Eigen::Dynamic> P2;
  P1.resize(3, N);
  P2.resize(3, N);
  for (size_t i = 0; i < points_cam1.size(); i++) {
    P1.col(i) = points_cam1[i] - centroid1;
    P2.col(i) = points_cam2[i] - centroid2;
  }
  H = P1 * P2.transpose();
}

/// @brief Compute the rotation matrix using svd with the covariance matrix
/// @param[in] H The covariance matrix of the 2 point sets
/// @param[out] R The output rotation matrix with respect to camera 2
void compute_rotation_svd(const M33& H, M33& R) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      H, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3d S;
  S.setIdentity();

  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0)
    S(2, 2) = -1;

  // Eigen::BDCSVD<M33> svd(
  //     H, Eigen::ComputeThinU | Eigen::ComputeThinV);  // compute the SVD
  // M33 U = svd.matrixU();  // get the left singular vectors
  // M33 V = svd.matrixV();  // get the right singular vectors
  R = svd.matrixV() * S * svd.matrixU().transpose();
}
/// @brief Compute the translation based on centroids and rotation matrix
/// @param R The rotation matrix with respect to camera 2
/// @param centroid1 The centroid for the first point set
/// @param centroid2 The centroid for the second point set
/// @param translation The output translation with respect to camera 2
void compute_translation(const M33& R, const Point& centroid1,
                         const Point& centroid2, Point& translation) {
  translation = centroid2 - R * centroid1;
}
/// @brief Compute the similarity transformation from cam2 to cam1
/// @param[in] cam1 The first candidate camera
/// @param[in] cam2 The first candidate camera
/// @param[in] landmarks All the landmarks (map points) in the map
/// @param[out] sim3 The output similarity transformation
void compute_sim3_1(const Camera& cam1, const Camera& cam2,
                    const Landmarks& landmarks, Sophus::SE3d& sim3) {
  Points p_cam1;
  Points p_cam2;
  for (const auto& kv : cam1.map_points) {
    const TrackId& landmark_id = kv.first;
    if (cam2.map_points.count(landmark_id)) {
      p_cam1.push_back(cam1.T_w_c.inverse() * landmarks.at(landmark_id).p);
      p_cam2.push_back(cam2.T_w_c.inverse() * landmarks.at(landmark_id).p);
    }
  }
  std::cout << "Compute Sim(3) by " << p_cam1.size() << " common map points"
            << std::endl;
  Point c1;
  Point c2;
  compute_centroid(p_cam1, c1);
  compute_centroid(p_cam2, c2);
  M33 H;
  compute_covariance(p_cam1, p_cam2, c1, c2, H);
  M33 R21;
  compute_rotation_svd(H, R21);
  Point t21;
  compute_translation(R21, c1, c2, t21);
  // std::cout << "For debug: print R21: " << std::endl;
  // std::cout << R21 << std::endl;
  sim3.translation() = t21;
  sim3.setRotationMatrix(R21);
  // sim3.rotationMatrix() = R21;
}

bool compute_sim3_opengv(const Calibration& calib_cam, const FrameCamId& fcid1,
                         const FrameCamId& fcid2, const Camera& cam1,
                         const Camera& cam2, const Corners& feature_corners,
                         Sophus::SE3d& sim3) {
  // pangolin::ManagedImage<uint8_t> img1 = pangolin::LoadImage(cam1.img_path);
  // pangolin::ManagedImage<uint8_t> img2 = pangolin::LoadImage(cam2.img_path);

  const KeypointsData& kd1 = feature_corners.at(fcid1);
  const KeypointsData& kd2 = feature_corners.at(fcid2);

  // detectKeypointsAndDescriptors(img1, kd1, 1500, true);
  // detectKeypointsAndDescriptors(img2, kd2, 1500, true);
  int max_iteration = 10;
  int total_iteration = 0;
  bool solution_found = false;
  while (!solution_found) {
    MatchData md;
    matchDescriptors(kd1.corner_descriptors, kd2.corner_descriptors, md.matches,
                     70, 1.2);

    opengv::bearingVectors_t bearingVectors1;
    opengv::bearingVectors_t bearingVectors2;
    for (const auto& match : md.matches) {
      bearingVectors1.push_back(calib_cam.intrinsics[fcid1.cam_id]->unproject(
          kd1.corners.at(match.first)));
      bearingVectors2.push_back(calib_cam.intrinsics[fcid2.cam_id]->unproject(
          kd2.corners.at(match.second)));
    }
    // create the central relative adapter
    opengv::relative_pose::CentralRelativeAdapter adapter(bearingVectors1,
                                                          bearingVectors2);
    // create a RANSAC object
    opengv::sac::Ransac<
        opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
        ransac;
    // create a CentralRelativePoseSacProblem
    // (set algorithm to STEWENIUS, NISTER, SEVENPT, or EIGHTPT)
    std::shared_ptr<
        opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
        relposeproblem_ptr(
            new opengv::sac_problems::relative_pose::
                CentralRelativePoseSacProblem(
                    adapter, opengv::sac_problems::relative_pose::
                                 CentralRelativePoseSacProblem::NISTER));
    // run ransac
    ransac.sac_model_ = relposeproblem_ptr;
    // ransac.threshold_ = threshold;
    // ransac.max_iterations_ = maxIterations;
    ransac.computeModel();
    opengv::transformation_t best_transformation = ransac.model_coefficients_;

    // Refinement using non-linear optimization
    adapter.sett12(best_transformation.block(0, 3, 3, 1));
    adapter.setR12(best_transformation.block(0, 0, 3, 3));
    opengv::transformation_t nonlinear_transformation =
        opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);
    if (ransac.inliers_.size() < 15) {
      total_iteration++;
    } else {
      sim3 = Sophus::SE3<double>(nonlinear_transformation.block(0, 0, 3, 3),
                                 nonlinear_transformation.block(0, 3, 3, 1));
      if (sim3.translation().cwiseAbs().sum() > 5) {
        total_iteration++;
      } else {
        // Not sure if bundle adjustment is needed here...
        solution_found = true;
      }
    }
    if (total_iteration > max_iteration) {
      break;
    }
  }
  return solution_found;
}
/// @brief Compute the Sim(3) for the current frame and the loop candidate
/// @param[in] calib_cam Camera model
/// @param[in] fcid1 Current frame ID
/// @param[in] fcid2 Loop candidate ID
/// @param[in] feature_corners Keypoint collections for all keyframes
/// @param[in] keyframes Collection of keyframes
/// @param[in] landmarks Current map
/// @param[in] graph Current covisibility graph
/// @param[in] reprojection_error_pnp_inlier_threshold_pixel RANSAC threshold
/// @param[out] sim3 Output similarity transforamation
/// @return If the Sim(3) is correctly computed
bool compute_sim3(const Calibration& calib_cam, const FrameCamId& fcid1,
                  const FrameCamId& fcid2, const Corners& feature_corners,
                  const Cameras& keyframes, const Landmarks& landmarks,
                  const CovisibilityGraph& graph,
                  double reprojection_error_pnp_inlier_threshold_pixel,
                  Sophus::SE3d& sim3) {
  std::cout << std::endl;
  std::cout << "===== SIM(3) =====" << std::endl;
  std::cout << std::endl;
  // fcid1 would be the current keyframe ID
  // fcid2 would be the candidate keyframe ID
  const std::set<FrameCamId>& candidates = graph.at(fcid2);
  // candidates.insert(fcid2);
  const KeypointsData& current_keypoints = feature_corners.at(fcid1);
  opengv::points_t points;
  opengv::bearingVectors_t bearingVectors;
  std::set<TrackId> already_added_landmarks;
  std::set<FeatureId> already_added_features;
  std::map<FeatureId, FeatureId> matches;
  MatchData md;
  // We first check the correspondence with the loop candidate
  matches.clear();
  md.matches.clear();
  // Note that we put candidata keypoints in the front
  matchDescriptors(feature_corners.at(fcid2).corner_descriptors,
                   current_keypoints.corner_descriptors, md.matches, 70, 1.2);

  // put matches to a hash map for better lookup
  for (int i = 0; i < md.matches.size(); i++) {
    matches.emplace(md.matches[i]);
  }

  for (const auto& kv : keyframes.at(fcid2).map_points) {
    if (matches.count(kv.second)) {
      if (!already_added_landmarks.count(kv.first) &&
          !already_added_features.count(matches.at(kv.second))) {
        points.push_back(landmarks.at(kv.first).p);
        bearingVectors.push_back(calib_cam.intrinsics[0]->unproject(
            current_keypoints.corners[matches.at(kv.second)]));
        already_added_landmarks.insert(kv.first);
        already_added_features.insert(matches.at(kv.second));
      }
    }
  }
  // Then we check the correspondence with loop candidates' neighbors
  for (auto it = candidates.begin(); it != candidates.end(); it++) {
    if (*it == fcid1) {
      continue;
    }
    matches.clear();
    md.matches.clear();

    // Note that we put candidata keypoints in the front
    matchDescriptors(feature_corners.at(*it).corner_descriptors,
                     current_keypoints.corner_descriptors, md.matches, 70, 1.2);

    // put matches to a hash map for better lookup
    for (int i = 0; i < md.matches.size(); i++) {
      matches.emplace(md.matches[i]);
    }

    for (const auto& kv : keyframes.at(*it).map_points) {
      if (matches.count(kv.second)) {
        if (!already_added_landmarks.count(kv.first) &&
            !already_added_features.count(matches.at(kv.second))) {
          points.push_back(landmarks.at(kv.first).p);
          bearingVectors.push_back(calib_cam.intrinsics[0]->unproject(
              current_keypoints.corners[matches.at(kv.second)]));
          already_added_landmarks.insert(kv.first);
          already_added_features.insert(matches.at(kv.second));
        }
      }
    }
  }
  if (points.size() < 5 || bearingVectors.size() < 5) {
    std::cout << "SIM3 FAIL...NOT ENOUGH CORRESPONDENCE...";
    return false;
  } else {
    std::cout << "SIM3 CONTINUE... ENOUGH CORRESPONDENCE FOUND...";
  }
  int total_iteration = 0;
  int max_iteration = 10;
  bool solution_found = false;

  while (!solution_found) {
    // create the central adapter
    opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearingVectors,
                                                          points);
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
    Sophus::SE3d cur_pose =
        Sophus::SE3<double>(nonlinear_transformation.block(0, 0, 3, 3),
                            nonlinear_transformation.block(0, 3, 3, 1));
    ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                            ransac.threshold_, ransac.inliers_);
    sim3 = keyframes.at(fcid2).T_w_c.inverse() * cur_pose;

    double rel_trans_sum = sim3.log().block(0, 0, 3, 1).cwiseAbs().sum();
    if (rel_trans_sum > 5) {
      std::cout << "SIM(3) SOLVER FAIL... RECOMPUTE..." << std::endl;
      total_iteration++;
    } else {
      solution_found = true;
      return solution_found;
    }
    if (total_iteration > max_iteration) {
      return solution_found;
    }
  }
}

}  // namespace visnav
