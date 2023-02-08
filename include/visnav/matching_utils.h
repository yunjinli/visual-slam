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

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

Eigen::Matrix3d skew(const Eigen::Vector3d w) {
  Eigen::Matrix3d w_hat;
  w_hat << 0, -w(2, 0), w(1, 0), w(2, 0), 0, -w(0, 0), -w(1, 0), w(0, 0), 0;
  return w_hat;
}
void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  // TODO SHEET 3: compute essential matrix
  E = skew(t_0_1.normalized()) * R_0_1;
}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    // TODO SHEET 3: determine inliers and store in md.inliers
    // First we need to compute the unproject point
    Eigen::Vector3d p0_3d =
        cam1->unproject(p0_2d);  // output is in its own camera frame
    Eigen::Vector3d p1_3d =
        cam2->unproject(p1_2d);  // output is in its own camera frame
    double err = p0_3d.transpose() * E * p1_3d;
    if (abs(err) > epipolar_error_threshold) {
      // Don't add
    } else {
      md.inliers.push_back(md.matches[j]);
    }
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();
  md.T_i_j = Sophus::SE3d();

  // TODO SHEET 3: Run RANSAC with using opengv's CentralRelativePose and store
  // the final inlier indices in md.inliers and the final relative pose in
  // md.T_i_j (normalize translation). If the number of inliers is smaller than
  // ransac_min_inliers, leave md.inliers empty. Note that if the initial RANSAC
  // was successful, you should do non-linear refinement of the model parameters
  // using all inliers, and then re-estimate the inlier set with the refined
  // model parameters.
  opengv::bearingVectors_t bearingVectors1;
  opengv::bearingVectors_t bearingVectors2;

  for (auto& m : md.matches) {
    bearingVectors1.push_back(cam1->unproject(kd1.corners[m.first]));
    bearingVectors2.push_back(cam2->unproject(kd2.corners[m.second]));
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
  ransac.threshold_ = ransac_thresh;
  ransac.computeModel();
  // get the result
  opengv::transformation_t best_transformation = ransac.model_coefficients_;

  for (auto& selected_index : ransac.inliers_) {
    md.inliers.push_back(md.matches[selected_index]);
  }
  md.T_i_j =
      Sophus::SE3<double>(best_transformation.block(0, 0, 3, 3),
                          best_transformation.block(0, 3, 3, 1).normalized());

  // Refine the model parameter using only inliers
  bearingVectors1.clear();
  bearingVectors2.clear();
  for (auto& m : md.inliers) {
    bearingVectors1.push_back(cam1->unproject(kd1.corners[m.first]));
    bearingVectors2.push_back(cam2->unproject(kd2.corners[m.second]));
  }
  opengv::relative_pose::CentralRelativeAdapter adapter_refined(
      bearingVectors1, bearingVectors2);
  adapter_refined.sett12(best_transformation.block(0, 3, 3, 1));
  adapter_refined.setR12(best_transformation.block(0, 0, 3, 3));
  opengv::transformation_t nonlinear_transformation =
      opengv::relative_pose::optimize_nonlinear(adapter_refined);
  md.T_i_j = Sophus::SE3<double>(
      nonlinear_transformation.block(0, 0, 3, 3),
      nonlinear_transformation.block(0, 3, 3, 1).normalized());

  // Re-update the inliner using the refined model parameter
  bearingVectors1.clear();
  bearingVectors2.clear();
  for (auto& m : md.matches) {
    bearingVectors1.push_back(cam1->unproject(kd1.corners[m.first]));
    bearingVectors2.push_back(cam2->unproject(kd2.corners[m.second]));
  }
  opengv::relative_pose::CentralRelativeAdapter adapter_update_inliers(
      bearingVectors1, bearingVectors2);
  opengv::sac::Ransac<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac_refine;
  std::shared_ptr<
      opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_refined_ptr(
          new opengv::sac_problems::relative_pose::
              CentralRelativePoseSacProblem(
                  adapter_update_inliers,
                  opengv::sac_problems::relative_pose::
                      CentralRelativePoseSacProblem::NISTER));

  ransac_refine.sac_model_ = relposeproblem_refined_ptr;
  ransac_refine.threshold_ = ransac_thresh;
  ransac_refine.sac_model_->selectWithinDistance(
      nonlinear_transformation, ransac_thresh, ransac_refine.inliers_);

  md.inliers.clear();

  for (auto& selected_index : ransac_refine.inliers_) {
    md.inliers.push_back(md.matches[selected_index]);
  }

  if (md.inliers.size() < ransac_min_inliers) {
    md.inliers.clear();
  }
}
}  // namespace visnav
