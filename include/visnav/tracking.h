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

#pragma once

#include <visnav/ORBVocabulary.h>
#include <visnav/calibration.h>
#include <visnav/common_types.h>
#include <visnav/keypoints.h>

#include <opencv2/opencv.hpp>
#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>
#include <set>
namespace visnav {

bool track_camera(const Sophus::SE3d& current_pose,
                  const std::shared_ptr<AbstractCamera<double>>& cam,
                  const KeypointsData& kdl, const Landmarks& landmarks,
                  const double reprojection_error_pnp_inlier_threshold_pixel,
                  LandmarkMatchData& md, const Sophus::SE3d& vel,
                  double motion_threshold, bool last_tracking_successful) {
  md.inliers.clear();

  // predict the pose based on constant motion model
  md.T_w_c = current_pose;

  if (md.matches.size() < 10) {
    if (last_tracking_successful) {
      // predict the pose based on constant motion model
      // if the last tracking is successful
      std::cout << "TRACKING LOST... NOT ENOUGH INLIERS... USING MOTION MODEL "
                   "PREDICTION..."
                << std::endl;
      md.T_w_c = current_pose * vel;
    } else {
      std::cout << "TRACKING LOST... NOT ENOUGH INLIERS... USING LAST POSE..."
                << std::endl;
      // maintain to the last position
    }
    return false;
  }

  int max_iteration = 5;
  int total_iteration = 0;
  bool solution_found = false;
  while (!solution_found) {
    opengv::points_t points;
    opengv::bearingVectors_t bearingVectors;

    for (const auto& kv : md.matches) {
      points.push_back(landmarks.at(kv.second).p);
      bearingVectors.push_back(cam->unproject(kdl.corners[kv.first]));
    }
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
    md.T_w_c = Sophus::SE3<double>(nonlinear_transformation.block(0, 0, 3, 3),
                                   nonlinear_transformation.block(0, 3, 3, 1));
    ransac.sac_model_->selectWithinDistance(nonlinear_transformation,
                                            ransac.threshold_, ransac.inliers_);

    // Check if camera is poorly tracked
    Sophus::Vector6d se3_vel = (current_pose.inverse() * md.T_w_c).log();

    double err = (se3_vel - vel.log()).block(0, 0, 3, 1).cwiseAbs().sum();

    if (err > motion_threshold) {
      if (last_tracking_successful) {
        md.T_w_c = current_pose * vel;
        std::cout << "TRACKING LOST... USING MOTION MODEL PREDICTION..."
                  << std::endl;
        std::cout << "The error is: " << err << std::endl;
      } else {
        md.T_w_c = current_pose;
        std::cout << "TRACKING LOST... USING LAST POSE..." << std::endl;
        std::cout << "The error is: " << err << std::endl;
      }
      total_iteration++;
    } else {
      std::cout << "TRACKING SUCCESSFUL..." << std::endl;
      std::cout << "The error is: " << err << std::endl;
      for (const auto& kv : ransac.inliers_) {
        md.inliers.push_back(md.matches[kv]);
      }
      solution_found = true;
      break;
    }
    if (total_iteration > max_iteration) {
      break;
    }
  }
  return solution_found;
}

bool detect_relocalization_candidate(
    const ORBVocabulary* voc, const DBoWInvertedFile& recognition_database,
    const DBoW2::BowVector& bow_vector, const Cameras& keyframes,
    std::vector<FrameCamId>& candidate_kf_fcids) {
  std::unordered_map<FrameCamId, int> num_sharing_words;
  bool has_any_sharing_words = false;

  for (DBoW2::BowVector::const_iterator vit = bow_vector.begin(),
                                        vend = bow_vector.end();
       vit != vend; vit++) {
    const std::vector<FrameCamId>& relocalization_keyframes =
        recognition_database[vit->first];
    has_any_sharing_words = true;
    for (const auto& rkf : relocalization_keyframes) {
      if (num_sharing_words.count(rkf)) {
        num_sharing_words[rkf] += 1;
      } else {
        num_sharing_words[rkf] = 0;
      }
    }
  }

  if (!has_any_sharing_words || num_sharing_words.empty()) {
    return false;
  }

  // Only compare against those keyframes that share enough words
  int max_num_sharing_words = 0;
  for (const auto& kv : num_sharing_words) {
    if (kv.second > max_num_sharing_words) {
      max_num_sharing_words = kv.second;
    }
  }
  int sharing_words_threshold = max_num_sharing_words * 0.8f;
  std::vector<std::pair<double, FrameCamId>> score_and_match;
  for (const auto& kv : num_sharing_words) {
    if (kv.second > sharing_words_threshold) {
      // double score = compute_bow_score(new_kf.bow_vector,
      //                                  keyframes.at(kv.first).bow_vector);
      double score = voc->score(bow_vector, keyframes.at(kv.first).bow_vector);
      score_and_match.push_back(std::make_pair(score, kv.first));
    }
  }
  std::partial_sort(
      score_and_match.begin(),
      score_and_match.begin() + std::min((int)(score_and_match.size()), 5),
      score_and_match.end(),
      [](const auto& a, const auto& b) { return a.first > b.first; });
  for (int i = 0; i < std::min((int)(score_and_match.size()), 5); i++) {
    candidate_kf_fcids.push_back(score_and_match[i].second);
  }
  return true;
}
bool relocalize_camera(const FrameCamId& fcid, const std::string& img_path,
                       const Calibration& calib_cam, cv::Ptr<cv::ORB> orb,
                       const ORBVocabulary* voc,
                       const DBoWInvertedFile& recognition_database,
                       const Cameras& keyframes, Sophus::SE3d vel,
                       const Sophus::SE3d& current_pose,
                       double motion_threshold, Sophus::SE3d& reloc_pose) {
  std::cout << "Relocalization begin..." << std::endl;
  DBoW2::BowVector bow_vector;
  DBoW2::FeatureVector feature_vector;
  cv::Mat img = cv::imread(img_path);
  Camera cam2;
  cam2.img_path = img_path;
  compute_bow_vector(img, orb, 1500, voc, bow_vector, feature_vector);
  std::vector<FrameCamId> reloc_fcids;
  bool reloc_pose_good = false;
  if (detect_relocalization_candidate(voc, recognition_database, bow_vector,
                                      keyframes, reloc_fcids)) {
    std::cout << reloc_fcids.size() << " candidates found..." << std::endl;
    for (const auto& reloc_fcid : reloc_fcids) {
      Sophus::SE3d sim3;
      pangolin::ManagedImage<uint8_t> img1 =
          pangolin::LoadImage(keyframes.at(reloc_fcid).img_path);
      pangolin::ManagedImage<uint8_t> img2 = pangolin::LoadImage(cam2.img_path);
      KeypointsData kd1;
      KeypointsData kd2;

      detectKeypointsAndDescriptors(img1, kd1, 1500, true);
      detectKeypointsAndDescriptors(img2, kd2, 1500, true);
      MatchData md;
      matchDescriptors(kd1.corner_descriptors, kd2.corner_descriptors,
                       md.matches, 70, 1.2);

      opengv::bearingVectors_t bearingVectors1;
      opengv::bearingVectors_t bearingVectors2;
      for (const auto& match : md.matches) {
        bearingVectors1.push_back(
            calib_cam.intrinsics[reloc_fcid.cam_id]->unproject(
                kd1.corners.at(match.first)));
        bearingVectors2.push_back(calib_cam.intrinsics[fcid.cam_id]->unproject(
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
      if (ransac.inliers_.size() < 15) {
        reloc_pose_good = false;
        continue;
      } else {
        std::cout << "Current Frame ID: " << fcid.frame_id << std::endl;
        std::cout << "Reference Frame ID: " << reloc_fcid.frame_id << std::endl;
        opengv::transformation_t nonlinear_transformation =
            opengv::relative_pose::optimize_nonlinear(adapter, ransac.inliers_);
        sim3 = Sophus::SE3<double>(nonlinear_transformation.block(0, 0, 3, 3),
                                   nonlinear_transformation.block(0, 3, 3, 1));
        reloc_pose = keyframes.at(reloc_fcid).T_w_c * sim3;
        // Check if camera is poorly tracked
        Sophus::Vector6d se3_vel = (current_pose.inverse() * reloc_pose).log();

        double err = (se3_vel - vel.log()).block(0, 0, 3, 1).cwiseAbs().sum();
        std::cout << "The error is: " << err << std::endl;
        if (err > motion_threshold) {
          reloc_pose_good = false;
          continue;
        } else {
          reloc_pose_good = true;
          break;
        }
      }
    }
  } else {
    return false;
  }
  return reloc_pose_good;
}
}  // namespace visnav
