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

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>
#include <visnav/ORBVocabulary.h>
#include <visnav/hash.h>

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <bitset>
#include <cstdint>
#include <map>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <unordered_map>
#include <vector>
#define UNUSED(x) (void)(x)

namespace visnav {

/// Identifies a frame of multiple images (stereo pair).
/// Use of int64_t has historic reasons (used to be timetamp).
using FrameId = int64_t;

/// identifies the camera (left or right)
using CamId = std::size_t;

/// image timestamp as nanoseconds since unix epoch
using Timestamp = int64_t;

/// pair of frame id and camera id identifies an image (== "image id")
struct FrameCamId {
  FrameCamId() : frame_id(0), cam_id(0){};
  FrameCamId(const FrameId& new_frame_id, const CamId& new_cam_id)
      : frame_id(new_frame_id), cam_id(new_cam_id){};

  // Frame id in range 0 ... (num_frames-1)
  FrameId frame_id;

  // Camera id in the setup (unusally 0 or 1, for left or right camera,
  // respectively)
  CamId cam_id;

  bool operator==(const FrameCamId& other) const {
    return (frame_id == other.frame_id) && (cam_id == other.cam_id);
  }

  bool operator!=(const FrameCamId& other) const {
    return (frame_id != other.frame_id) || (cam_id != other.cam_id);
  }

  bool operator<(const FrameCamId& other) const {
    if (frame_id == other.frame_id) return cam_id < other.cam_id;
    return frame_id < other.frame_id;
  }

  // For tbb hashing
  explicit operator size_t() const {
    size_t seed = 0;
    hash_combine(seed, frame_id);
    hash_combine(seed, cam_id);
    return seed;
  }
};

std::ostream& operator<<(std::ostream& os, const FrameCamId& fcid) {
  os << fcid.frame_id << "_" << fcid.cam_id;
  return os;
}

/// ids for 2D features detected in images
using FeatureId = int;

/// Ids for feature tracks; also used for landmarks created from (some of) the
/// tracks;
using TrackId = int64_t;

/// keypoint positions and descriptors for an image
struct KeypointsData {
  /// collection of 2d corner points (indexed by FeatureId)
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      corners;
  /// collection of feature orientation (in radian) with same index as
  // `corners`
  /// (indexed by FeatureId)
  std::vector<double> corner_angles;
  /// collection of feature descriptors with same index as `corners` (indexed by
  /// FeatureId)
  std::vector<std::bitset<256>> corner_descriptors;
};
/// keypoint positions and descriptors for an image
// struct KeypointsData {
//   /// collection of 2d corner points (indexed by FeatureId)
//   std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
//       corners;
//   /// collection of feature descriptors with same index as `corners` (indexed
//   by
//   /// FeatureId)
//   cv::Mat corner_descriptors;
// };

/// feature corners is a collection of { imageId => KeypointsData }
using Corners = tbb::concurrent_unordered_map<FrameCamId, KeypointsData>;

/// feature matches for an image pair
struct MatchData {
  /// estimated transformation (based on inliers or calibration) from the second
  /// image's coordinate system to the first image's corrdinate system
  Sophus::SE3d T_i_j;
  /// collection of {featureId_i, featureId_j} pairs of all matches
  std::vector<std::pair<FeatureId, FeatureId>> matches;
  /// collection of {featureId_i, featureId_j} pairs of inlier matches
  std::vector<std::pair<FeatureId, FeatureId>> inliers;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// image to landmark matches
struct LandmarkMatchData {
  /// camera pose estimated from landmarks in world frame
  Sophus::SE3d T_w_c;
  /// collection of {featureId, trackId} pairs of all matches
  std::vector<std::pair<FeatureId, TrackId>> matches;
  /// subset of matches that are localization inliers
  std::vector<std::pair<FeatureId, TrackId>> inliers;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// feature matches is a collection of { (imageId, imageId) => MatchData }
using Matches = tbb::concurrent_unordered_map<
    std::pair<FrameCamId, FrameCamId>, MatchData,
    visnav_hash<std::pair<FrameCamId, FrameCamId>>,
    std::equal_to<std::pair<FrameCamId, FrameCamId>>,
    Eigen::aligned_allocator<
        std::pair<const std::pair<FrameCamId, FrameCamId>, MatchData>>>;

/// pair of image and feature indices
using ImageFeaturePair = std::pair<FrameCamId, FeatureId>;

/// Feature tracks are collections of {ImageId => FeatureId}.
/// I.e. a collection of all images that observed this feature and the
/// corresponding feature index in that image.
using FeatureTrack = std::map<FrameCamId, FeatureId>;

/// FeatureTracks is a collection {TrackId => FeatureTrack}
using FeatureTracks = std::unordered_map<TrackId, FeatureTrack>;

/// Word id for BoW.
using WordId = unsigned int;

/// Value of a word
using WordValue = double;

/// Normalized sparse vector of words to represent images. "Sparse" means that
/// words with value 0 don't appear explicitly.
using BowVector = std::vector<std::pair<WordId, WordValue>>;

/// Result of BoW query. Should be sorted by the confidence.
using BowQueryResult = std::vector<std::pair<FrameCamId, double>>;

/// Inverse index used in Bow database. Not suited for concurrent computation.
using BowDBInverseIndex =
    std::unordered_map<WordId, std::vector<std::pair<FrameCamId, WordValue>>>;

/// Inverse index used in Bow database. Suited for concurrent computation.
using BowDBInverseIndexConcurrent = tbb::concurrent_unordered_map<
    WordId, tbb::concurrent_vector<std::pair<FrameCamId, WordValue>>>;

using DBoWInvertedFile = std::vector<std::vector<FrameCamId>>;
/// cameras in the map
struct Camera {
  /// camera pose (transforms from camera to world)
  Sophus::SE3d T_w_c;
  bool active;
  // std::vector<FrameCamId> covisible_camera_fcids;
  std::map<FrameCamId, int> covisible_weights;
  std::map<FrameCamId, Sophus::SE3d> covisible_rel_poses;
  FrameCamId last_fcid;
  // BowVector bow_vector;
  DBoW2::BowVector bow_vector;
  DBoW2::FeatureVector feature_vector;
  // Associated landmarks with keypoins
  std::map<TrackId, FeatureId> map_points;
  std::string img_path;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using CovisibilityGraph = std::unordered_map<FrameCamId, std::set<FrameCamId>>;

using ConsistentGroup = std::pair<std::set<FrameCamId>, int>;
using ConsistentGroups = std::vector<ConsistentGroup>;
/// landmarks in the map
struct Landmark {
  /// 3d position in world coordinates
  Eigen::Vector3d p;

  /// 3d position in the camera coordinates -> the camera that created this
  /// landmark
  Eigen::Vector3d p_c;
  /// Which camera create this landmark
  FrameCamId from_fcid;
  /// Inlier observations in the current map.
  /// This is a subset of the original feature track.
  FeatureTrack obs;

  /// Keep track of all inlier observation of cameras even though the camera is
  /// deactivated
  FeatureTrack all_obs;
  /// Outlier observations in the current map.
  /// This is a subset of the original feature track.
  FeatureTrack outlier_obs;

  /// If there's no keyframe observation,
  /// deactivate the landmark.
  bool active;
};

/// collection {imageId => Camera} for all cameras in the map
using Cameras =
    std::map<FrameCamId, Camera, std::less<FrameCamId>,
             Eigen::aligned_allocator<std::pair<const FrameCamId, Camera>>>;

/// collection {trackId => Landmark} for all landmarks in the map.
/// trackIds correspond to feature_tracks
using Landmarks = std::unordered_map<TrackId, Landmark>;

/// camera candidate to be added to map
struct CameraCandidate {
  FrameCamId fcid;
  std::vector<TrackId> shared_tracks;

  // keep track of different stages of adding a set of candidate cameras and its
  // landmarks to the map
  bool tried = false;            //!< tried to add to map
  bool camera_added = false;     //!< succeeded to add to map
  bool landmarks_added = false;  //!< added new landmarks to map
};

/// list of current candidates and some book keeping for the different stages
struct CameraCandidates {
  enum Stage {
    ComputeCandidates,
    AddCameras,
    AddLandmarks,
    Optimize,
    RemoveOutliers,
    Done
  };

  std::vector<CameraCandidate> cameras;
  Stage current_stage = ComputeCandidates;
  int min_localization_inliers = 0;
  int max_cameras_to_add = 0;

  int num_cameras_added() {
    int num_added = 0;
    for (const auto& c : cameras) {
      if (c.camera_added) {
        ++num_added;
      }
    }
    return num_added;
  }

  int num_landmarks_added() {
    int num_added = 0;
    for (const auto& c : cameras) {
      if (c.landmarks_added) {
        ++num_added;
      }
    }
    return num_added;
  }
};

/// Flags for different landmark outlier criteria
enum OutlierFlags {
  OutlierNone = 0,
  // reprojection error much too large
  OutlierReprojectionErrorHuge = 1 << 0,
  // reprojection error too large
  OutlierReprojectionErrorNormal = 1 << 1,
  // distance to a camera too small
  OutlierCameraDistance = 1 << 2,
  // z-coord in some camera frame too small
  OutlierZCoordinate = 1 << 3
};

/// info on a single projected landmark
struct ProjectedLandmark {
  Eigen::Vector2d point_measured;            //!< detected feature location
  Eigen::Vector2d point_reprojected;         //!< landmark projected into image
  Eigen::Vector3d point_3d_c;                //!< 3d point in camera coordinates
  TrackId track_id = -1;                     //!< corresponding track_id
  double reprojection_error = 0;             //!< current reprojection error
  unsigned int outlier_flags = OutlierNone;  //!< flags for outlier
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

using ProjectedLandmarkPtr = std::shared_ptr<ProjectedLandmark>;
using ProjectedLandmarkConstPtr = std::shared_ptr<const ProjectedLandmark>;

/// all landmark projections for inlier and outlier observations for a single
/// image
struct ImageProjection {
  std::vector<ProjectedLandmarkConstPtr> obs;
  std::vector<ProjectedLandmarkConstPtr> outlier_obs;
};

/// projections for all images
using ImageProjections = std::map<FrameCamId, ImageProjection>;

/// inlier projections indexed per track
using TrackProjections =
    std::unordered_map<TrackId,
                       std::map<FrameCamId, ProjectedLandmarkConstPtr>>;

}  // namespace visnav

namespace std {

template <>
struct hash<visnav::FrameCamId> {
  inline std::size_t operator()(const visnav::FrameCamId& val) const noexcept {
    std::size_t seed = 0;
    visnav::hash_combine(seed, val.frame_id);
    visnav::hash_combine(seed, val.cam_id);
    return seed;
  }
};

}  // namespace std
