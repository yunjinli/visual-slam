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

#include <visnav/calibration.h>
#include <visnav/common_types.h>

#include <Eigen/Dense>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/bitset.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>

namespace cereal {

using namespace visnav;

// Serialization for dynamic Eigen matrices
template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options,
          int _MaxRows, int _MaxCols>
void save(Archive& archive, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options,
                                          _MaxRows, _MaxCols> const& m) {
  int32_t rows = m.rows();
  int32_t cols = m.cols();
  // If a matrix is dynamic, we write the corresponding computed dimensions.
  if (_Rows == Eigen::Dynamic) archive(rows);
  if (_Cols == Eigen::Dynamic) archive(cols);
  for (int32_t i = 0; i < rows; i++)
    for (int32_t j = 0; j < cols; j++) archive(m(i, j));
}

template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options,
          int _MaxRows, int _MaxCols>
void load(
    Archive& archive,
    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& m) {
  int32_t rows;
  int32_t cols;
  // This allows to read both dynamic and static matrices.
  if (_Rows == Eigen::Dynamic)
    archive(rows);
  else
    rows = _Rows;
  if (_Cols == Eigen::Dynamic)
    archive(cols);
  else
    cols = _Cols;

  if (_Rows == Eigen::Dynamic || _Cols == Eigen::Dynamic) m.resize(rows, cols);
  for (int32_t i = 0; i < rows; i++)
    for (int32_t j = 0; j < cols; j++) archive(m(i, j));
}

template <class Archive>
void save(Archive& ar, const DoubleSphereCamera<double>& cam) {
  ar(cereal::make_nvp("fx", cam.getParam()[0]),
     cereal::make_nvp("fy", cam.getParam()[1]),
     cereal::make_nvp("cx", cam.getParam()[2]),
     cereal::make_nvp("cy", cam.getParam()[3]),
     cereal::make_nvp("xi", cam.getParam()[4]),
     cereal::make_nvp("alpha", cam.getParam()[5]));
}

template <class Archive>
void load(Archive& ar, DoubleSphereCamera<double>& cam) {
  Eigen::Matrix<double, 8, 1> intr;

  intr.setZero();

  ar(cereal::make_nvp("fx", intr[0]), cereal::make_nvp("fy", intr[1]),
     cereal::make_nvp("cx", intr[2]), cereal::make_nvp("cy", intr[3]),
     cereal::make_nvp("xi", intr[4]), cereal::make_nvp("alpha", intr[5]));

  cam = intr;
}

template <class Archive>
void save(Archive& ar, const std::shared_ptr<AbstractCamera<double>>& cam) {
  const std::string cam_type = cam->name();
  const Eigen::Matrix<double, 8, 1> intr = cam->getParam();
  ar(cereal::make_nvp("cam_type", cam_type), cereal::make_nvp("fx", intr[0]),
     cereal::make_nvp("fy", intr[1]), cereal::make_nvp("cx", intr[2]),
     cereal::make_nvp("cy", intr[3]), cereal::make_nvp("p1", intr[4]),
     cereal::make_nvp("p2", intr[5]), cereal::make_nvp("p3", intr[6]),
     cereal::make_nvp("p4", intr[7]), cereal::make_nvp("width", cam->width()),
     cereal::make_nvp("height", cam->height()));
}

template <class Archive>
void load(Archive& ar, std::shared_ptr<AbstractCamera<double>>& cam) {
  Eigen::Matrix<double, 8, 1> intr;
  std::string cam_type;
  int width, height;

  ar(cereal::make_nvp("cam_type", cam_type), cereal::make_nvp("fx", intr[0]),
     cereal::make_nvp("fy", intr[1]), cereal::make_nvp("cx", intr[2]),
     cereal::make_nvp("cy", intr[3]), cereal::make_nvp("p1", intr[4]),
     cereal::make_nvp("p2", intr[5]), cereal::make_nvp("p3", intr[6]),
     cereal::make_nvp("p4", intr[7]), cereal::make_nvp("width", width),
     cereal::make_nvp("height", height));

  cam = AbstractCamera<double>::from_data(cam_type, intr.data());
  cam->width() = width;
  cam->height() = height;
}

template <class Archive>
void serialize(Archive& ar, CalibCornerData& c) {
  ar(c.corners, c.corner_ids);
}

template <class Archive>
void serialize(Archive& ar, CalibInitPoseData& c) {
  ar(c.T_a_c, c.num_inliers, c.reprojected_corners);
}

template <class Archive>
void serialize(Archive& ar, Sophus::SE3d& p) {
  ar(cereal::make_nvp("px", p.translation()[0]),
     cereal::make_nvp("py", p.translation()[1]),
     cereal::make_nvp("pz", p.translation()[2]),
     cereal::make_nvp("qx", p.so3().data()[0]),
     cereal::make_nvp("qy", p.so3().data()[1]),
     cereal::make_nvp("qz", p.so3().data()[2]),
     cereal::make_nvp("qw", p.so3().data()[3]));
}

template <class Archive>
void serialize(Archive& ar, Calibration& cam) {
  ar(CEREAL_NVP(cam.T_i_c), CEREAL_NVP(cam.intrinsics));
}

template <class Archive, class Scalar, class CamT>
void serialize(Archive& ar, LoadCalibration<Scalar, CamT>& cam) {
  ar(CEREAL_NVP(cam.T_i_c), CEREAL_NVP(cam.intrinsics));
}

template <class Archive>
void serialize(Archive& ar, MatchData& m) {
  ar(CEREAL_NVP(m.T_i_j), CEREAL_NVP(m.inliers), CEREAL_NVP(m.matches));
}

template <class Archive>
void serialize(Archive& ar, LandmarkMatchData& m) {
  ar(CEREAL_NVP(m.T_w_c), CEREAL_NVP(m.inliers), CEREAL_NVP(m.matches));
}

template <class Archive>
void serialize(Archive& ar, KeypointsData& m) {
  ar(CEREAL_NVP(m.corners), CEREAL_NVP(m.corner_angles),
     CEREAL_NVP(m.corner_descriptors));
}

template <class Archive>
void serialize(Archive& ar, Camera& c) {
  ar(CEREAL_NVP(c.T_w_c));
}

template <class Archive>
void serialize(Archive& ar, Landmark& lm) {
  ar(CEREAL_NVP(lm.p), CEREAL_NVP(lm.obs), CEREAL_NVP(lm.outlier_obs));
}

template <class Archive>
void serialize(Archive& ar, FrameCamId& fcid) {
  ar(fcid.frame_id, fcid.cam_id);
}

}  // namespace cereal
