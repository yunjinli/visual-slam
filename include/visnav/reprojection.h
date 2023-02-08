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

#include <memory>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <visnav/common_types.h>
#include <visnav/imu/preintegration.h>

namespace visnav {

template <class T>
class AbstractCamera;

struct ReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                          const Eigen::Vector3d& p_3d,
                          const std::string& cam_model)
      : p_2d(p_2d), p_3d(p_3d), cam_model(cam_model) {}

  template <class T>
  bool operator()(T const* const sT_w_i, T const* const sT_i_c,
                  T const* const sIntr, T* sResiduals) const {
    Eigen::Map<Sophus::SE3<T> const> const T_w_i(sT_w_i);
    Eigen::Map<Sophus::SE3<T> const> const T_i_c(sT_i_c);

    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);

    // TODO SHEET 2: implement the rest of the functor

    Eigen::Matrix<T, 3, 1> p_c_3d =
        (T_w_i * T_i_c).inverse().rotationMatrix() * p_3d +
        (T_w_i * T_i_c).inverse().translation();

    Eigen::Matrix<T, 2, 1> p_2d_hat = cam->project(p_c_3d);

    residuals = p_2d - p_2d_hat;

    return true;
  }

  Eigen::Vector2d p_2d;
  Eigen::Vector3d p_3d;
  std::string cam_model;
};

struct BundleAdjustmentReprojectionCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentReprojectionCostFunctor(const Eigen::Vector2d& p_2d,
                                          const std::string& cam_model)
      : p_2d(p_2d), cam_model(cam_model) {}

  template <class T>
  bool operator()(T const* const sT_w_c, T const* const sp_3d_w,
                  T const* const sIntr, T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SE3<T> const> const T_w_c(sT_w_c);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const p_3d_w(sp_3d_w);
    Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(sResiduals);
    const std::shared_ptr<AbstractCamera<T>> cam =
        AbstractCamera<T>::from_data(cam_model, sIntr);

    // TODO SHEET 4: Compute reprojection error
    Eigen::Matrix<T, 2, 1> p_2d_w = cam->project(T_w_c.inverse() * p_3d_w);
    residuals = p_2d - p_2d_w;
    return true;
  }

  Eigen::Vector2d p_2d;
  std::string cam_model;
};

// Make the cost functor for the IMU
struct BundleAdjustmentImuCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BundleAdjustmentImuCostFunctor(
      //      const IntegratedImuMeasurement<double>& imu_meas,
      const PoseVelState<double>& delta_state, const Eigen::Vector3d& g,
      const Timestamp& state0_t_ns,
      const Timestamp& state1_t_ns)  // might have to use int_64t if this breaks
                                     //      : imu_meas(imu_meas),
      : delta_state_(delta_state),
        g(g),
        state0_t_ns(state0_t_ns),
        state1_t_ns(state1_t_ns) {}

  template <class T>
  bool operator()(T const* const sstate0_T_w_i, T const* const sstate1_T_w_i,
                  T const* const sstate0_v_w_i, T const* const sstate1_v_w_i,
                  T* sResiduals) const {
    // Map inputs
    Eigen::Map<Sophus::SE3<T> const> const state0_T_w_i(sstate0_T_w_i);
    Eigen::Map<Sophus::SE3<T> const> const state1_T_w_i(sstate1_T_w_i);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const state0_v_w_i(sstate0_v_w_i);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const state1_v_w_i(sstate1_v_w_i);
    Eigen::Map<Eigen::Matrix<T, POSE_VEL_SIZE, 1>> residuals(sResiduals);
    // Rebuild the state variables
    //    PoseVelState<T> state0(state0_t_ns, state0_T_w_i, state0_v_w_i);
    //    state0.T_w_i = state0_T_w_i;
    //    state0.vel_w_i = state0_v_w_i;
    //    state0.t_ns = state0_t_ns;
    //    PoseVelState<T> state1(state1_t_ns, state1_T_w_i, state1_v_w_i);
    //    state1.T_w_i = state1_T_w_i;
    //    state1.vel_w_i = state1_v_w_i;
    //    state1.t_ns = state1_t_ns;

    // Compute the residuals
    //    residuals =
    //        imu_meas.residual(state0, g, state1, Eigen::Matrix<T, 3,
    //        1>::Zero(),
    //                          Eigen::Matrix<T, 3, 1>::Zero());
    using Vec3 = Eigen::Matrix<T, 3, 1>;
    using VecN = Eigen::Matrix<T, POSE_VEL_SIZE, 1>;
    using Mat3 = Eigen::Matrix<T, 3, 3>;

    double dt = delta_state_.t_ns * (1e-9);
    //    VecN res;

    Mat3 R0_inv = state0_T_w_i.so3().inverse().matrix();
    Vec3 tmp =
        R0_inv * (state1_T_w_i.translation() - state0_T_w_i.translation() -
                  state0_v_w_i * dt - (0.5) * g * dt * dt);

    residuals.template segment<3>(0) = tmp - (delta_state_.T_w_i.translation());
    residuals.template segment<3>(3) =
        (delta_state_.T_w_i.so3() * state1_T_w_i.so3().inverse() *
         state0_T_w_i.so3())
            .log();

    Vec3 tmp2 = R0_inv * (state1_v_w_i - state0_v_w_i - g * dt);
    residuals.template segment<3>(6) = tmp2 - (delta_state_.vel_w_i);
    return true;
  }
  int64_t state0_t_ns;
  int64_t state1_t_ns;
  Eigen::Vector3d g;
  //  IntegratedImuMeasurement<double> imu_meas;
  PoseVelState<double> delta_state_;
};
}  // namespace visnav
