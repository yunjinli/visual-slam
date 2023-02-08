/**
BSD 3-Clause License

Copyright (c) 2021, Vladyslav Usenko and Nikolaus Demmel.
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
#include <sophus/so3.hpp>

#include <visnav/common_types.h>

namespace visnav {

struct RotationAveragingCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RotationAveragingCostFunctor(const Sophus::SO3d& R_i_j) : R_i_j_(R_i_j) {}

  template <class T>
  bool operator()(T const* const sR_w_i, T const* const sR_w_j,
                  T* sResiduals) const {
    // map inputs
    Eigen::Map<Sophus::SO3<T> const> const R_w_i(sR_w_i);
    Eigen::Map<Sophus::SO3<T> const> const R_w_j(sR_w_j);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(sResiduals);

    residuals = (R_i_j_ * R_w_j.inverse() * R_w_i).log();

    return true;
  }

  const Sophus::SO3d R_i_j_;
};

struct TranslationAveragingCostFunctor {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  TranslationAveragingCostFunctor(const Eigen::Vector3d& t_hat_i_j)
      : t_hat_i_j_(t_hat_i_j) {}

  template <class T>
  bool operator()(T const* const st_w_i, T const* const st_w_j,
                  T* sResiduals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const t_w_i(st_w_i);
    Eigen::Map<Eigen::Matrix<T, 3, 1> const> const t_w_j(st_w_j);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(sResiduals);

    Eigen::Matrix<T, 3, 1> diff = t_w_j - t_w_i;

    residuals = t_hat_i_j_ - diff / (diff.norm() + 1e-6);

    return true;
  }

  const Eigen::Vector3d t_hat_i_j_;
};

}  // namespace visnav
