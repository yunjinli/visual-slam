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

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

// Implement skew-symmetric matrix
template <class T>
Eigen::Matrix<T, 3, 3> get_skew_matrix3(const Eigen::Matrix<T, 3, 1>& xi) {
  Eigen::Matrix<T, 3, 3> x_hat;

  x_hat << 0, -xi(2, 0), xi(1, 0), xi(2, 0), 0, -xi(0, 0), -xi(1, 0), xi(0, 0),
      0;

  return x_hat;
}
// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {
  // TODO SHEET 1: implement
  // Based on Rodrigues' Formula
  Eigen::Matrix<T, 3, 3> R;
  Eigen::Matrix<T, 3, 3> I = Eigen::Matrix3d::Identity();
  double theta = xi.norm();
  //    std::cout << "The norm of w is :" << theta << "\n";
  if (theta == 0) {
    return I;
  } else {
    Eigen::Matrix<T, 3, 1> v = xi / theta;
    Eigen::Matrix<T, 3, 3> v_hat = get_skew_matrix3<T>(v);
    R = I + sin(theta) * v_hat + (1 - cos(theta)) * v_hat * v_hat;
    return R;
  }
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {
  // TODO SHEET 1: implement
  double theta = acos((mat.trace() - 1) / 2);
  Eigen::Matrix<T, 3, 1> w;

  if (theta == 0) {
    w << 0, 0, 0;
  } else {
    w << mat(2, 1) - mat(1, 2), mat(0, 2) - mat(2, 0), mat(1, 0) - mat(0, 1);

    w *= theta / (2 * sin(theta));
  }

  return w;
}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 4, 4> SE3;
  Eigen::Matrix<T, 3, 1> v = xi.block(0, 0, 3, 1);
  Eigen::Matrix<T, 3, 1> w = xi.block(3, 0, 3, 1);
  Eigen::Matrix<T, 3, 3> R = user_implemented_expmap<T>(w);
  Eigen::Matrix<T, 3, 1> trans;
  SE3.block(0, 0, 3, 3) = R;
  SE3(3, 0) = 0;
  SE3(3, 1) = 0;
  SE3(3, 2) = 0;
  SE3(3, 3) = 1;
  Eigen::Matrix<T, 3, 3> w_hat = get_skew_matrix3<T>(w);

  double theta = w.norm();
  if (theta == 0) {
    trans = v;
  } else {
    Eigen::Matrix<T, 3, 3> I = Eigen::Matrix3d::Identity();
    trans = (I + ((1 - cos(theta)) / (theta * theta)) * w_hat +
             ((theta - sin(theta)) / (theta * theta * theta)) * w_hat * w_hat) *
            v;
  }
  SE3.block(0, 3, 3, 1) = trans;

  return SE3;
}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {
  // TODO SHEET 1: implement
  Eigen::Matrix<T, 6, 1> twist;
  Eigen::Matrix<T, 3, 3> R = mat.block(0, 0, 3, 3);
  Eigen::Matrix<T, 3, 1> w = user_implemented_logmap(R);
  Eigen::Matrix<T, 3, 1> v;
  Eigen::Matrix<T, 3, 1> trans = mat.block(0, 3, 3, 1);
  double theta = w.norm();

  if (theta == 0) {
    v = trans;
  } else {
    Eigen::Matrix<T, 3, 3> w_hat = get_skew_matrix3<T>(w);
    Eigen::Matrix<T, 3, 3> I = Eigen::Matrix3d::Identity();
    v = (I + ((1 - cos(theta)) / (theta * theta)) * w_hat +
         ((theta - sin(theta)) / (theta * theta * theta)) * w_hat * w_hat)
            .inverse() *
        trans;
  }

  twist.block(0, 0, 3, 1) = v;
  twist.block(3, 0, 3, 1) = w;

  return twist;
}

}  // namespace visnav
