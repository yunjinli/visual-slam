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

#include <gtest/gtest.h>

#include "visnav/ex1.h"

using namespace visnav;

TEST(Ex1TestSuite, SO3ExpMapTest) {
  Sophus::Vector3d xi;
  xi.setZero();

  for (int i = 0; i < 100; i++) {
    Eigen::Matrix3d res1 = user_implemented_expmap(xi);
    Eigen::Matrix3d res2 = Sophus::SO3d::exp(xi).matrix();

    ASSERT_TRUE(res1.isApprox(res2))
        << "res1 " << res1 << "\nres2 " << res2 << "\n";

    xi.setRandom();
  }
}

TEST(Ex1TestSuite, SO3LogMapTest) {
  Sophus::Vector3d xi;
  xi.setZero();

  for (int i = 0; i < 100; i++) {
    Eigen::Matrix3d mat = Sophus::SO3d::exp(xi).matrix();

    Sophus::Vector3d xi1 = user_implemented_logmap(mat);

    ASSERT_TRUE(xi.isApprox(xi1))
        << "xi " << xi.transpose() << "\nxi1 " << xi1.transpose() << "\n";

    xi.setRandom();
    xi /= 10;
  }
}

TEST(Ex1TestSuite, SE3ExpMapTest) {
  Sophus::Vector6d xi;
  xi.setZero();

  for (int i = 0; i < 100; i++) {
    Eigen::Matrix4d res1 = user_implemented_expmap(xi);
    Eigen::Matrix4d res2 = Sophus::SE3d::exp(xi).matrix();

    ASSERT_TRUE(res1.isApprox(res2))
        << "res1 " << res1 << "\nres2 " << res2 << "\n";

    xi.setRandom();

    if (i == 0) xi.head<3>().setZero();
    if (i == 1) xi.tail<3>().setZero();
  }
}

TEST(Ex1TestSuite, SE3LogMapTest) {
  Sophus::Vector6d xi;
  xi.setZero();

  for (int i = 0; i < 100; i++) {
    Eigen::Matrix4d mat = Sophus::SE3d::exp(xi).matrix();

    Sophus::Vector6d xi1 = user_implemented_logmap(mat);

    ASSERT_TRUE(xi.isApprox(xi1))
        << "xi " << xi.transpose() << "\nxi1 " << xi1.transpose() << "\n";

    xi.setRandom();
    xi /= 10;
    if (i == 0) xi.head<3>().setZero();
    if (i == 1) xi.tail<3>().setZero();
  }
}
