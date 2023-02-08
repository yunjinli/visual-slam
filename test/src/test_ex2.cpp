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

#include "visnav/camera_models.h"

using namespace visnav;

template <typename CamT>
void test_project_unproject() {
  CamT cam = CamT::getTestProjections();

  typedef typename CamT::Vec2 Vec2;
  typedef typename CamT::Vec3 Vec3;

  for (int x = -10; x <= 10; x++) {
    for (int y = -10; y <= 10; y++) {
      Vec3 p(x, y, 5);

      Vec3 p_normalized = p.normalized();
      Vec2 res = cam.project(p);
      Vec3 p_uproj = cam.unproject(res);

      ASSERT_TRUE(p_normalized.isApprox(p_uproj))
          << "p_normalized " << p_normalized.transpose() << " p_uproj "
          << p_uproj.transpose();
    }
  }
}

TEST(Ex2TestSuite, PinholeProjectUnproject) {
  test_project_unproject<PinholeCamera<double>>();
}

TEST(Ex2TestSuite, ExtendedUnifiedProjectUnproject) {
  test_project_unproject<ExtendedUnifiedCamera<double>>();
}

TEST(Ex2TestSuite, DoubleSphereProjectUnproject) {
  test_project_unproject<DoubleSphereCamera<double>>();
}

TEST(Ex2TestSuite, KannalaBrandt4ProjectUnproject) {
  test_project_unproject<KannalaBrandt4Camera<double>>();
}
