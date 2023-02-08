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

#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>

#include "visnav/keypoints.h"
#include "visnav/matching_utils.h"

#include "visnav/serialization.h"

#include "visnav/bow_db.h"
#include "visnav/bow_voc.h"

#include <fstream>
#include <random>
#include <chrono>

using namespace visnav;

const int NUM_FEATURES = 1500;
const int MATCH_THRESHOLD = 70;
const double DIST_2_BEST = 1.2;

const std::string img0_path = "../../test/ex3_test_data/0_0.jpg";
const std::string img1_path = "../../test/ex3_test_data/0_1.jpg";

const std::string kd0_path = "../../test/ex3_test_data/kd0.json";
const std::string kd1_path = "../../test/ex3_test_data/kd1.json";

const std::string matches_stereo_path =
    "../../test/ex3_test_data/matches_stereo.json";
const std::string matches_path = "../../test/ex3_test_data/matches.json";

const std::string calib_path = "../../test/ex3_test_data/calib.json";

const std::string vocab_path = "../../data/ORBvoc.cereal";
const std::string bow_res_path = "../../test/ex3_test_data/bow_res.json";
const std::string bow_res2_path = "../../test/ex3_test_data/bow_res2.json";

const std::string bow_db_path = "../../test/ex3_test_data/bow_db.json";
const std::string bow_dist_path = "../../test/ex3_test_data/bow_dist.cereal";

TEST(Ex3TestSuite, KeypointAngles) {
  pangolin::ManagedImage<uint8_t> img0 = pangolin::LoadImage(img0_path);
  pangolin::ManagedImage<uint8_t> img1 = pangolin::LoadImage(img1_path);

  KeypointsData kd0, kd1, kd0_loaded, kd1_loaded;

  detectKeypointsAndDescriptors(img0, kd0, NUM_FEATURES, true);
  detectKeypointsAndDescriptors(img1, kd1, NUM_FEATURES, true);

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  ASSERT_TRUE(kd0_loaded.corner_angles.size() == kd0.corner_angles.size());
  ASSERT_TRUE(kd1_loaded.corner_angles.size() == kd1.corner_angles.size());

  for (size_t i = 0; i < kd0_loaded.corner_angles.size(); i++) {
    ASSERT_TRUE(std::abs(kd0_loaded.corner_angles[i] - kd0.corner_angles[i]) <
                1e-8);
  }

  for (size_t i = 0; i < kd1_loaded.corner_angles.size(); i++) {
    ASSERT_TRUE(std::abs(kd1_loaded.corner_angles[i] - kd1.corner_angles[i]) <
                1e-8);
  }
}

TEST(Ex3TestSuite, KeypointDescriptors) {
  pangolin::ManagedImage<uint8_t> img0 = pangolin::LoadImage(img0_path);
  pangolin::ManagedImage<uint8_t> img1 = pangolin::LoadImage(img1_path);

  KeypointsData kd0, kd1, kd0_loaded, kd1_loaded;

  detectKeypointsAndDescriptors(img0, kd0, NUM_FEATURES, true);
  detectKeypointsAndDescriptors(img1, kd1, NUM_FEATURES, true);

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  ASSERT_TRUE(kd0_loaded.corner_descriptors.size() ==
              kd0.corner_descriptors.size());
  ASSERT_TRUE(kd1_loaded.corner_descriptors.size() ==
              kd1.corner_descriptors.size());

  for (size_t i = 0; i < kd0_loaded.corner_descriptors.size(); i++) {
    ASSERT_TRUE((kd0_loaded.corner_descriptors[i] ^ kd0.corner_descriptors[i])
                    .count() == 0);
  }

  for (size_t i = 0; i < kd1_loaded.corner_descriptors.size(); i++) {
    ASSERT_TRUE((kd1_loaded.corner_descriptors[i] ^ kd1.corner_descriptors[i])
                    .count() == 0);
  }
}

TEST(Ex3TestSuite, DescriptorMatching) {
  MatchData md, md_loaded;
  KeypointsData kd0_loaded, kd1_loaded;

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  matchDescriptors(kd0_loaded.corner_descriptors, kd1_loaded.corner_descriptors,
                   md.matches, MATCH_THRESHOLD, DIST_2_BEST);

  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  // order of matches shouldn't matter. Sort them by cam0 descriptor index.
  auto order = [](auto& a, auto& b) { return a.first < b.first; };
  std::sort(md.matches.begin(), md.matches.end(), order);
  std::sort(md_loaded.matches.begin(), md_loaded.matches.end(), order);

  ASSERT_TRUE(md_loaded.matches.size() == md.matches.size())
      << "md_loaded.matches.size() " << md_loaded.matches.size()
      << " md.matches.size() " << md.matches.size();

  for (size_t i = 0; i < md_loaded.matches.size(); i++) {
    ASSERT_TRUE(md_loaded.matches[i] == md.matches[i]);
  }
}

TEST(Ex3TestSuite, KeypointsAll) {
  pangolin::ManagedImage<uint8_t> img0 = pangolin::LoadImage(img0_path);
  pangolin::ManagedImage<uint8_t> img1 = pangolin::LoadImage(img1_path);

  MatchData md, md_loaded;
  KeypointsData kd0, kd1, kd0_loaded, kd1_loaded;

  detectKeypointsAndDescriptors(img0, kd0, NUM_FEATURES, true);
  detectKeypointsAndDescriptors(img1, kd1, NUM_FEATURES, true);

  matchDescriptors(kd0.corner_descriptors, kd1.corner_descriptors, md.matches,
                   MATCH_THRESHOLD, DIST_2_BEST);

  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  // order of matches shouldn't matter. Sort them by cam0 descriptor index.
  auto order = [](auto& a, auto& b) { return a.first < b.first; };
  std::sort(md.matches.begin(), md.matches.end(), order);
  std::sort(md_loaded.matches.begin(), md_loaded.matches.end(), order);

  ASSERT_TRUE(md_loaded.matches.size() == md.matches.size())
      << "md_loaded.matches.size() " << md_loaded.matches.size()
      << " md.matches.size() " << md.matches.size();

  for (size_t i = 0; i < md_loaded.matches.size(); i++) {
    ASSERT_TRUE(md_loaded.matches[i] == md.matches[i]);
  }
}

TEST(Ex3TestSuite, EpipolarInliers) {
  Calibration calib;

  MatchData md, md_loaded;
  KeypointsData kd0_loaded, kd1_loaded;

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib);
    } else {
      ASSERT_TRUE(false) << "could not load camera ";
    }
  }

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  {
    std::ifstream os(matches_stereo_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  // Essential matrix
  Eigen::Matrix3d E;
  Sophus::SE3d T_0_1 = calib.T_i_c[0].inverse() * calib.T_i_c[1];

  computeEssential(T_0_1, E);

  md.matches = md_loaded.matches;
  findInliersEssential(kd0_loaded, kd1_loaded, calib.intrinsics[0],
                       calib.intrinsics[1], E, 1e-3, md);

  // Order of inliers shouldn't matter, even though there is no reason to change
  // it from matches... Still, sort them by cam0 descriptor index.
  auto order = [](auto& a, auto& b) { return a.first < b.first; };
  std::sort(md.inliers.begin(), md.inliers.end(), order);
  std::sort(md_loaded.inliers.begin(), md_loaded.inliers.end(), order);

  ASSERT_TRUE(md_loaded.inliers.size() == md.inliers.size())
      << "md_loaded.inliers.size() " << md_loaded.inliers.size()
      << " md.inliers.size() " << md.inliers.size();

  for (size_t i = 0; i < md_loaded.inliers.size(); i++) {
    ASSERT_TRUE(md_loaded.inliers[i] == md.inliers[i]);
  }
}

TEST(Ex3TestSuite, RansacInliers) {
  Calibration calib;

  MatchData md, md_loaded;
  KeypointsData kd0_loaded, kd1_loaded;

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib);
    } else {
      ASSERT_TRUE(false) << "could not load camera ";
    }
  }

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0_loaded);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1_loaded);
  }

  {
    std::ifstream os(matches_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(md_loaded);
  }

  md.matches = md_loaded.matches;
  findInliersRansac(kd0_loaded, kd1_loaded, calib.intrinsics[0],
                    calib.intrinsics[1], 1e-5, 20, md);

  // Translation is only determined up to scale, so normalize before comparison.
  const double dist = (md_loaded.T_i_j.translation().normalized() -
                       md.T_i_j.translation().normalized())
                          .norm();
  const double angle = md_loaded.T_i_j.unit_quaternion().angularDistance(
      md.T_i_j.unit_quaternion());

  const int inlier_count_diff =
      std::abs(int(md_loaded.inliers.size()) - int(md.inliers.size()));

  std::set<std::pair<int, int>> md_loaded_inliers(md_loaded.inliers.begin(),
                                                  md_loaded.inliers.end()),
      md_inliers(md.inliers.begin(), md.inliers.end()), md_itersection_inliers;

  // compute set intersection
  size_t max_size = std::max(md_loaded_inliers.size(), md_inliers.size());

  std::set_intersection(
      md_loaded_inliers.begin(), md_loaded_inliers.end(), md_inliers.begin(),
      md_inliers.end(),
      std::inserter(md_itersection_inliers, md_itersection_inliers.begin()));

  double intersection_fraction =
      double(md_itersection_inliers.size()) / max_size;
  ASSERT_TRUE(intersection_fraction > 0.99)
      << "intersection_fraction " << intersection_fraction;

  ASSERT_TRUE(inlier_count_diff < 20) << "inlier " << inlier_count_diff;
  ASSERT_TRUE(dist < 0.05) << "dist " << dist;
  ASSERT_TRUE(angle < 0.01) << "angle " << angle;
}

TEST(Ex3TestSuite, BowSingleFeatureTransform) {
  KeypointsData kd0;
  std::vector<WordId> w_id, w_id_loaded;
  std::vector<WordValue> w_val, w_val_loaded;

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0);
  }

  {
    std::ifstream os(bow_res_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(cereal::make_nvp("w_id", w_id_loaded));
    archive(cereal::make_nvp("w_val", w_val_loaded));
  }

  BowVocabulary voc(vocab_path);

  w_id.resize(kd0.corner_descriptors.size());
  w_val.resize(kd0.corner_descriptors.size());

  for (size_t i = 0; i < kd0.corner_descriptors.size(); i++) {
    voc.transformFeatureToWord(kd0.corner_descriptors[i], w_id[i], w_val[i]);
  }

  ASSERT_TRUE(w_id.size() == w_id_loaded.size());
  ASSERT_TRUE(w_val.size() == w_val_loaded.size());

  for (size_t i = 0; i < kd0.corner_descriptors.size(); i++) {
    ASSERT_TRUE(w_id[i] == w_id_loaded[i]);
    ASSERT_TRUE(std::abs(w_val[i] - w_val_loaded[i]) < 1e-10);
  }
}

TEST(Ex3TestSuite, BowVectorTransform) {
  KeypointsData kd0, kd1;

  BowVector bow0, bow1, bow0_loaded, bow1_loaded;

  {
    std::ifstream os(kd0_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd0);
  }

  {
    std::ifstream os(kd1_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(kd1);
  }

  {
    std::ifstream os(bow_res2_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(cereal::make_nvp("bow0", bow0_loaded));
    archive(cereal::make_nvp("bow1", bow1_loaded));
  }

  BowVocabulary voc(vocab_path);
  voc.transform(kd0.corner_descriptors, bow0);
  voc.transform(kd1.corner_descriptors, bow1);

  {
    std::unordered_map<WordId, WordValue> m(bow0_loaded.begin(),
                                            bow0_loaded.end());
    double sum = 0;
    for (const auto& kv : bow0) {
      sum += std::abs(kv.second);

      auto it = m.find(kv.first);
      ASSERT_TRUE(it != m.end());
      ASSERT_TRUE(std::abs(it->second - kv.second) < 1e-10);
    }

    ASSERT_TRUE(std::abs(sum - 1.0) < 1e-10) << sum;
  }

  {
    std::unordered_map<WordId, WordValue> m(bow1_loaded.begin(),
                                            bow1_loaded.end());

    double sum = 0;
    for (const auto& kv : bow1) {
      sum += std::abs(kv.second);

      auto it = m.find(kv.first);
      ASSERT_TRUE(it != m.end());
      ASSERT_TRUE(std::abs(it->second - kv.second) < 1e-10);
    }

    ASSERT_TRUE(std::abs(sum - 1.0) < 1e-10) << sum;
  }
}

TEST(Ex3TestSuite, BowDBInsert) {
  KeypointsData kd0, kd1;
  std::ifstream os_0(kd0_path, std::ios::binary);
  cereal::JSONInputArchive archive_0(os_0);
  archive_0(kd0);
  std::ifstream os_1(kd1_path, std::ios::binary);
  cereal::JSONInputArchive archive_1(os_1);
  archive_1(kd1);
  // merge all descriptors into one vector
  std::vector<std::bitset<256>> descriptors;
  descriptors.reserve(
      4 * (kd0.corner_descriptors.size() + kd1.corner_descriptors.size()));
  descriptors.insert(descriptors.end(), kd0.corner_descriptors.begin(),
                     kd0.corner_descriptors.end());
  descriptors.insert(descriptors.end(), kd1.corner_descriptors.begin(),
                     kd1.corner_descriptors.end());
  descriptors.insert(descriptors.end(), descriptors.begin(), descriptors.end());
  descriptors.insert(descriptors.end(), descriptors.begin(), descriptors.end());

  int num_descriptors = int(descriptors.size());
  int num_descriptors_per_image = 100;
  int N = num_descriptors / num_descriptors_per_image;  // Num images N = 43

  BowVocabulary voc(vocab_path);
  BowDatabase bow_db;
  std::vector<BowVector> bows(N);

  for (int j = 0; j < N; j++) {
    std::vector<std::bitset<256>> v(
        descriptors.begin() + j * num_descriptors_per_image,
        descriptors.begin() + (j + 1) * num_descriptors_per_image);
    voc.transform(v, bows[j]);
    FrameCamId fcid(j, 0);
    // This insertion is being tested
    bow_db.insert(fcid, bows[j]);
  }
  // Load from saved reference data.
  BowDBInverseIndex inverseIndexLoaded;
  {
    std::ifstream os(bow_db_path, std::ios::binary);
    cereal::JSONInputArchive archive(os);
    archive(inverseIndexLoaded);
  }

  for (const auto& kv : bow_db.getInvertedIndex()) {
    auto iterIndexLoaded = inverseIndexLoaded.find(kv.first);
    ASSERT_TRUE(iterIndexLoaded != inverseIndexLoaded.end());

    std::vector<std::pair<FrameCamId, WordValue>> indexVector(kv.second.begin(),
                                                              kv.second.end());
    ASSERT_TRUE(indexVector.size() == iterIndexLoaded->second.size());
    // Order of frames doesn't matter, but we sort it by WordValue, FrameCamId
    // to compare with loaded vector.
    const auto order = [](const auto& a, const auto& b) {
      return a.second < b.second || (a.second == b.second && a.first < b.first);
    };
    std::sort(indexVector.begin(), indexVector.end(), order);
    std::sort(iterIndexLoaded->second.begin(), iterIndexLoaded->second.end(),
              order);

    for (size_t i = 0; i < indexVector.size(); i++) {
      ASSERT_TRUE(indexVector[i].first == iterIndexLoaded->second[i].first);
      ASSERT_TRUE(std::abs(indexVector[i].second -
                           iterIndexLoaded->second[i].second) < 1e-10);
    }
    indexVector.clear();
  }
}

void test_query_result(const BowQueryResult& dists_ref,
                       const BowQueryResult& result, int k) {
  // Assumption: dists_ref contains the ordered distances for ALL possible fcids
  // in the database. So the size is >= the size of result and we should find
  // for all entries in result the corresponding distance in dists_ref. And the
  // size of dists_ref is equal to the size of the database.

  // The exact order of the query result is not uniquely defined. E.g. if the
  // there are multiple images with the same score, the order is arbitrary. We
  // therefore do the following tests:
  // 1. Check that the result size is as expected.
  // 2. Check that the result is indeed sorted by the score.
  // 3. Check that the result's scores are close to the expected.
  // 4. Check that the score of the last element in the result is close to the
  //    element at the corresponding position in the reference.

  // 1. Check that the result size is as expected.
  const int N =
      int(dists_ref.size());  // this is the number of elements in the database.
  const int num_res = int(result.size());
  if (num_res < std::min(k, N)) {
    // this is only right, if there are only num_res results where there is at
    // least 1 word that matched. images that have 0 words overlap will not
    // appear in the result vector. Therefore we test if the reference data has
    // score 2.0 at index num_res (and therefore also for all indices above,
    // since 2.0 is the highest score and the list is sorted)
    EXPECT_NEAR(2.0, dists_ref[num_res].second, 1e-10)
        << "Query result has less than the expected number of results, but "
           "there are still more elements in the database with score < 2.0 "
           "according to the test data. Size: "
        << num_res << ", k = " << k << ", N = " << N;
  } else {
    ASSERT_EQ(std::min(k, N), num_res)
        << "Query result has more than the expected number of elements. Size: "
        << num_res << ", k = " << k << ", N = " << N;
  }

  // 2. Check that the result is indeed sorted by the score.
  for (size_t i = 1; i < result.size(); ++i) {
    EXPECT_LE(result[i - 1].second, result[i].second)
        << "Query result not sorted by score at index " << i - 1;
  }

  // 3. Check that the result's scores are close to the expected.

  // Create map for lookup of scores
  std::unordered_map<FrameCamId, double> dists_map(dists_ref.begin(),
                                                   dists_ref.end());
  for (const auto& p : result) {
    ASSERT_TRUE(dists_map.count(p.first))
        << "Unknown fcid " << p.first << " in query result";
    EXPECT_NEAR(dists_map.at(p.first), p.second, 1e-10)
        << "Score (distance) of query result " << p.second
        << " does not match score (distance) of test data "
        << dists_map.at(p.first) << " for fcid " << p.first;
  }

  // 4. Check that the score of the last element in the result is close to the
  //    element at the corresponding position in the reference.
  int last_idx = int(result.size()) - 1;
  EXPECT_NEAR(dists_ref[last_idx].second, result[last_idx].second, 1e-10);
}

TEST(Ex3TestSuite, BowDBQuery) {
  KeypointsData kd0, kd1;
  std::ifstream os_0(kd0_path, std::ios::binary);
  cereal::JSONInputArchive archive_0(os_0);
  archive_0(kd0);
  std::ifstream os_1(kd1_path, std::ios::binary);
  cereal::JSONInputArchive archive_1(os_1);
  archive_1(kd1);
  // merge all descriptors into one vector
  std::vector<std::bitset<256>> descriptors;
  descriptors.reserve(
      4 * (kd0.corner_descriptors.size() + kd1.corner_descriptors.size()));
  descriptors.insert(descriptors.end(), kd0.corner_descriptors.begin(),
                     kd0.corner_descriptors.end());
  descriptors.insert(descriptors.end(), kd1.corner_descriptors.begin(),
                     kd1.corner_descriptors.end());
  descriptors.insert(descriptors.end(), descriptors.begin(), descriptors.end());
  descriptors.insert(descriptors.end(), descriptors.begin(), descriptors.end());

  int num_descriptors = int(descriptors.size());
  int num_descriptors_per_image = 100;
  int N = num_descriptors / num_descriptors_per_image;  // Num images N = 43

  BowVocabulary voc(vocab_path);
  std::vector<BowVector> bows(N);
  for (int j = 0; j < N; j++) {
    std::vector<std::bitset<256>> v(
        descriptors.begin() + j * num_descriptors_per_image,
        descriptors.begin() + (j + 1) * num_descriptors_per_image);
    voc.transform(v, bows[j]);
  }
  // Matrix of <FrameCamId, distance> between BoWs, where the row `i`
  // represents `distance` between `bows[i]` and `bows[FrameCamId.frame_id]`.
  // BoWs are added into `bow_db` as in previous test `BowDBInsert`, with
  // FrameId equal to the order of bow in `bows` vector.

  Eigen::Matrix<std::pair<FrameCamId, double>, Eigen::Dynamic, Eigen::Dynamic>
      dist_loaded;
  std::ifstream in(bow_dist_path, std::ios::binary);
  cereal::BinaryInputArchive archive_i(in);
  archive_i(dist_loaded);
  // Size of dynamic Eigen matrix is derived from binary file. It is equal to
  // (N, N). `cereal` is reloaded for dynamic matrices.

  // Load bow database (created as in previous test BowDBInsert)
  BowDatabase bow_db_loaded;
  bow_db_loaded.load(bow_db_path);

  int num_bow_candidates_small =
      4;  // smaller than than database neighbors size
  int num_bow_candidates_large = N + 4;  // larger than database neighbors size

  for (int i = 0; i < N; i++) {
    // Create reference scores for ALL fcids in the database, and sort by
    // scores.
    BowQueryResult sorted_dists;
    for (int j = 0; j < N; j++) {
      sorted_dists.emplace_back(dist_loaded(i, j));
    }
    std::sort(sorted_dists.begin(), sorted_dists.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });

    BowQueryResult res_small;
    // this query is what we test:
    bow_db_loaded.query(bows[i], num_bow_candidates_small, res_small);

    test_query_result(sorted_dists, res_small, num_bow_candidates_small);

    BowQueryResult res_large;
    // this query is what we test:
    bow_db_loaded.query(bows[i], num_bow_candidates_large, res_large);

    test_query_result(sorted_dists, res_large, num_bow_candidates_large);
  }
}

// For now we disable this test, b/c it is too brittle. It was working ok when
// at some point the implementation was sorting the result deterministically by
// sorting equal scores by fcid. But now that we relaxed this requirement again,
// it fails often. The thresholds might now be different. But it could as well
// depend a lot on the hardwere where we test.
#if 0
TEST(Ex3TestSuite, BowDBSortPerformance) {


  // Here we test if query() with small number of results for a large database
  // is implemented efficiently. For large databases, fully sorting the whole
  // vector of distances can become an avoidable bottleneck. Use of partial_sort
  // can avoid this. However, beware that partial_sort can also be (a little
  // bit) slower than sort for smaller arrays (but we don't consider this a
  // problem).

  constexpr int Dsize = 256;
  constexpr int N = 10000;
  // NF is small in order to make the sort operation the most time consuming
  // part. Otherwise distance computation takes the most time.
  constexpr int NF = 2;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::bernoulli_distribution d(0.5);
  auto rand_desc = [&]() {
    std::bitset<Dsize> bits;
    for (int n = 0; n < Dsize; ++n) {
      bits[n] = d(gen);
    }
    return bits;
  };

  BowVocabulary voc(vocab_path);
  BowDatabase bow_db;

  std::vector<BowVector> bows(N);

  std::vector<std::bitset<Dsize>> v;
  v.resize(NF);
  for (int i = 0; i < NF; i++) {
    v[i] = rand_desc();
  }

  voc.transform(v, bows[0]);

  for (int j = 0; j < N; j++) {
    // Same bow for all frames. It ensures that the number of neighbors in
    // inverse index of bow_db will be equal to N.
    // Otherwise results.size() was around 15 when N=10000.
    FrameCamId fcid(j, 0);
    bow_db.insert(fcid, bows[0]);
  }

  constexpr int num_bow_candidates_small = 4;
  constexpr int num_bow_candidates_large = N + 1;
  double avg_time_small_query = 0;
  double avg_time_large_query = 0;
  int num_runs = 5;
  int NQuery = 1;

  for (int j = 0; j < NQuery; j++) {
    BowQueryResult res_small, res_large;
    for (int run_id = 0; run_id < num_runs; run_id++) {
      auto t1 = std::chrono::high_resolution_clock::now();
      bow_db.query(bows[0], num_bow_candidates_small, res_small);
      auto t2 = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> ms_double = t2 - t1;
      avg_time_small_query += ms_double.count();
      ASSERT_TRUE(res_small.size() <= num_bow_candidates_small);
      res_small.clear();
    }

    for (int run_id = 0; run_id < num_runs; run_id++) {
      auto t1 = std::chrono::high_resolution_clock::now();
      bow_db.query(bows[0], num_bow_candidates_large, res_large);
      // Resizing takes time!
      res_large.resize(num_bow_candidates_small);
      auto t2 = std::chrono::high_resolution_clock::now();

      std::chrono::duration<double, std::milli> ms_double = t2 - t1;
      avg_time_large_query += ms_double.count();
      ASSERT_TRUE(res_large.size() <= num_bow_candidates_large &&
                  res_large.size() <= N);
      res_large.clear();
    }
    res_small.clear();
    res_large.clear();
  }
  avg_time_small_query = avg_time_small_query / double(num_runs * NQuery);
  avg_time_large_query = avg_time_large_query / double(num_runs * NQuery);

  //  double r = (avg_time_large_query - avg_time_small_query) * 200 /
  //             (avg_time_large_query + avg_time_small_query);

  double diff = (avg_time_large_query - avg_time_small_query) / double(N);

  // diff over N was (2.7 +- 0.3) * 1e-5 (almost do not depend on N).
  // r highly depends on the number N.
  // Partial sort O(n * log(k)), sort O(n * log(n)).

  // According to experimental data:
  // *Conclusion*: `partial_sort` is better than `sort` when `k` is less than
  // 15% of the whole array, i.e k <= 0.15 * N. N needs to be >= 1000.

  ASSERT_GE(diff, 5.0 * 1e-6)
      << "Time of query for small number of bow candidates (<<N) = "
      << std::setprecision(4) << avg_time_small_query
      << " ms,\ntime of query for large number of bow candidates (>N) = "
      << avg_time_large_query
      << " ms,\nthe performance difference over N, i.e. "
         "(time_large - time_small) / N = "
      << diff
      << " ,\nbut should be greater than 1e-5. (Hint: Use partial_sort.)";
}
#endif
