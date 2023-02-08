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

#include <fstream>

#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_vector.h>

#include <visnav/common_types.h>
#include <visnav/serialization.h>

namespace visnav {

class BowDatabase {
 public:
  BowDatabase() {}

  inline void insert(const FrameCamId& fcid, const BowVector& bow_vector) {
    // TODO SHEET 3: add a bow_vector that corresponds to frame fcid to the
    // inverted index. You can assume the image hasn't been added before.
    for (auto& vi : bow_vector) {
      inverted_index[vi.first].emplace_back(std::make_pair(fcid, vi.second));
    }
  }
  inline void query(const BowVector& bow_vector, size_t num_results,
                    BowQueryResult& results) const {
    // TODO SHEET 3: find num_results closest matches to the bow_vector in the
    // inverted index. Hint: for good query performance use std::unordered_map
    // to accumulate scores and std::partial_sort for getting the closest
    // results. You should use L1 difference as the distance measure. You can
    // assume that BoW descripors are L1 normalized.

    /* Short Reminder
     * BowVector = std::vector<std::pair<WordId, WordValue>>
     *
     * BowDBInverseIndexConcurrent = tbb::concurrent_unordered_map<WordId,
     * tbb::concurrent_vector<std::pair<FrameCamId, WordValue>>>
     *
     * BowQueryResult = std::vector<std::pair<FrameCamId, double>>
     */

    /*
     * Use the formula given by the nister2006cvpr_vocab-tree paper
     * |q-d|=2+sigma_(i|qi !=0, di != 0){|qi - di| - |qi| - di}
     * Accodding to the formula, we only do some action when a specific wordId
     * could be found in both the query and the database
     */
    // My thought: Iterate through the inverted_index, check if the wordId can
    // be also found in query, append the score with respected to the fcid
    auto find_wid_in_query = [&v = bow_vector](WordId wid) {
      //      std::pair<WordId, WordValue>* ptr = nullptr;
      int found_idx = -1;
      int idx = 0;
      for (const auto& kv : v) {
        if (kv.first == wid) {
          found_idx = idx;
          break;
        } else {
          // Do nothing
        }
        idx++;
      }
      return found_idx;
    };

    std::unordered_map<FrameCamId, double> image_score_map;

    auto has_key = [&m = image_score_map](FrameCamId key) {
      for (const auto& kv : m) {
        if (kv.first.frame_id == key.frame_id &&
            kv.first.cam_id == key.cam_id) {
          return true;
        } else {
        }
      }
      return false;
    };

    for (const auto& kv : inverted_index) {
      int index_found_in_query = find_wid_in_query(kv.first);
      if (index_found_in_query != -1) {
        // Iterate through the fcid in this wordId in the inverted_index
        for (const auto& fcid : kv.second) {
          // fcid.first -> FrameCamId
          // fcid.second -> Weight
          /*
           * Check if our map already have specific FrameCamId
           */
          if (has_key(fcid.first)) {  // we add value to the existing value
            image_score_map.at(fcid.first) +=
                std::abs(bow_vector[index_found_in_query].second -
                         fcid.second) -
                std::abs(bow_vector[index_found_in_query].second) -
                std::abs(fcid.second);
          } else {  // we push back new fcid, score pair
            image_score_map.emplace(std::make_pair(
                fcid.first,
                std::abs(bow_vector[index_found_in_query].second -
                         fcid.second) -
                    std::abs(bow_vector[index_found_in_query].second) -
                    std::abs(fcid.second)));
          }
        }
      } else {
        // Do nothing
      }
    }

    for (auto& m : image_score_map) {
      m.second += 2;
    }
    std::vector<std::pair<FrameCamId, double>> image_score_vec{
        image_score_map.begin(), image_score_map.end()};
    std::partial_sort(
        image_score_vec.begin(),
        image_score_vec.begin() + std::min(image_score_vec.size(), num_results),
        image_score_vec.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    for (int i = 0; i < std::min(image_score_vec.size(), num_results); i++) {
      results.push_back(image_score_vec[i]);
    }
  }

  void clear() { inverted_index.clear(); }

  void save(const std::string& out_path) {
    BowDBInverseIndex state;
    for (const auto& kv : inverted_index) {
      for (const auto& a : kv.second) {
        state[kv.first].emplace_back(a);
      }
    }
    std::ofstream os;
    os.open(out_path, std::ios::binary);
    cereal::JSONOutputArchive archive(os);
    archive(state);
  }

  void load(const std::string& in_path) {
    BowDBInverseIndex inverseIndexLoaded;
    {
      std::ifstream os(in_path, std::ios::binary);
      cereal::JSONInputArchive archive(os);
      archive(inverseIndexLoaded);
    }
    for (const auto& kv : inverseIndexLoaded) {
      for (const auto& a : kv.second) {
        inverted_index[kv.first].emplace_back(a);
      }
    }
  }

  const BowDBInverseIndexConcurrent& getInvertedIndex() {
    return inverted_index;
  }

 protected:
  BowDBInverseIndexConcurrent inverted_index;
};  // namespace visnav

}  // namespace visnav
