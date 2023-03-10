#pragma once

#include <visnav/ORBVocabulary.h>
#include <visnav/bow_db.h>
#include <visnav/common_types.h>
#include <visnav/keypoints.h>
namespace visnav {

/**
 * Connect covisible edges between the new keyframe and the existing collection
 * of keyframes
 *
 * @param[in] new_fcid The FrameCamId of the new keyframe.
 * @param[in] cameras The collection of keyframes.
 * @param[in] landmarks The collection of map points
 * @param[out] new_camera Newly computed keyframe that would be added into
 * `cameras` later.
 * @param[out] graph Add new keyframe as new vertex and update the edges of the
 * graph
 * @param[in] threshold The threshold for number of common map points such that
 * both frames could be considered as covisible pair
 */
void construct_visibility_graph(const FrameCamId& new_fcid,
                                const Cameras& cameras,
                                const Landmarks& landmarks, Camera& new_camera,
                                CovisibilityGraph& graph, int threshold) {
  std::map<FrameCamId, int> share_lm_count;
  for (const auto& tid_lm : landmarks) {
    const TrackId& tid = tid_lm.first;
    const Landmark& lm = tid_lm.second;

    if (lm.all_obs.count(new_fcid)) {
      new_camera.map_points.emplace(
          std::make_pair(tid, lm.all_obs.at(new_fcid)));
    }
    for (const auto& fcid_cam : cameras) {
      const FrameCamId& fcid = fcid_cam.first;
      if (lm.all_obs.count(new_fcid) > 0 && lm.all_obs.count(fcid) > 0) {
        if (share_lm_count.count(fcid)) {
          share_lm_count.at(fcid) += 1;  // increment
        } else {
          share_lm_count.emplace(std::make_pair(fcid, 1));  // initialize
          // share_lm_count.at(fcid) = 1;
        }
      } else {
        // nop
      }
    }
  }
  // iterate throught the share_lm_count and find fcid that is above the
  // threshold
  std::set<FrameCamId> new_edges;
  for (const auto& fcid_count : share_lm_count) {
    if (fcid_count.first.cam_id == 0) {
      if (fcid_count.second >= threshold) {
        new_camera.covisible_weights.emplace(
            std::make_pair(fcid_count.first, fcid_count.second));
        new_edges.insert(fcid_count.first);
        graph[fcid_count.first].insert(new_fcid);
      }
    }
  }
  graph[new_fcid] = new_edges;
}

/**
 * Compute the similarity score for bag-of-word vector
 * @param v1 The bag-of-word vector
 * @param v2 The bag-of-word vector
 * @return Return the score of similarity score
 */
// double compute_bow_score(const BowVector& v1, const BowVector& v2) {
//   double score = 0;
//   auto find_wid_in_query = [&v = v1](WordId wid) {
//     int found_idx = -1;
//     int idx = 0;
//     for (const auto& kv : v) {
//       if (kv.first == wid) {
//         found_idx = idx;
//         break;
//       } else {
//         // Do nothing
//       }
//       idx++;
//     }
//     return found_idx;
//   };
//   // Use the formula given by the nister2006cvpr_vocab-tree paper:
//   // |q-d|=2+sigma_(i|qi !=0, di != 0){|qi - di| - |qi| - di}
//   for (const auto& kv : v2) {
//     // key: WordId
//     // value: Weight
//     int index_found_in_query = find_wid_in_query(kv.first);
//     if (index_found_in_query != -1) {
//       score += std::abs(v1[index_found_in_query].second - kv.second) -
//                std::abs(v1[index_found_in_query].second) -
//                std::abs(kv.second);
//     } else {
//       // nop
//     }
//   }
//   score = -score / 2.0;  // let the score be in range [0, 1]
//   return score;
// }
/**
 * Compute the minumum bag-of-word score for the newly added keyframe with all
 * of its covisible neighbors that have weight above `threshold`
 * @param new_kf Newly computed keyframe that will be added into `keyframes`
 * later
 * @param keyframes The collection of keyframes
 * @param threshold Threshold of the weight of the covisible edge
 * @return min_score The
 */
double compute_min_connected_covisible(const Camera& new_kf,
                                       const Cameras& keyframes,
                                       const ORBVocabulary* voc,
                                       int threshold) {
  double min_score = 1;
  for (const auto& kv : new_kf.covisible_weights) {
    if (kv.second > threshold) {
      // double score = compute_bow_score(new_kf.bow_vector,
      //                                  keyframes.at(kv.first).bow_vector);
      double score =
          voc->score(new_kf.bow_vector, keyframes.at(kv.first).bow_vector);
      if (score < min_score) {
        min_score = score;
      }
    }
  }
  return min_score;
}
/**
 * Get possible loop candidates
 *
 * @param new_kf_fcid
 * @param new_kf Newly computed keyframe that would be added into `keyframes`
 * later.
 * @param keyframes The collection of keyframes.
 * @param graph The covisibility graph
 * @param min_score
 * @param recognition_database The bag-of-word database for the collection of
 * keyframes
 */
std::vector<FrameCamId> detect_loop_candidates(
    const FrameCamId& new_kf_fcid, const Camera& new_kf,
    const Cameras& keyframes, const CovisibilityGraph& graph, double min_score,
    DBoWInvertedFile& recognition_database, const ORBVocabulary* voc) {
  // Search all keyframes that share a word with current keyframes
  // Discard keyframes connected to the query keyframe

  std::set<FrameCamId> connected_frames = graph.at(new_kf_fcid);
  std::unordered_map<FrameCamId, int> num_sharing_words;
  bool has_any_sharing_words = false;

  for (DBoW2::BowVector::const_iterator vit = new_kf.bow_vector.begin(),
                                        vend = new_kf.bow_vector.end();
       vit != vend; vit++) {
    // for (const auto& kv1 : new_kf.bow_vector) {
    std::vector<FrameCamId>& loop_keyframes = recognition_database[vit->first];
    // if (recognition_database->getInvertedIndex().count(kv1.first)) {
    //   tbb::concurrent_vector<std::pair<FrameCamId, WordValue>> loop_keyframes
    //   =
    //       recognition_database->getInvertedIndex().at(kv1.first);
    has_any_sharing_words = true;
    for (const auto& lkf : loop_keyframes) {
      if (!connected_frames.count(
              lkf)) {  // we discard keyframes that connect with in
        // the covisibility graph
        if (num_sharing_words.count(lkf)) {
          num_sharing_words[lkf] += 1;
        } else {
          num_sharing_words[lkf] = 0;
        }
      } else {
        if (new_kf.covisible_weights.at(lkf) < 30) {
          if (num_sharing_words.count(lkf)) {
            num_sharing_words[lkf] += 1;
          } else {
            num_sharing_words[lkf] = 0;
          }
        }
      }
    }
  }

  if (!has_any_sharing_words || num_sharing_words.empty()) {
    return std::vector<FrameCamId>();
  }
  // Only compare against those keyframes that share enough words
  int max_num_sharing_words = 0;
  for (const auto& kv : num_sharing_words) {
    if (kv.second > max_num_sharing_words) {
      max_num_sharing_words = kv.second;
    }
  }
  int sharing_words_threshold = max_num_sharing_words * 0.8f;
  std::vector<std::pair<double, FrameCamId>> loop_score_and_match;
  std::unordered_map<FrameCamId, double> loop_score;
  for (const auto& kv : num_sharing_words) {
    if (kv.second > sharing_words_threshold) {
      // double score = compute_bow_score(new_kf.bow_vector,
      //                                  keyframes.at(kv.first).bow_vector);
      double score =
          voc->score(new_kf.bow_vector, keyframes.at(kv.first).bow_vector);
      loop_score[kv.first] = score;
      if (score >= min_score) {
        loop_score_and_match.push_back(std::make_pair(score, kv.first));
      }
    }
  }
  if (loop_score_and_match.empty()) {
    return std::vector<FrameCamId>();
  }

  std::vector<std::pair<double, FrameCamId>> loop_accscore_and_match;
  double best_acc_score = min_score;

  for (const auto& score_fcid : loop_score_and_match) {
    std::set<FrameCamId> neighbors = graph.at(score_fcid.second);
    double best_score = score_fcid.first;
    double acc_score = score_fcid.first;
    FrameCamId best_fcid = score_fcid.second;
    for (std::set<FrameCamId>::iterator it = neighbors.begin(),
                                        end = neighbors.end();
         it != end; it++) {
      if (num_sharing_words.count(*it)) {
        if (num_sharing_words.at(*it) > sharing_words_threshold) {
          acc_score += loop_score.at(*it);
          if (loop_score.at(*it) > best_score) {
            best_fcid = *it;
            best_score = loop_score.at(*it);
          }
        }
      }
    }
    loop_accscore_and_match.push_back(std::make_pair(acc_score, best_fcid));
    if (acc_score > best_acc_score) best_acc_score = acc_score;
  }

  // Return all those keyframes with a score higher than 0.75*bestScore
  double min_score_to_retain = 0.75f * best_acc_score;

  std::set<FrameCamId> already_added_kf;
  std::vector<FrameCamId> loop_candidates;
  loop_candidates.reserve(loop_accscore_and_match.size());
  for (const auto& score_fcid : loop_score_and_match) {
    if (score_fcid.first > min_score_to_retain) {
      if (!already_added_kf.count(score_fcid.second)) {
        loop_candidates.push_back(score_fcid.second);
        already_added_kf.insert(score_fcid.second);
      }
    }
  }

  // std::partial_sort(
  //     loop_score_and_match.begin(),
  //     loop_score_and_match.begin() +
  //         std::min((int)(loop_score_and_match.size()), 1),
  //     loop_score_and_match.end(),
  //     [](const auto& a, const auto& b) { return a.first > b.first; });
  // for (int i = 0; i < std::min((int)(loop_score_and_match.size()), 1); i++)
  // {
  //   loop_candidates.push_back(loop_score_and_match[i].second);
  // }
  return loop_candidates;
}  // namespace visnav

/**
 * Detect possible loop closure
 *
 * @param new_kf_fcid
 * @param new_kf Newly computed keyframe that would be added into `keyframes`
 * later.
 * @param keyframes The collection of keyframes.
 * @param recognition_database The bag-of-word database for the collection of
 * keyframes
 * @param graph The covisibility graph
 * @param consistent_groups This constains set of frames of the possible loop
 * candidate frames
 * @param enough_consistent_candidates
 * @param threshold The threshold for the connected covisible pair
 * @param num_consistency_threshold
 * @return return true if consistent loop candidate found
 */
void insert_new_kf_to_db(const FrameCamId& new_kf_fcid, const Camera& new_kf,
                         DBoWInvertedFile& recognition_database) {
  for (DBoW2::BowVector::const_iterator vit = new_kf.bow_vector.begin(),
                                        vend = new_kf.bow_vector.end();
       vit != vend; vit++)
    recognition_database[vit->first].push_back(new_kf_fcid);
}
bool detect_loop_closure(const FrameCamId& new_kf_fcid, const Camera& new_kf,
                         const Cameras& keyframes,
                         DBoWInvertedFile& recognition_database,
                         const ORBVocabulary* voc,
                         const CovisibilityGraph& graph,
                         ConsistentGroups& consistent_groups,
                         std::vector<FrameCamId>& enough_consistent_candidates,
                         int threshold, int num_consistency_threshold) {
  double min_score = compute_min_connected_covisible(
      new_kf, keyframes, voc,
      threshold);  // This minimum score would be use to fine the loop
                   // candidate
  // std::vector<FrameCamId> candidate =
  // recognition_database->query(new_kf.bow_vector, min_score, results);
  std::vector<FrameCamId> loop_candidates =
      detect_loop_candidates(new_kf_fcid, new_kf, keyframes, graph, min_score,
                             recognition_database, voc);
  // If no possible loop candidate found, we clear all the buffer
  if (loop_candidates.empty()) {
    consistent_groups.clear();
    if (new_kf_fcid.cam_id == 0) {
      // recognition_database->insert(new_kf_fcid, new_kf.bow_vector);
      insert_new_kf_to_db(new_kf_fcid, new_kf, recognition_database);
    }
    return false;
  }
  enough_consistent_candidates.clear();
  ConsistentGroups current_consistent_groups;
  std::vector<bool> is_old_groups_consistent(consistent_groups.size(), false);

  // We iterate through the current candidate keyframes
  // and we also get the set of all of its covisible keyframes
  for (int i = 0; i < loop_candidates.size(); i++) {
    FrameCamId candidate_fcid = loop_candidates[i];
    std::set<FrameCamId> candidate_group = graph.at(candidate_fcid);
    candidate_group.insert(candidate_fcid);
    // We iterate through the current consistent groups to see if we can
    // find
    // any repeated keyframes shown in the group
    bool enough_consistent = false;
    bool consistent_in_some_groups = false;
    int idx = 0;
    for (auto& g : consistent_groups) {
      std::set<FrameCamId>& prev_group = g.first;
      bool is_consistent = false;
      for (std::set<FrameCamId>::iterator it = candidate_group.begin(),
                                          end = candidate_group.end();
           it != end; it++) {
        if (prev_group.count(*it)) {
          is_consistent = true;
          consistent_in_some_groups = true;
          break;
        }
      }

      if (is_consistent) {
        int num_prev_consistency = g.second;
        int num_curr_consistency = num_prev_consistency + 1;
        num_prev_consistency++;
        if (!is_old_groups_consistent[idx]) {
          ConsistentGroup cg =
              std::make_pair(candidate_group, num_curr_consistency);
          current_consistent_groups.push_back(cg);
          is_old_groups_consistent[idx] =
              true;  // this avoid to include the same group more than once
        }
        if (num_curr_consistency >= num_consistency_threshold &&
            !enough_consistent) {
          enough_consistent_candidates.push_back(candidate_fcid);
          enough_consistent =
              true;  // this avoid to insert the same candidate more than
          // once
        }
      }
      idx++;
    }
    if (!consistent_in_some_groups) {
      ConsistentGroup cg = std::make_pair(candidate_group, 0);
      current_consistent_groups.push_back(cg);
    }
  }
  // Update Covisibility Consistent Groups
  consistent_groups = current_consistent_groups;

  if (new_kf_fcid.cam_id == 0) {
    // recognition_database->insert(new_kf_fcid, new_kf.bow_vector);
    insert_new_kf_to_db(new_kf_fcid, new_kf, recognition_database);
  }
  if (enough_consistent_candidates.empty()) {
    return false;
  } else {
    return true;
  }
  return false;
}

}  // namespace visnav
