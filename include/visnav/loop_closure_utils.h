#pragma once

#include <visnav/common_types.h>
#include <visnav/keypoints.h>

namespace visnav {

void construct_visibility_graph(
    const FrameCamId& new_fcid, const Cameras& cameras,
    const Landmarks& landmarks,
    std::vector<std::pair<FrameCamId, FrameCamId>>& covisible_pair,
    int threshold) {
  std::map<FrameCamId, int> share_lm_count;
  for (const auto& tid_lm : landmarks) {
    const TrackId& tid = tid_lm.first;
    const Landmark& lm = tid_lm.second;

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
    // if (lm.obs.count(new_fcid)) {
    //   for (const auto& ft : lm.obs) {
    //     const FrameCamId& fcid = ft.first;
    //     if (share_lm_count.count(fcid)) {
    //       share_lm_count.at(fcid) += 1;  // increment
    //     } else {
    //       share_lm_count.at(fcid) = 1;  // initailize
    //     }
    //   }
    // } else {
    //   // nop
    // }
  }

  // for (const auto& l : old_landmarks) {
  //   const TrackId& tid = l.first;
  //   const Landmark& lm = l.second;

  //   for (const auto& fcid_cam : cameras) {
  //     const FrameCamId& fcid = fcid_cam.first;
  //     if (lm.obs.count(new_fcid) && lm.obs.count(fcid)) {
  //       if (share_lm_count.count(fcid)) {
  //         share_lm_count.at(fcid) += 1;  // increment
  //       } else {
  //         share_lm_count.at(fcid) = 1;  // initailize
  //       }
  //     } else {
  //       // nop
  //     }
  //   }
  //   // if (lm.obs.count(new_fcid)) {
  //   //   for (const auto& ft : lm.obs) {
  //   //     const FrameCamId& fcid = ft.first;
  //   //     if (share_lm_count.count(fcid)) {
  //   //       share_lm_count.at(fcid) += 1;  // increment
  //   //     } else {
  //   //       share_lm_count.at(fcid) = 1;  // initailize
  //   //     }
  //   //   }
  //   // } else {
  //   //   // nop
  //   // }
  // }

  // iterate throught the share_lm_count and find fcid that is above the
  // threshold
  for (const auto& fcid_count : share_lm_count) {
    if (fcid_count.second >= threshold) {
      covisible_pair.push_back(std::make_pair(new_fcid, fcid_count.first));
    }
  }
}
}  // namespace visnav