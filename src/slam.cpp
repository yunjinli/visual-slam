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

#include <pangolin/display/image_view.h>
#include <pangolin/gl/gldraw.h>
#include <pangolin/image/image.h>
#include <pangolin/image/image_io.h>
#include <pangolin/image/typed_image.h>
#include <pangolin/pangolin.h>
#include <tbb/concurrent_unordered_map.h>
#include <visnav/calibration.h>
#include <visnav/common_types.h>
#include <visnav/gui_helper.h>
#include <visnav/keypoints.h>
#include <visnav/loop_closure_utils.h>
#include <visnav/map_utils.h>
#include <visnav/matching_utils.h>
#include <visnav/serialization.h>
#include <visnav/tracks.h>
#include <visnav/vo_utils.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <queue>
#include <sophus/se3.hpp>
#include <sstream>
#include <thread>

using namespace visnav;

///////////////////////////////////////////////////////////////////////////////
/// Declarations
///////////////////////////////////////////////////////////////////////////////

void draw_image_overlay(pangolin::View& v, size_t view_id);
void change_display_to_image(const FrameCamId& fcid);
void draw_scene();
void load_data(const std::string& path, const std::string& calib_path);
bool next_step();
void optimize();
void compute_projections();

///////////////////////////////////////////////////////////////////////////////
/// Constants
///////////////////////////////////////////////////////////////////////////////

constexpr int UI_WIDTH = 200;
constexpr int NUM_CAMS = 2;

///////////////////////////////////////////////////////////////////////////////
/// Variables
///////////////////////////////////////////////////////////////////////////////

int current_frame = 0;
Sophus::SE3d current_pose;
bool take_keyframe = true;
TrackId next_landmark_id = 0;

std::atomic<bool> opt_running{false};
std::atomic<bool> opt_finished{false};

std::set<FrameId> kf_frames;

std::shared_ptr<std::thread> opt_thread;

/// intrinsic calibration
Calibration calib_cam;
Calibration calib_cam_opt;

/// loaded images
tbb::concurrent_unordered_map<FrameCamId, std::string> images;

/// timestamps for all stereo pairs
std::vector<Timestamp> timestamps;

/// detected feature locations and descriptors
Corners feature_corners;

/// pairwise feature matches
Matches feature_matches;

/// camera poses in the current map
Cameras cameras;

/// camera poses that's been discarded
// Cameras deactive_cameras;

/// copy of cameras for optimization in parallel thread
Cameras cameras_opt;

/// landmark positions and feature observations in current map
Landmarks landmarks;

/// copy of landmarks for optimization in parallel thread
Landmarks landmarks_opt;

/// landmark positions that were removed from the current map
// Landmarks old_landmarks;

/// Record all history tracks
// FeatureTracks feature_tracks;

/// cashed info on reprojected landmarks; recomputed every time time from
/// cameras, landmarks, and feature_tracks; used for visualization and
/// determining outliers; indexed by images
ImageProjections image_projections;

std::vector<std::pair<FrameCamId, FrameCamId>> covisible_pair;
///////////////////////////////////////////////////////////////////////////////
/// GUI parameters
///////////////////////////////////////////////////////////////////////////////

// The following GUI elements can be enabled / disabled from the main panel
// by switching the prefix from "ui" to "hidden" or vice verca. This way you
// can show only the elements you need / want for development.

pangolin::Var<bool> ui_show_hidden("ui.show_extra_options", false, true);

//////////////////////////////////////////////
/// Image display options

pangolin::Var<int> show_frame1("ui.show_frame1", 0, 0, 1500);
pangolin::Var<int> show_cam1("ui.show_cam1", 0, 0, NUM_CAMS - 1);
pangolin::Var<int> show_frame2("ui.show_frame2", 0, 0, 1500);
pangolin::Var<int> show_cam2("ui.show_cam2", 1, 0, NUM_CAMS - 1);
pangolin::Var<bool> lock_frames("ui.lock_frames", true, true);
pangolin::Var<bool> show_detected("ui.show_detected", true, true);
pangolin::Var<bool> show_matches("ui.show_matches", true, true);
pangolin::Var<bool> show_inliers("ui.show_inliers", true, true);
pangolin::Var<bool> show_reprojections("ui.show_reprojections", true, true);
pangolin::Var<bool> show_outlier_observations("ui.show_outlier_obs", false,
                                              true);
pangolin::Var<bool> show_ids("ui.show_ids", false, true);
pangolin::Var<bool> show_epipolar("hidden.show_epipolar", false, true);
pangolin::Var<bool> show_cameras3d("hidden.show_cameras", true, true);
pangolin::Var<bool> show_points3d("hidden.show_points", true, true);
pangolin::Var<bool> show_old_points3d("hidden.show_old_points3d", true, true);
pangolin::Var<bool> show_gt("hidden.show_gt", false, true);
pangolin::Var<bool> show_vio_pt("hidden.show_vio_pt", false, true);

//////////////////////////////////////////////
/// Feature extraction and matching options

pangolin::Var<int> num_features_per_image("hidden.num_features", 1500, 10,
                                          5000);
pangolin::Var<bool> rotate_features("hidden.rotate_features", true, true);
pangolin::Var<int> feature_match_max_dist("hidden.match_max_dist", 70, 1, 255);
pangolin::Var<double> feature_match_test_next_best("hidden.match_next_best",
                                                   1.2, 1, 4);

pangolin::Var<double> match_max_dist_2d("hidden.match_max_dist_2d", 20.0, 1.0,
                                        50);

pangolin::Var<int> new_kf_min_inliers("hidden.new_kf_min_inliers", 80, 1, 200);

pangolin::Var<int> max_num_kfs("hidden.max_num_kfs", 10, 5, 20);

pangolin::Var<double> cam_z_threshold("hidden.cam_z_threshold", 0.1, 1.0, 0.0);

pangolin::Var<bool> show_all_keyframes("hidden.show_all_keyframes", true, true);
// pangolin::Var<double> new_kf_observed_features_threshold(
//     "hidden.new_kf_observed_features_threshold", 0.75, 0.1, 0.95);
// pangolin::Var<double> covisible_threshold("hidden.covisible_threshold", 0.1,
//                                           0.0, 1.0);
pangolin::Var<int> num_cov_threshold("hidden.num_cov_threshold", 15, 1, 100);

//////////////////////////////////////////////
/// Adding cameras and landmarks options

pangolin::Var<double> reprojection_error_pnp_inlier_threshold_pixel(
    "hidden.pnp_inlier_thresh", 3.0, 0.1, 10);

//////////////////////////////////////////////
/// Bundle Adjustment Options

pangolin::Var<bool> ba_optimize_intrinsics("hidden.ba_opt_intrinsics", false,
                                           true);
pangolin::Var<int> ba_verbose("hidden.ba_verbose", 1, 0, 2);

pangolin::Var<double> reprojection_error_huber_pixel("hidden.ba_huber_width",
                                                     1.0, 0.1, 10);

///////////////////////////////////////////////////////////////////////////////
/// GUI buttons
///////////////////////////////////////////////////////////////////////////////

// if you enable this, next_step is called repeatedly until completion
pangolin::Var<bool> continue_next("ui.continue_next", false, true);

using Button = pangolin::Var<std::function<void(void)>>;

Button next_step_btn("ui.next_step", &next_step);

///////////////////////////////////////////////////////////////////////////////
/// GUI and Boilerplate Implementation
///////////////////////////////////////////////////////////////////////////////

// Parse parameters, load data, and create GUI window and event loop (or
// process everything in non-gui mode).
int main(int argc, char** argv) {
  bool show_gui = true;
  std::string dataset_path = "../data/V1_01_easy/mav0/";
  std::string cam_calib = "../opt_calib.json";
  CLI::App app{"Visual odometry."};

  app.add_option("--show-gui", show_gui, "Show GUI");
  app.add_option("--dataset-path", dataset_path,
                 "Dataset path. Default: " + dataset_path);
  app.add_option("--cam-calib", cam_calib,
                 "Path to camera calibration. Default: " + cam_calib);
  try {
    app.parse(argc, argv);
  } catch (const CLI::ParseError& e) {
    return app.exit(e);
  }

  load_data(dataset_path, cam_calib);

  if (show_gui) {
    pangolin::CreateWindowAndBind("Main", 1800, 1000);

    glEnable(GL_DEPTH_TEST);

    // main parent display for images and 3d viewer
    pangolin::View& main_view =
        pangolin::Display("main")
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH), 1.0)
            .SetLayout(pangolin::LayoutEqualVertical);

    // parent display for images
    pangolin::View& img_view_display =
        pangolin::Display("images").SetLayout(pangolin::LayoutEqual);
    main_view.AddDisplay(img_view_display);

    // main ui panel
    pangolin::CreatePanel("ui").SetBounds(0.0, 1.0, 0.0,
                                          pangolin::Attach::Pix(UI_WIDTH));

    // extra options panel
    pangolin::View& hidden_panel = pangolin::CreatePanel("hidden").SetBounds(
        0.0, 1.0, pangolin::Attach::Pix(UI_WIDTH),
        pangolin::Attach::Pix(2 * UI_WIDTH));
    ui_show_hidden.Meta().gui_changed = true;

    // 2D image views
    std::vector<std::shared_ptr<pangolin::ImageView>> img_view;
    while (img_view.size() < NUM_CAMS) {
      std::shared_ptr<pangolin::ImageView> iv(new pangolin::ImageView);

      size_t idx = img_view.size();
      img_view.push_back(iv);

      img_view_display.AddDisplay(*iv);
      iv->extern_draw_function =
          std::bind(&draw_image_overlay, std::placeholders::_1, idx);
    }

    // 3D visualization (initial camera view optimized to see full map)
    pangolin::OpenGlRenderState camera(
        pangolin::ProjectionMatrix(640, 480, 400, 400, 320, 240, 0.001, 10000),
        pangolin::ModelViewLookAt(-3.4, -3.7, -8.3, 2.1, 0.6, 0.2,
                                  pangolin::AxisNegY));

    pangolin::View& display3D =
        pangolin::Display("scene")
            .SetAspect(-640 / 480.0)
            .SetHandler(new pangolin::Handler3D(camera));
    main_view.AddDisplay(display3D);

    while (!pangolin::ShouldQuit()) {
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      if (ui_show_hidden.GuiChanged()) {
        hidden_panel.Show(ui_show_hidden);
        const int panel_width = ui_show_hidden ? 2 * UI_WIDTH : UI_WIDTH;
        main_view.SetBounds(0.0, 1.0, pangolin::Attach::Pix(panel_width), 1.0);
      }

      display3D.Activate(camera);
      glClearColor(0.95f, 0.95f, 0.95f, 1.0f);  // light gray background

      draw_scene();

      img_view_display.Activate();

      if (lock_frames) {
        // in case of locking frames, chaning one should change the other
        if (show_frame1.GuiChanged()) {
          change_display_to_image(FrameCamId(show_frame1, 0));
          change_display_to_image(FrameCamId(show_frame1, 1));
        } else if (show_frame2.GuiChanged()) {
          change_display_to_image(FrameCamId(show_frame2, 0));
          change_display_to_image(FrameCamId(show_frame2, 1));
        }
      }

      if (show_frame1.GuiChanged() || show_cam1.GuiChanged()) {
        auto frame_id = static_cast<FrameId>(show_frame1);
        auto cam_id = static_cast<CamId>(show_cam1);

        FrameCamId fcid;
        fcid.frame_id = frame_id;
        fcid.cam_id = cam_id;
        if (images.find(fcid) != images.end()) {
          pangolin::TypedImage img = pangolin::LoadImage(images[fcid]);
          img_view[0]->SetImage(img);
        } else {
          img_view[0]->Clear();
        }
      }

      if (show_frame2.GuiChanged() || show_cam2.GuiChanged()) {
        auto frame_id = static_cast<FrameId>(show_frame2);
        auto cam_id = static_cast<CamId>(show_cam2);

        FrameCamId fcid;
        fcid.frame_id = frame_id;
        fcid.cam_id = cam_id;
        if (images.find(fcid) != images.end()) {
          pangolin::GlPixFormat fmt;
          fmt.glformat = GL_LUMINANCE;
          fmt.gltype = GL_UNSIGNED_BYTE;
          fmt.scalable_internal_format = GL_LUMINANCE8;

          pangolin::TypedImage img = pangolin::LoadImage(images[fcid]);
          img_view[1]->SetImage(img);
        } else {
          img_view[1]->Clear();
        }
      }

      pangolin::FinishFrame();

      if (continue_next) {
        // stop if there is nothing left to do
        continue_next = next_step();
      } else {
        // if the gui is just idling, make sure we don't burn too much CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    }
  } else {
    // non-gui mode: Process all frames, then exit
    while (next_step()) {
      // nop
    }
    return 0;
  }
}

// Visualize features and related info on top of the image views
void draw_image_overlay(pangolin::View& v, size_t view_id) {
  UNUSED(v);

  auto frame_id =
      static_cast<FrameId>(view_id == 0 ? show_frame1 : show_frame2);
  auto cam_id = static_cast<CamId>(view_id == 0 ? show_cam1 : show_cam2);

  FrameCamId fcid(frame_id, cam_id);

  float text_row = 20;

  if (show_detected) {
    glLineWidth(1.0);
    glColor3f(1.0, 0.0, 0.0);  // red
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (feature_corners.find(fcid) != feature_corners.end()) {
      const KeypointsData& cr = feature_corners.at(fcid);

      for (size_t i = 0; i < cr.corners.size(); i++) {
        Eigen::Vector2d c = cr.corners[i];
        double angle = cr.corner_angles[i];
        pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

        Eigen::Vector2d r(3, 0);
        Eigen::Rotation2Dd rot(angle);
        r = rot * r;

        pangolin::glDrawLine(c, c + r);
      }

      pangolin::GlFont::I()
          .Text("Detected %d corners", cr.corners.size())
          .Draw(5, text_row);

    } else {
      glLineWidth(1.0);

      pangolin::GlFont::I().Text("Corners not processed").Draw(5, text_row);
    }
    text_row += 20;
  }

  if (show_matches || show_inliers) {
    glLineWidth(1.0);
    glColor3f(0.0, 0.0, 1.0);  // blue
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto o_frame_id =
        static_cast<FrameId>(view_id == 0 ? show_frame2 : show_frame1);
    auto o_cam_id = static_cast<CamId>(view_id == 0 ? show_cam2 : show_cam1);

    FrameCamId o_fcid(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(fcid, o_fcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_fcid, fcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && show_matches) {
      if (feature_corners.find(fcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(fcid);

        for (size_t i = 0; i < it->second.matches.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.matches[i].first
                                  : it->second.matches[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d matches", it->second.matches.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }

    glColor3f(0.0, 1.0, 0.0);  // green

    if (idx >= 0 && show_inliers) {
      if (feature_corners.find(fcid) != feature_corners.end()) {
        const KeypointsData& cr = feature_corners.at(fcid);

        for (size_t i = 0; i < it->second.inliers.size(); i++) {
          size_t c_idx = idx == 0 ? it->second.inliers[i].first
                                  : it->second.inliers[i].second;

          Eigen::Vector2d c = cr.corners[c_idx];
          double angle = cr.corner_angles[c_idx];
          pangolin::glDrawCirclePerimeter(c[0], c[1], 3.0);

          Eigen::Vector2d r(3, 0);
          Eigen::Rotation2Dd rot(angle);
          r = rot * r;

          pangolin::glDrawLine(c, c + r);

          if (show_ids) {
            pangolin::GlFont::I().Text("%d", i).Draw(c[0], c[1]);
          }
        }

        pangolin::GlFont::I()
            .Text("Detected %d inliers", it->second.inliers.size())
            .Draw(5, text_row);
        text_row += 20;
      }
    }
  }

  if (show_reprojections) {
    if (image_projections.count(fcid) > 0) {
      glLineWidth(1.0);
      glColor3f(1.0, 0.0, 0.0);  // red
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      const size_t num_points = image_projections.at(fcid).obs.size();
      double error_sum = 0;
      size_t num_outliers = 0;

      // count up and draw all inlier projections
      for (const auto& lm_proj : image_projections.at(fcid).obs) {
        error_sum += lm_proj->reprojection_error;

        if (lm_proj->outlier_flags != OutlierNone) {
          // outlier point
          glColor3f(1.0, 0.0, 0.0);  // red
          ++num_outliers;
        } else if (lm_proj->reprojection_error >
                   reprojection_error_huber_pixel) {
          // close to outlier point
          glColor3f(1.0, 0.5, 0.0);  // orange
        } else {
          // clear inlier point
          glColor3f(1.0, 1.0, 0.0);  // yellow
        }
        pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
        pangolin::glDrawLine(lm_proj->point_measured,
                             lm_proj->point_reprojected);
      }

      // only draw outlier projections
      if (show_outlier_observations) {
        glColor3f(1.0, 0.0, 0.0);  // red
        for (const auto& lm_proj : image_projections.at(fcid).outlier_obs) {
          pangolin::glDrawCirclePerimeter(lm_proj->point_reprojected, 3.0);
          pangolin::glDrawLine(lm_proj->point_measured,
                               lm_proj->point_reprojected);
        }
      }

      glColor3f(1.0, 0.0, 0.0);  // red
      pangolin::GlFont::I()
          .Text("Average repr. error (%u points, %u new outliers): %.2f",
                num_points, num_outliers, error_sum / num_points)
          .Draw(5, text_row);
      text_row += 20;
    }
  }

  if (show_epipolar) {
    glLineWidth(1.0);
    glColor3f(0.0, 1.0, 1.0);  // bright teal
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    auto o_frame_id =
        static_cast<FrameId>(view_id == 0 ? show_frame2 : show_frame1);
    auto o_cam_id = static_cast<CamId>(view_id == 0 ? show_cam2 : show_cam1);

    FrameCamId o_fcid(o_frame_id, o_cam_id);

    int idx = -1;

    auto it = feature_matches.find(std::make_pair(fcid, o_fcid));

    if (it != feature_matches.end()) {
      idx = 0;
    } else {
      it = feature_matches.find(std::make_pair(o_fcid, fcid));
      if (it != feature_matches.end()) {
        idx = 1;
      }
    }

    if (idx >= 0 && it->second.inliers.size() > 20) {
      Sophus::SE3d T_this_other =
          idx == 0 ? it->second.T_i_j : it->second.T_i_j.inverse();

      Eigen::Vector3d p0 = T_this_other.translation().normalized();

      int line_id = 0;
      for (double i = -M_PI_2 / 2; i <= M_PI_2 / 2; i += 0.05) {
        Eigen::Vector3d p1(0, sin(i), cos(i));

        if (idx == 0) p1 = it->second.T_i_j * p1;

        p1.normalize();

        std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
            line;
        for (double j = -1; j <= 1; j += 0.001) {
          line.emplace_back(calib_cam.intrinsics[cam_id]->project(
              p0 * j + (1 - std::abs(j)) * p1));
        }

        Eigen::Vector2d c = calib_cam.intrinsics[cam_id]->project(p1);
        pangolin::GlFont::I().Text("%d", line_id).Draw(c[0], c[1]);
        line_id++;

        pangolin::glDrawLineStrip(line);
      }
    }
  }
}

// Update the image views to a given image id
void change_display_to_image(const FrameCamId& fcid) {
  if (0 == fcid.cam_id) {
    // left view
    show_cam1 = 0;
    show_frame1 = fcid.frame_id;
    show_cam1.Meta().gui_changed = true;
    show_frame1.Meta().gui_changed = true;
  } else {
    // right view
    show_cam2 = fcid.cam_id;
    show_frame2 = fcid.frame_id;
    show_cam2.Meta().gui_changed = true;
    show_frame2.Meta().gui_changed = true;
  }
}

// Render the 3D viewer scene of cameras and points
void draw_scene() {
  const FrameCamId fcid1(show_frame1, show_cam1);
  const FrameCamId fcid2(show_frame2, show_cam2);

  const u_int8_t color_camera_current[3]{255, 0, 0};         // red
  const u_int8_t color_camera_left[3]{0, 125, 0};            // dark green
  const u_int8_t color_camera_right[3]{0, 0, 125};           // dark blue
  const u_int8_t color_points[3]{0, 0, 0};                   // black
  const u_int8_t color_old_points[3]{170, 170, 170};         // gray
  const u_int8_t color_selected_left[3]{0, 250, 0};          // green
  const u_int8_t color_selected_right[3]{0, 0, 250};         // blue
  const u_int8_t color_selected_both[3]{0, 250, 250};        // teal
  const u_int8_t color_outlier_observation[3]{250, 0, 250};  // purple

  // render cameras
  if (show_cameras3d) {
    for (const auto& cam : cameras) {
      if (cam.second.active) {
        if (cam.first == fcid1) {
          render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_left,
                        0.1f);
        } else if (cam.first == fcid2) {
          render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_right,
                        0.1f);
        } else if (cam.first.cam_id == 0) {
          render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_left,
                        0.1f);
        } else {
          render_camera(cam.second.T_w_c.matrix(), 2.0f, color_camera_right,
                        0.1f);
        }
      }
    }
    render_camera(current_pose.matrix(), 2.0f, color_camera_current, 0.1f);
  }

  if (show_all_keyframes) {
    for (const auto& cam : cameras) {
      render_camera(cam.second.T_w_c.matrix(), 3.0f, color_selected_left, 0.1f);
    }
    for (const auto& pair : covisible_pair) {
      // Draw the line
      // Convert the Eigen::Vector3d points to GLfloat arrays
      GLfloat start_gl[3];
      start_gl[0] =
          static_cast<GLfloat>(cameras[pair.first].T_w_c.translation()[0]);
      start_gl[1] =
          static_cast<GLfloat>(cameras[pair.first].T_w_c.translation()[1]);
      start_gl[2] =
          static_cast<GLfloat>(cameras[pair.first].T_w_c.translation()[2]);
      GLfloat end_gl[3];
      end_gl[0] =
          static_cast<GLfloat>(cameras[pair.second].T_w_c.translation()[0]);
      end_gl[1] =
          static_cast<GLfloat>(cameras[pair.second].T_w_c.translation()[1]);
      end_gl[2] =
          static_cast<GLfloat>(cameras[pair.second].T_w_c.translation()[2]);

      glBegin(GL_LINES);
      glVertex3fv(start_gl);
      glVertex3fv(end_gl);
      glEnd();
    }
  }

  // render points
  if (show_points3d && landmarks.size() > 0) {
    glPointSize(3.0);
    glBegin(GL_POINTS);
    for (const auto& kv_lm : landmarks) {
      const bool in_cam_1 = kv_lm.second.obs.count(fcid1) > 0;
      const bool in_cam_2 = kv_lm.second.obs.count(fcid2) > 0;

      const bool outlier_in_cam_1 = kv_lm.second.outlier_obs.count(fcid1) > 0;
      const bool outlier_in_cam_2 = kv_lm.second.outlier_obs.count(fcid2) > 0;

      if (in_cam_1 && in_cam_2) {
        glColor3ubv(color_selected_both);
      } else if (in_cam_1) {
        glColor3ubv(color_selected_left);
      } else if (in_cam_2) {
        glColor3ubv(color_selected_right);
      } else if (outlier_in_cam_1 || outlier_in_cam_2) {
        glColor3ubv(color_outlier_observation);
      } else {
        if (kv_lm.second.active)
          glColor3ubv(color_points);
        else {
          if (show_old_points3d) {
            glColor3ubv(color_old_points);
          } else {
            // nop
          }
        }
      }

      pangolin::glVertex(kv_lm.second.p);
    }
    glEnd();
  }

  // render points
  // if (show_old_points3d && old_landmarks.size() > 0) {
  //   glPointSize(3.0);
  //   glBegin(GL_POINTS);

  //   for (const auto& kv_lm : old_landmarks) {
  //     glColor3ubv(color_old_points);
  //     pangolin::glVertex(kv_lm.second.p);
  //   }
  //   glEnd();
  // }
}
// Load images, calibration, and features / matches if available
void load_data(const std::string& dataset_path, const std::string& calib_path) {
  const std::string timestams_path = dataset_path + "/cam0/data.csv";

  {
    std::ifstream times(timestams_path);

    int id = 0;

    while (times) {
      std::string line;
      std::getline(times, line);

      //      if (line.size() < 20 || line[0] == '#' || id > 2700) continue;
      if (line.size() < 20 || line[0] == '#') continue;
      {
        std::string timestamp_str = line.substr(0, 19);
        std::istringstream ss(timestamp_str);
        Timestamp timestamp;
        ss >> timestamp;
        timestamps.push_back(timestamp);
      }

      std::string img_name = line.substr(20, line.size() - 21);

      for (int i = 0; i < NUM_CAMS; i++) {
        FrameCamId fcid(id, i);

        std::stringstream ss;
        ss << dataset_path << "/cam" << i << "/data/" << img_name;

        images[fcid] = ss.str();
      }

      id++;
    }

    std::cerr << "Loaded " << id << " image pairs" << std::endl;
  }

  {
    std::ifstream os(calib_path, std::ios::binary);

    if (os.is_open()) {
      cereal::JSONInputArchive archive(os);
      archive(calib_cam);
      std::cout << "Loaded camera from " << calib_path << " with models ";
      for (const auto& cam : calib_cam.intrinsics) {
        std::cout << cam->name() << " ";
      }
      std::cout << std::endl;
    } else {
      std::cerr << "could not load camera calibration " << calib_path
                << std::endl;
      std::abort();
    }
  }

  show_frame1.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame1.Meta().gui_changed = true;
  show_frame2.Meta().range[1] = images.size() / NUM_CAMS - 1;
  show_frame2.Meta().gui_changed = true;
}

///////////////////////////////////////////////////////////////////////////////
/// Here the algorithmically interesting implementation begins
///////////////////////////////////////////////////////////////////////////////

// Execute next step in the overall odometry pipeline. Call this repeatedly
// until it returns false for automatic execution.
bool next_step() {
  if (current_frame >= int(images.size()) / NUM_CAMS) return false;
  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];

  if (take_keyframe) {
    take_keyframe = false;

    FrameCamId fcidl(current_frame, 0), fcidr(current_frame, 1);

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;

    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    std::cout << "KF Projected " << projected_track_ids.size() << " points."
              << std::endl;

    MatchData md_stereo;
    KeypointsData kdl, kdr;

    pangolin::ManagedImage<uint8_t> imgl = pangolin::LoadImage(images[fcidl]);
    pangolin::ManagedImage<uint8_t> imgr = pangolin::LoadImage(images[fcidr]);

    detectKeypointsAndDescriptors(imgl, kdl, num_features_per_image,
                                  rotate_features);
    detectKeypointsAndDescriptors(imgr, kdr, num_features_per_image,
                                  rotate_features);

    md_stereo.T_i_j = T_0_1;

    Eigen::Matrix3d E;
    computeEssential(T_0_1, E);

    matchDescriptors(kdl.corner_descriptors, kdr.corner_descriptors,
                     md_stereo.matches, feature_match_max_dist,
                     feature_match_test_next_best);

    findInliersEssential(kdl, kdr, calib_cam.intrinsics[0],
                         calib_cam.intrinsics[1], E, 1e-3, md_stereo);

    std::cout << "KF Found " << md_stereo.inliers.size() << " stereo-matches."
              << std::endl;

    feature_corners[fcidl] = kdl;
    feature_corners[fcidr] = kdr;
    feature_matches[std::make_pair(fcidl, fcidr)] = md_stereo;

    LandmarkMatchData md;

    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           feature_match_max_dist, feature_match_test_next_best,
                           md);

    std::cout << "KF Found " << md.matches.size() << " matches." << std::endl;

    localize_camera(current_pose, calib_cam.intrinsics[0], kdl, landmarks,
                    reprojection_error_pnp_inlier_threshold_pixel, md);

    current_pose = md.T_w_c;

    add_new_landmarks(fcidl, fcidr, kdl, kdr, calib_cam, md_stereo, md,
                      landmarks, next_landmark_id);

    construct_visibility_graph(fcidl, cameras, landmarks, covisible_pair,
                               num_cov_threshold);
    construct_visibility_graph(fcidr, cameras, landmarks, covisible_pair,
                               num_cov_threshold);

    cameras[fcidl].T_w_c = current_pose;
    cameras[fcidl].active = true;
    cameras[fcidr].T_w_c = current_pose * T_0_1;
    cameras[fcidr].active = true;

    remove_old_keyframes(fcidl, max_num_kfs, cameras, landmarks, kf_frames);

    // Ducument the removed keyframe

    // if (removed) {
    //   // Construct the visibility graph
    //   FrameCamId removed_fcid(removed_fid, 0);
    //   construct_visibility_graph(removed_fcid, removed_camera,
    //   deactive_cameras,
    //                              feature_tracks, covisible_pair,
    //                              num_cov_threshold);
    //   deactive_cameras[removed_fcid] = removed_camera;
    // }

    // Ducument the removed keyframe

    optimize();

    current_pose = cameras[fcidl].T_w_c;

    // update image views
    change_display_to_image(fcidl);
    change_display_to_image(fcidr);

    compute_projections();

    current_frame++;
    return true;
  } else {
    FrameCamId fcidl(current_frame, 0), fcidr(current_frame, 1);

    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
        projected_points;
    std::vector<TrackId> projected_track_ids;

    project_landmarks(current_pose, calib_cam.intrinsics[0], landmarks,
                      cam_z_threshold, projected_points, projected_track_ids);

    std::cout << "Projected " << projected_track_ids.size() << " points."
              << std::endl;

    KeypointsData kdl;

    pangolin::ManagedImage<uint8_t> imgl = pangolin::LoadImage(images[fcidl]);

    detectKeypointsAndDescriptors(imgl, kdl, num_features_per_image,
                                  rotate_features);

    feature_corners[fcidl] = kdl;

    LandmarkMatchData md;
    find_matches_landmarks(kdl, landmarks, feature_corners, projected_points,
                           projected_track_ids, match_max_dist_2d,
                           feature_match_max_dist, feature_match_test_next_best,
                           md);

    std::cout << "Found " << md.matches.size() << " matches." << std::endl;

    localize_camera(current_pose, calib_cam.intrinsics[0], kdl, landmarks,
                    reprojection_error_pnp_inlier_threshold_pixel, md);

    current_pose = md.T_w_c;

    // Compute the number of newly observed features with the most recently
    // selected keyframe
    // const auto& recent_kf_fcid = cameras.rbegin()->first;
    // MatchData md_most_recent_kf;
    // matchDescriptors(kdl.corner_descriptors,
    //                  feature_corners.at(recent_kf_fcid).corner_descriptors,
    //                  md_most_recent_kf.matches, feature_match_max_dist,
    //                  feature_match_test_next_best);
    // double overlapping_fraction =
    //     (double)(md_most_recent_kf.matches.size()) /
    //     (double)(feature_corners.at(recent_kf_fcid).corner_descriptors.size());

    if (int(md.inliers.size()) < new_kf_min_inliers && !opt_running &&
        !opt_finished) {
      take_keyframe = true;
    }

    if (!opt_running && opt_finished) {
      opt_thread->join();
      for (const auto& kv : landmarks_opt) {
        landmarks.at(kv.first) = kv.second;
      }
      for (const auto& kv : cameras_opt) {
        cameras.at(kv.first) = kv.second;
      }
      // landmarks = landmarks_opt;
      // cameras = cameras_opt;
      calib_cam = calib_cam_opt;
      opt_finished = false;
    }

    // update image views
    change_display_to_image(fcidl);
    change_display_to_image(fcidr);

    current_frame++;
    return true;
  }
}

// Compute reprojections for all landmark observations for visualization and
// outlier removal.
void compute_projections() {
  image_projections.clear();

  for (const auto& kv_lm : landmarks) {
    const TrackId track_id = kv_lm.first;

    for (const auto& kv_obs : kv_lm.second.obs) {
      const FrameCamId& fcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(fcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(fcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(fcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[fcid].obs.push_back(proj_lm);
    }

    for (const auto& kv_obs : kv_lm.second.outlier_obs) {
      const FrameCamId& fcid = kv_obs.first;
      const Eigen::Vector2d p_2d_corner =
          feature_corners.at(fcid).corners[kv_obs.second];

      const Eigen::Vector3d p_c =
          cameras.at(fcid).T_w_c.inverse() * kv_lm.second.p;
      const Eigen::Vector2d p_2d_repoj =
          calib_cam.intrinsics.at(fcid.cam_id)->project(p_c);

      ProjectedLandmarkPtr proj_lm(new ProjectedLandmark);
      proj_lm->track_id = track_id;
      proj_lm->point_measured = p_2d_corner;
      proj_lm->point_reprojected = p_2d_repoj;
      proj_lm->point_3d_c = p_c;
      proj_lm->reprojection_error = (p_2d_corner - p_2d_repoj).norm();

      image_projections[fcid].outlier_obs.push_back(proj_lm);
    }
  }
}

// Optimize the active map with bundle adjustment
void optimize() {
  cameras_opt.clear();
  landmarks_opt.clear();
  size_t num_obs = 0;
  size_t num_lm = 0;
  for (const auto& kv : landmarks) {
    num_obs += kv.second.obs.size();
    if (kv.second.active) {
      num_lm += 1;
      TrackId tid = kv.first;
      Landmark lm = kv.second;
      landmarks_opt.emplace(std::make_pair(tid, lm));
    }
  }
  size_t num_cam = 0;
  for (const auto& kv : cameras) {
    if (kv.second.active) {
      num_cam += 1;
      FrameCamId fcid = kv.first;
      Camera cam = kv.second;
      cameras_opt.emplace(std::make_pair(fcid, cam));
    }
  }

  std::cerr << "Optimizing map with " << num_cam << " cameras, " << num_lm
            << " points and " << num_obs << " observations." << std::endl;

  // Fix oldest two cameras to fix SE3 and scale gauge. Making the whole
  // second camera constant is a bit suboptimal, since we only need 1 DoF, but
  // it's simple and the initial poses should be good from calibration.
  FrameId fid = *(kf_frames.begin());
  std::cout << "fid " << fid << std::endl;

  // Prepare bundle adjustment
  BundleAdjustmentOptions ba_options;
  ba_options.optimize_intrinsics = ba_optimize_intrinsics;
  ba_options.use_huber = true;
  ba_options.huber_parameter = reprojection_error_huber_pixel;
  ba_options.max_num_iterations = 20;
  ba_options.verbosity_level = ba_verbose;

  calib_cam_opt = calib_cam;
  // cameras_opt = cameras;
  // landmarks_opt = landmarks;

  opt_running = true;

  opt_thread.reset(new std::thread([fid, ba_options] {
    std::set<FrameCamId> fixed_cameras = {{fid, 0}, {fid, 1}};

    bundle_adjustment(feature_corners, ba_options, fixed_cameras, calib_cam_opt,
                      cameras_opt, landmarks_opt);

    opt_finished = true;
    opt_running = false;
  }));

  // Update project info cache
  compute_projections();
}
