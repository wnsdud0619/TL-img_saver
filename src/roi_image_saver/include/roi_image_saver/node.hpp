/*
 * Copyright 2020 Tier IV, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include <ros/ros.h>

#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include "autoware_perception_msgs/TrafficLightRoiArray.h"
#include "autoware_perception_msgs/TrafficLightRoiArray.h"
#include "autoware_perception_msgs/TrafficLightStateArray.h"
#include "message_filters/subscriber.h"
#include "message_filters/sync_policies/approximate_time.h"
#include "message_filters/synchronizer.h"
#include "sensor_msgs/Image.h"

#include <opencv2/highgui/highgui.hpp>

#include <memory>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <chrono>
#include <random>
#include <map>

#define CLASS_NUM 6

enum FILTER {
    LOW_CONF,        // assigned 0
    LOW_CONF_AREA,   // assigned 1
    AREA,            // assigned 2
    TYPE,            // assigned 3
    BALANCE          // assigned 4
};

enum CLASS_ID {
    UNKNOWN,    // assigned 0
    GREEN,      // assigned 1
    RED,        // assigned 2
    YELLOW,     // assigned 3
    RED_LEFT,   // assigned 4
    GREEN_LEFT  // assigned 5
};

namespace traffic_light
{
class TrafficLightRoiImageSaver
{
public:
  TrafficLightRoiImageSaver();
  virtual ~TrafficLightRoiImageSaver();

public:
 
private:
  struct Config {
    int area_th;
    double conf_th;
    int target_num;
    int min_prob;
    std::string save_static_path;
  };

  struct cls_data {
    int num;
    int prob;
    std::uniform_int_distribution<int> distribution;
    std::default_random_engine generator;
  };

 struct roi_data {
    int id;
    std::string type;
    int x;
    int y;
    int width;
    int height;
    double conf;
    double dist;
    bool filter[5];
    roi_data() {id = 0; type = ""; x = 0; y = 0; width = 0; height = 0; conf = 0.0; dist = 0.0;
    filter[FILTER::LOW_CONF] = false;
    filter[FILTER::LOW_CONF_AREA] = true;
    filter[FILTER::AREA] = true;
    filter[FILTER::TYPE] = false;
    filter[FILTER::BALANCE] = false; }
  };

  struct scene_data {
    std::string type_anno;
    std::string type_balance;
    cv::Mat img;
    bool filter[5];
    scene_data() {
    type_anno = "";
    type_balance = "";
    filter[FILTER::LOW_CONF] = false;
    filter[FILTER::LOW_CONF_AREA] = false;
    filter[FILTER::AREA] = true;
    filter[FILTER::TYPE] = false;
    filter[FILTER::BALANCE] = false; } 
  };

private:
  Config config_;
  cls_data cls_data[CLASS_NUM];
  std::vector<roi_data> roi_old; //roi_msg, roi_state 데이터 관리
  std::vector<roi_data> type_old;
  int name_cnt;
  std::string scene_log_full_path;
  std::string roi_log_full_path;

  void imageRoiCallback(
    const sensor_msgs::ImageConstPtr & input_image_msg,
    const autoware_perception_msgs::TrafficLightRoiArray::ConstPtr & input_tl_roi_msg,
    const autoware_perception_msgs::TrafficLightStateArrayConstPtr & input_tl_states_msg);
  std::string AAP_type2label(int type);
  int TL_cls2idx(std::string cls);
  void save_TL_data(std::string save_png_path, std::string save_txt_path, 
      struct std::vector<roi_data> roi, struct scene_data scene, int cnt);
  std::string init_scen_log(std::string static_path);
  std::string init_roi_log(std::string static_path);
  void save_roi_log(std::string full_path, std::vector<roi_data> roi, int cnt);
  void save_scen_log(std::string full_path, struct scene_data scene, int cnt);
  void setType_anno(std::vector<roi_data> &_roi_new, struct scene_data &_scene);
  int setType_balance(std::vector<roi_data> &_roi_new, struct scene_data &_scene, struct cls_data _cls_data[CLASS_NUM]);
  void filterLowConf(std::vector<roi_data> &_roi_new, struct scene_data &_scene, double _conf_th);
  void filterArea(std::vector<roi_data> &_roi_new, std::vector<roi_data> &_roi_old, struct scene_data &_scene, int _area_th);
  void filterType(std::vector<roi_data> &_roi_new, std::vector<roi_data> &_roi_old, struct scene_data &_scene);
  void filterBalance(std::vector<roi_data> &_roi_new,struct scene_data &_scene, struct cls_data _cls_data[CLASS_NUM] ,int _target_roi);    

  ros::NodeHandle nh_, pnh_;
  image_transport::ImageTransport image_transport_;
  image_transport::SubscriberFilter image_sub_;
  message_filters::Subscriber<autoware_perception_msgs::TrafficLightRoiArray> roi_sub_;
  message_filters::Subscriber<autoware_perception_msgs::TrafficLightStateArray> tl_states_sub_;
  image_transport::Publisher image_pub_;
  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, autoware_perception_msgs::TrafficLightRoiArray,
    autoware_perception_msgs::TrafficLightStateArray>
    SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;
  Sync sync_;
};

}  // namespace traffic_light