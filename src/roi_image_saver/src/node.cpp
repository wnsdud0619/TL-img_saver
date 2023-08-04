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


#include "roi_image_saver/node.hpp"
#include <cv_bridge/cv_bridge.h>

#include <boost/filesystem.hpp>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#define WINDOW_NAME "DGIST_TL_SAVE"

namespace traffic_light
{
TrafficLightRoiImageSaver::TrafficLightRoiImageSaver()
: nh_(""),
  pnh_("~"),
  image_transport_(pnh_),
  image_sub_(image_transport_, "/sensing/camera/traffic_light/image_raw", 1000),
  roi_sub_(pnh_, "/traffic_light_recognition/rois", 1000),
  tl_states_sub_(pnh_, "/perception/camera/traffic_light_states", 1000),
  sync_(SyncPolicy(1000), image_sub_, roi_sub_, tl_states_sub_), name_cnt(0)
{
  sync_.registerCallback(boost::bind(&TrafficLightRoiImageSaver::imageRoiCallback, this, _1, _2, _3));

  //roslaunch의 파라미터 받아옴
  if(!nh_.getParam("/save_static_path", config_.save_static_path))
    ROS_ERROR("No parameter : save_static_path");
  
  if(!nh_.getParam("/area_threshold", config_.area_th))  
    ROS_ERROR("No parameter : area_threshold");

  if(!nh_.getParam("/confidence_threshold", config_.conf_th))   
    ROS_ERROR("No parameter : confidence_threshold");

  if(!nh_.getParam("/target_num", config_.target_num))  
    ROS_ERROR("No parameter : target_num");

  if(!nh_.getParam("/min_probability", config_.min_prob)) 
    ROS_ERROR("No parameter : min_probability");
  
  //get class num
  if(!nh_.getParam("/Green_num", cls_data[CLASS_ID::GREEN].num))   
    ROS_ERROR("No parameter : Green_num");

  if(!nh_.getParam("/Red_num", cls_data[CLASS_ID::RED].num))
    ROS_ERROR("No parameter : Red_num");

  if(!nh_.getParam("/Yellow_num", cls_data[CLASS_ID::YELLOW].num)) 
    ROS_ERROR("No parameter : Yellow_num");

  if(!nh_.getParam("/Green_Left_num", cls_data[CLASS_ID::GREEN_LEFT].num))   
    ROS_ERROR("No parameter : Green_Left_num");

  if(!nh_.getParam("/Red_Left_num", cls_data[CLASS_ID::RED_LEFT].num))  
    ROS_ERROR("No parameter : Red_Left_num");

  if(!nh_.getParam("/Unknown_num", cls_data[CLASS_ID::UNKNOWN].num))   
    ROS_ERROR("No parameter : Unknown_num");

  //최솟값 추출
  int min_num = INT_MAX;
  for(int idx=0; idx<CLASS_NUM; idx++) {
    if(cls_data[idx].num < min_num)
       min_num = cls_data[idx].num;
  }

  for(int idx=0; idx<CLASS_NUM; idx++) {
    //난수 생성관련 변수들 선언
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    cls_data[idx].generator = std::default_random_engine(seed);
    cls_data[idx].distribution = std::uniform_int_distribution<int>(1,100); //1~100까지 난수생성

    //확률 부여
    if(cls_data[idx].num >= config_.target_num) //지정된 타입의 숫자가 목표량이상 최소 확률 부여
      cls_data[idx].prob = config_.min_prob;
    else{//지정된 타입의 숫자가 목표량보다 작으면 차등적으로 확률 부여
      int prob = (double)(config_.target_num - cls_data[idx].num)/(config_.target_num - min_num) * 100;
      cls_data[idx].prob = prob;
    }
  }

  scene_log_full_path = init_scen_log(config_.save_static_path);
  roi_log_full_path = init_roi_log(config_.save_static_path);

  ROS_INFO("---------------------------------------------------------------");
  ROS_INFO("save_static_path: %s ", config_.save_static_path.c_str());
  ROS_INFO("target_num: %d, min_probability: %d", config_.target_num, config_.min_prob); 
  ROS_INFO("area_threshold: %d, confidence_threshold: %.2f ", config_.area_th, config_.conf_th);
  ROS_INFO("Class num: Green: %d, Red: %d, Yellow: %d, Green_Left: %d, Red_Left: %d, Unknown: %d", 
  cls_data[CLASS_ID::GREEN].num, cls_data[CLASS_ID::RED].num, cls_data[CLASS_ID::YELLOW].num,
  cls_data[CLASS_ID::GREEN_LEFT].num, cls_data[CLASS_ID::RED_LEFT].num, cls_data[CLASS_ID::UNKNOWN].num);
  ROS_INFO("Class probability: Green: %d%, Red: %d%, Yellow: %d%, Green_Left: %d%, Red_Left: %d%, Unknown: %d%", 
  cls_data[CLASS_ID::GREEN].prob, cls_data[CLASS_ID::RED].prob, cls_data[CLASS_ID::YELLOW].prob,
  cls_data[CLASS_ID::GREEN_LEFT].prob, cls_data[CLASS_ID::RED_LEFT].prob, cls_data[CLASS_ID::UNKNOWN].prob);
  ROS_INFO("---------------------------------------------------------------");
}
TrafficLightRoiImageSaver::~TrafficLightRoiImageSaver() {}
void TrafficLightRoiImageSaver::imageRoiCallback(
  const sensor_msgs::ImageConstPtr & input_image_msg,
  const autoware_perception_msgs::TrafficLightRoiArrayConstPtr & input_tl_roi_msg,
  const autoware_perception_msgs::TrafficLightStateArrayConstPtr & input_tl_states_msg)
{
  std::vector<roi_data> roi_new;
  scene_data scene = scene_data(); //데이터 생성시 filter값 false
  
  if(input_tl_states_msg->states.size() == 0) { // state 토픽 없을경우 데이터 초기화
    roi_old.clear();
    type_old.clear();
  }

  //현재 msg의 데이터 저장
  try {
    scene.img = cv_bridge::toCvCopy(input_image_msg, sensor_msgs::image_encodings::BGR8)->image;

    for(auto tl_roi : input_tl_roi_msg->rois) { //roi 수만큼 반복
      roi_data tmp = roi_data(); //기본값 초기화
      tmp.id = tl_roi.id;
      tmp.x = tl_roi.roi.x_offset;
      tmp.y = tl_roi.roi.y_offset;
      tmp.width = tl_roi.roi.width;
      tmp.height = tl_roi.roi.height;
      tmp.dist = tl_roi.distance;

      for(const auto & tl_state : input_tl_states_msg->states) { //state 정보저장
        if(tl_roi.id != tl_state.id) continue; // 해당 아이디인경우에만 처리
        for(int idx = 0; idx < tl_state.lamp_states.size(); idx++) {
          auto state = tl_state.lamp_states.at(idx);
          tmp.conf = state.confidence;
          tmp.type += AAP_type2label(state.type);
        if(idx < tl_state.lamp_states.size() - 1) tmp.type += "_";
        }
      }
      roi_new.push_back(tmp);
    }
  } catch (cv_bridge::Exception & e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", input_image_msg->encoding.c_str());
  }

  setType_anno(roi_new, scene);
  int target_roi = 0;
  target_roi = setType_balance(roi_new, scene, cls_data);
  
  filterLowConf(roi_new, scene, config_.conf_th);
  filterType(roi_new, type_old, scene);
  filterArea(roi_new, roi_old, scene, config_.area_th);
  filterBalance(roi_new, scene, cls_data , target_roi);
  /*
  std::cout<<"low_conf: "<<scene.filter[FILTER::LOW_CONF]<<std::endl;
  std::cout<<"low_conf_area: "<<scene.filter[FILTER::LOW_CONF_AREA]<<std::endl;
  std::cout<<"area: "<<scene.filter[FILTER::AREA]<<std::endl;
  std::cout<<"type: "<<scene.filter[FILTER::TYPE]<<std::endl;
  std::cout<<"balance: "<<scene.filter[FILTER::BALANCE]<<std::endl;*/
  
  if(scene.filter[FILTER::LOW_CONF_AREA] || scene.filter[FILTER::BALANCE]) {
    std::string str_png = scene.filter[FILTER::LOW_CONF_AREA] ? "/low_png" : "/png";
    std::string str_txt = scene.filter[FILTER::LOW_CONF_AREA] ? "/low_txt" : "/txt";

    save_TL_data(config_.save_static_path+str_png, config_.save_static_path+str_txt, roi_new, scene, name_cnt);
    roi_old.clear();
    roi_old.assign(roi_new.begin(), roi_new.end());
  }

  //roi log save
  save_roi_log(roi_log_full_path, roi_new, name_cnt);
  save_scen_log(scene_log_full_path, scene, name_cnt);

  if(input_tl_states_msg->states.size() != 0)
    name_cnt++;
   
  //std::cout<<"***********callback_End**************"<<std::endl;
}

int TrafficLightRoiImageSaver::TL_cls2idx(std::string cls) {
  int result;
  if(cls == "Green")
    result = CLASS_ID::GREEN;
  else if(cls == "Red")
    result = CLASS_ID::RED;
  else if(cls == "Yellow")
    result = CLASS_ID::YELLOW;
  else if(cls == "Green_Left")
    result = CLASS_ID::GREEN_LEFT;
  else if(cls == "Red_Left")
    result = CLASS_ID::RED_LEFT;
  else if(cls == "Unknown")
    result = CLASS_ID::UNKNOWN;
  else
    ROS_ERROR("Could not convert class");
  return result;
}

void TrafficLightRoiImageSaver::save_TL_data(std::string save_png_path, std::string save_txt_path, 
struct std::vector<roi_data> roi, struct scene_data scene, int cnt) {

  std::string png_name = cv::format("/%.10d.png", cnt);
  if (!boost::filesystem::is_directory(save_png_path))
    boost::filesystem::create_directories(save_png_path);
  
  std::string txt_name = cv::format("/%.10d.txt", cnt);
  if (!boost::filesystem::is_directory(save_txt_path))
    boost::filesystem::create_directories(save_txt_path);
  
  std::ofstream label_txt(save_txt_path + txt_name);
  //*************save param**************** 저장할 변수들 0으로 초기화(KITTI format)
  float truncated = 0.0, occluded = 0.0, alpha = 0.0, left = 0.0, top = 0.0, right = 0.0, bottom = 0.0, height = 0.0, width = 0.0, length = 0.0,
    x = 0.0, y = 0.0, z = 0.0, rotation_y = 0.0, score = 0.0;

  for(int idx = 0; idx<roi.size(); idx++) {
    top = roi[idx].y;
    left = roi[idx].x;
    right = roi[idx].x + roi[idx].width;
    bottom = roi[idx].y + roi[idx].height;
    
    label_txt << roi[idx].type <<" " 
              << truncated <<" "
              << occluded <<" "
              << alpha <<" "
              << left <<" "
              << top <<" "
              << right <<" "
              << bottom <<" "
              << height <<" "
              << width <<" "
              << length <<" "
              << x <<" "
              << y <<" "
              << scene.type_anno <<" "
              << roi[idx].conf <<std::endl;
  }
  cv::imwrite(save_png_path + png_name, scene.img); // 전체 이미지저장
  label_txt.close(); // label 저장  
}
std::string TrafficLightRoiImageSaver::AAP_type2label(int type) {
  std::string lable;
  if(type == 0)
    lable = "Unknown";
  else if(type == 1)
    lable = "Red";
  else if(type == 2)
    lable = "Green";
  else if(type == 3)
    lable = "Yellow";
  else if(type == 4)
    lable = "Left";
  else
    ROS_ERROR("Could not convert type to lable");
  return lable;
}
std::string TrafficLightRoiImageSaver::init_scen_log(std::string static_path) {
  //scene log save
  std::string log_path = "/log";
  std::string full_path = static_path + log_path + "/scene_log_data.txt";
  if(!boost::filesystem::is_directory(static_path + log_path))
    boost::filesystem::create_directories(static_path + log_path);
  std::ofstream scene_log_txt(full_path, std::ios_base::out | std::ios::trunc);

  scene_log_txt<<"Green_prob: "<<cls_data[CLASS_ID::GREEN].prob<<
              ", Red_prob: "<<cls_data[CLASS_ID::RED].prob<<
              ", Yellow_prob: "<<cls_data[CLASS_ID::YELLOW].prob<<
              ", Green_Left_prob: "<<cls_data[CLASS_ID::GREEN_LEFT].prob<<
              ", Red_Left_prob: "<<cls_data[CLASS_ID::RED_LEFT].prob<<
              ", Unknown_prob: "<<cls_data[CLASS_ID::UNKNOWN].prob<<std::endl;
  scene_log_txt<<"파일명"<<" "<<"대표타입"<<" "<<"(bool)low_conf"<<" "<<"(bool)low_conf_area"<<" "<<"(bool)area"<<" "<<"(bool)type"<<" "<<"(bool)balance"<<std::endl;
  scene_log_txt.close();

  return full_path;
}
std::string TrafficLightRoiImageSaver::init_roi_log(std::string static_path) {
  //roi log save
  std::string log_path = "/log";
  std::string full_path = config_.save_static_path + log_path + "/roi_log_data.txt";
  std::ofstream roi_log_txt(full_path, std::ios_base::out | std::ios::trunc);

  roi_log_txt<<"파일명"<<" "<<
            "id"<<" "<<
            "type"<<" "<<
            "area"<<" "<<
            "conf"<<" "<<
            "dist"<<" "<<
            "(bool)low_conf"<<" "<<
            "(bool)low_conf_area"<<" "<<
            "(bool)area"<<" "<<
            "(bool)type"<<" "<<
            "(bool)balance"<<std::endl;
  roi_log_txt.close();

  return full_path;                    
}
void TrafficLightRoiImageSaver::save_roi_log(std::string _full_path, std::vector<roi_data> _roi, int _cnt) {

  std::ofstream roi_log_txt(_full_path, std::ios_base::out | std::ios_base::app);
  for(int idx = 0; idx<_roi.size(); idx++) {
    int area = _roi[idx].width * _roi[idx].height;
    roi_log_txt<<_cnt<<" "<<
    _roi[idx].id<<" "<<
    _roi[idx].type<<" "<<
    area<<" "<<
    _roi[idx].conf<<" "<<
    _roi[idx].dist<<" "<<
    _roi[idx].filter[FILTER::LOW_CONF]<<" "<<
    _roi[idx].filter[FILTER::LOW_CONF_AREA]<<" "<<
    _roi[idx].filter[FILTER::AREA]<<" "<<
    _roi[idx].filter[FILTER::TYPE]<<" "<<
    _roi[idx].filter[FILTER::BALANCE]<<std::endl;
  }
  roi_log_txt.close();
}
void TrafficLightRoiImageSaver::save_scen_log(std::string _full_path, struct scene_data _scene, int _cnt) {
  
  if(_scene.type_anno == "") return;

  std::ofstream scene_log_txt(_full_path, std::ios_base::out | std::ios_base::app);
    scene_log_txt<<_cnt<<" "<<
    _scene.type_balance<<" "<<
    _scene.filter[FILTER::LOW_CONF]<<" "<<
    _scene.filter[FILTER::LOW_CONF_AREA]<<" "<<
    _scene.filter[FILTER::AREA]<<" "<<
    _scene.filter[FILTER::TYPE]<<" "<<
    _scene.filter[FILTER::BALANCE]<<std::endl;
    scene_log_txt.close();  
}  
void TrafficLightRoiImageSaver::setType_anno(std::vector<roi_data> &_roi_new, struct scene_data &_scene) {
  double max_conf = 0.0;
  for(int idx = 0; idx < _roi_new.size(); idx ++) {
    if(max_conf < _roi_new[idx].conf) {
      max_conf = _roi_new[idx].conf;
      _scene.type_anno = _roi_new[idx].type;
    }
  }
}
int TrafficLightRoiImageSaver::setType_balance(std::vector<roi_data> &_roi_new, struct scene_data &_scene, struct cls_data _cls_data[CLASS_NUM]) {
  int max_prob = 0;
  int target_roi = 0;
  for(int idx = 0; idx < _roi_new.size(); idx ++){
    if(max_prob < _cls_data[TL_cls2idx(_roi_new[idx].type)].prob) { //max porb이용 balance filter용 type설정
      max_prob = _cls_data[TL_cls2idx(_roi_new[idx].type)].prob;
      _scene.type_balance = _roi_new[idx].type;
      target_roi = idx;
    }
  }
  return target_roi;
}
void TrafficLightRoiImageSaver::filterLowConf(std::vector<roi_data> &_roi_new, struct scene_data &_scene, double _conf_th) {
  for(int idx = 0; idx < _roi_new.size(); idx++){
    if(_roi_new[idx].conf < _conf_th){
      _roi_new[idx].filter[FILTER::LOW_CONF] = true;
      _scene.filter[FILTER::LOW_CONF] = true;
      _scene.filter[FILTER::LOW_CONF_AREA] = true;     
    }
  }
}
void TrafficLightRoiImageSaver::filterType(std::vector<roi_data> &_roi_new, std::vector<roi_data> &_roi_old, struct scene_data &_scene) {
  for(int idx = 0; idx < _roi_new.size(); idx++) {
    for(int jdx = 0; jdx < _roi_old.size(); jdx++) {
      if(_roi_old[jdx].id != _roi_new[idx].id) continue;
      if(_roi_old[jdx].type != _roi_new[jdx].type) {
        _roi_new[idx].filter[FILTER::TYPE] = true;
        _scene.filter[FILTER::TYPE] = true;
      }    
    }
  }
  _roi_old.clear();
  _roi_old.assign(_roi_new.begin(), _roi_new.end());
}
void TrafficLightRoiImageSaver::filterArea(std::vector<roi_data> &_roi_new, std::vector<roi_data> &_roi_old, struct scene_data &_scene, int _area_th) {
  for(int idx = 0; idx < _roi_new.size(); idx ++) {
    for(int jdx = 0; jdx<_roi_old.size() ;jdx++) {
      if(_roi_old[jdx].id != _roi_new[idx].id) continue;
      int old_area = _roi_old[jdx].width * _roi_old[jdx].height; 
      int new_area = _roi_new[jdx].width * _roi_new[jdx].height;
      int area_diff = abs(new_area - old_area);

      if(area_diff < _area_th){
        _roi_new[idx].filter[FILTER::AREA] = false;
        _scene.filter[FILTER::AREA] = false;
        _roi_new[idx].filter[FILTER::LOW_CONF_AREA] = false;
        _scene.filter[FILTER::LOW_CONF_AREA] = false;
      }
    }
  }
}
void TrafficLightRoiImageSaver::filterBalance(std::vector<roi_data> &_roi_new, struct scene_data &_scene, struct cls_data _cls_data[CLASS_NUM] ,int _target_roi) {
  if(_scene.type_balance == "" || !(_scene.filter[FILTER::TYPE] || _scene.filter[FILTER::AREA])) return;
 
  int random = _cls_data[TL_cls2idx(_scene.type_balance)].distribution(_cls_data[TL_cls2idx(_scene.type_balance)].generator);
  if(cls_data[TL_cls2idx(_scene.type_balance)].prob >= random) {//balance filter
    _roi_new[_target_roi].filter[FILTER::BALANCE] = true;
    _scene.filter[FILTER::BALANCE] = true;
  }
 
  //std::cout<<"random: "<<random<<", "<<"prob: "<<_cls_data[TL_cls2idx(_scene.type_balance)].prob<<std::endl;
}  

}  // namespace traffic_light


