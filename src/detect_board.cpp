/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
 */
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>

#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <cv_bridge/cv_bridge.h>

#include <ros/ros.h>

#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/aruco/charuco.hpp>

#include <boost/thread.hpp>

#include <vector>
#include <iostream>

using namespace std;
using namespace cv;

//rosrun baxter_4_powder detect_board `rospack find baxter_4_powder`/config/detector_params.yml -w=5 -h=7 -sl=0.035 -ml=0.0228 -d=10
//rosrun baxter_4_powder detect_board `rospack find baxter_4_powder`/config/detector_params.yml -w=2 -h=3 -sl=0.079 -ml=0.059 -d=11
//rosrun baxter_4_powder detect_board `rospack find baxter_4_powder`/config/detector_params.yml -w=2 -h=3 -sl=0.079 -ml=0.059 -d=15
//rosrun baxter_4_powder detect_board `rospack find baxter_4_powder`/config/detector_params.yml -w=2 -h=3 -sl=0.079 -ml=0.059 -d=16


namespace {
const char* about = "Pose estimation using a ChArUco board";
const char* keys  =
    "{w        |       | Number of squares in X direction }"
    "{h        |       | Number of squares in Y direction }"
    "{sl       |       | Square side length (in meters) }"
    "{ml       |       | Marker side length (in meters) }"
    "{d        |       | dictionary: DICT_4X4_50=0, DICT_4X4_100=1, DICT_4X4_250=2,"
    "DICT_4X4_1000=3, DICT_5X5_50=4, DICT_5X5_100=5, DICT_5X5_250=6, DICT_5X5_1000=7, "
    "DICT_6X6_50=8, DICT_6X6_100=9, DICT_6X6_250=10, DICT_6X6_1000=11, DICT_7X7_50=12,"
    "DICT_7X7_100=13, DICT_7X7_250=14, DICT_7X7_1000=15, DICT_ARUCO_ORIGINAL = 16}"
    "{dp       |       | File of marker detector parameters }"
    "{rs       |       | Apply refind strategy }"
    "{r        |       | show rejected candidates too }";
}

class boardDetector
{
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::CameraInfo, sensor_msgs::Image> SyncPolice;

public:
  boardDetector(const CommandLineParser& parser, const std::string& cam_name, const std::string& cam_info_topic, const std::string& image_topic) :
    cam_name_(cam_name)
  {
    squaresX = parser.get<int>("w");
    squaresY = parser.get<int>("h");
    squareLength = parser.get<float>("sl");
    markerLength = parser.get<float>("ml");
    dictionaryId = parser.get<int>("d");
    showRejected = parser.has("r");
    refindStrategy = parser.has("rs");


    detectorParams = aruco::DetectorParameters::create();
    if(parser.has("dp"))
    {
      bool readOk = readDetectorParameters(parser.get<string>("dp"), detectorParams);
      if(!readOk)
      {
        cerr << "Invalid detector parameters file" << endl;
        assert(!readOk);
      }
    }

    if(!parser.check())
    {
      parser.printErrors();
      assert(!parser.check());
    }

    dictionary = aruco::getPredefinedDictionary(aruco::PREDEFINED_DICTIONARY_NAME(dictionaryId));

    axisLength = 0.5f * ((float)min(squaresX, squaresY) * (squareLength));

    // create charuco board object
    charucoboard = aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
    board = charucoboard.staticCast<aruco::Board>();

//    totalTime = 0;
//    totalIterations = 0;

    namedWindow(cam_name_, CV_WINDOW_AUTOSIZE);

    nh_.reset(new ros::NodeHandle);

    cam_info_sub_.reset(new message_filters::Subscriber<sensor_msgs::CameraInfo>(*nh_, cam_info_topic, 10)) ;
    image_sub_.reset(new message_filters::Subscriber<sensor_msgs::Image>(*nh_, image_topic, 10)) ;
    sync_.reset(new message_filters::Synchronizer<SyncPolice> (SyncPolice(10), *cam_info_sub_, *image_sub_));
  }

  virtual ~boardDetector () { destroyWindow(cam_name_); }

  void
  start()
  {
    sync_->registerCallback(boost::bind(&boardDetector::readCameraParametersCb, this, _1, _2));
  }

  bool
  getBoardPose(Eigen::Affine3f& T, boost::unique_lock<boost::mutex>& lock)
  {
    bool has_data = data_ready_cond_.timed_wait (lock, boost::posix_time::millisec(100)); // has_bd_rs = bd_realsense_->getBoardPose(rs_T);

    if (has_data)
    {
      for (int i = 0 ; i < 4; i ++)
        for (int j = 0; j < 4; j ++)
          T(i,j) = T_(i,j);
    }
    return has_data;
  }

  boost::mutex data_ready_mtx_;
  boost::condition_variable data_ready_cond_ ;

private:

  void readCameraParametersCb (const sensor_msgs::CameraInfoConstPtr& cam_info_msg, const sensor_msgs::ImageConstPtr& image_msg)
  {
    // Read camera Parameters
    Mat camMatrix, distCoeffs;
    camMatrix = Mat(3,3, CV_32F);
    for (int i = 0; i < 3; i++) // x
      for (int j = 0; j < 3; j++ ) // y
        camMatrix.at<float>(Point(i,j)) = cam_info_msg->K[j*3 + i];

    if (cam_info_msg->distortion_model == "plumb_bob")
    {
      distCoeffs = Mat(1,5, CV_32F);
      for (int i = 0; i < 5; i++) distCoeffs.at<float>(Point(i,0)) = cam_info_msg->D[i];
    }

    // Read sensor image
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    Mat image = cv_ptr->image;

    detectBoard(image, camMatrix, distCoeffs);
  }

  void detectBoard(const Mat& image, const Mat& camMatrix, const Mat& distCoeffs)
  {
    Mat imageCopy;

    double tick = (double)getTickCount();

    vector< int > markerIds, charucoIds;
    vector< vector< Point2f > > markerCorners, rejectedMarkers;
    vector< Point2f > charucoCorners;
    Vec3d rvec, tvec;

    // detect markers
    aruco::detectMarkers(image, dictionary, markerCorners, markerIds, detectorParams,
                         rejectedMarkers);

    // refind strategy to detect more markers
    if(refindStrategy)
      aruco::refineDetectedMarkers(image, board, markerCorners, markerIds, rejectedMarkers,
                                   camMatrix, distCoeffs);

    // interpolate charuco corners
    int interpolatedCorners = 0;
    if(markerIds.size() > 0)
      interpolatedCorners =
          aruco::interpolateCornersCharuco(markerCorners, markerIds, image, charucoboard,
                                           charucoCorners, charucoIds, camMatrix, distCoeffs);

    // estimate charuco board pose
    bool validPose = false;
    if(camMatrix.total() != 0)
      validPose = aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, charucoboard,
                                                  camMatrix, distCoeffs, rvec, tvec);

//    double currentTime = ((double)getTickCount() - tick) / getTickFrequency();
//    totalTime += currentTime;
//    totalIterations++;
//    if(totalIterations % 30 == 0) {
//      cout << cam_name_ << ": Detection Time = " << currentTime * 1000 << " ms "
//          << "(Mean = " << 1000 * totalTime / double(totalIterations) << " ms)" << endl;
//    }

    // draw results
    image.copyTo(imageCopy);
    if(markerIds.size() > 0) {
      aruco::drawDetectedMarkers(imageCopy, markerCorners);
    }

    if(showRejected && rejectedMarkers.size() > 0)
      aruco::drawDetectedMarkers(imageCopy, rejectedMarkers, noArray(), Scalar(100, 0, 255));

    if(interpolatedCorners > 0) {
      Scalar color;
      color = Scalar(255, 0, 0);
      aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, color);
    }

    if(validPose)
      aruco::drawAxis(imageCopy, camMatrix, distCoeffs, rvec, tvec, axisLength);
    else
    {
      imshow(cam_name_, imageCopy);
      waitKey(1);

      return;
    }

    Mat rmat (3,3,CV_32F);
    cv::Rodrigues (rvec, rmat);

    Eigen::Matrix<float, 3, 3> Rmat_e;
    Eigen::Matrix<double, 3, 1> tvec_e;
    cv::cv2eigen(rmat, Rmat_e);
    cv::cv2eigen(tvec, tvec_e);



    {
      boost::mutex::scoped_try_lock lock(data_ready_mtx_);
      if(lock)
      {
        T_ = Eigen::Affine3f::Identity();
        T_.matrix().block<3,3>(0,0) = Rmat_e;
        T_.matrix().block<3,1>(0,3) = tvec_e.cast<float>(); // @suppress("Invalid arguments") // @suppress("Symbol is not resolved")

        data_ready_cond_.notify_one();
      }
    }

    imshow(cam_name_, imageCopy);
    waitKey(1);
  }

  /**
   */
  static bool readDetectorParameters(string filename, Ptr<aruco::DetectorParameters> &params)
  {
    FileStorage fs(filename, FileStorage::READ);
    if(!fs.isOpened())
      return false;
    fs["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs["markerBorderBits"] >> params->markerBorderBits;
    fs["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs["errorCorrectionRate"] >> params->errorCorrectionRate;
    return true;
  }

  boost::shared_ptr<ros::NodeHandle> nh_;
  boost::shared_ptr<message_filters::Subscriber<sensor_msgs::CameraInfo> > cam_info_sub_ ;
  boost::shared_ptr<message_filters::Subscriber<sensor_msgs::Image> > image_sub_ ;
  boost::shared_ptr< message_filters::Synchronizer<SyncPolice> > sync_ ;

  // Parameters
  int squaresX;
  int squaresY;
  float squareLength;
  float markerLength;
  int dictionaryId;
  bool showRejected;
  bool refindStrategy;

  Ptr<aruco::DetectorParameters> detectorParams;

  Ptr<aruco::Dictionary> dictionary;

  float axisLength;

  Ptr<aruco::CharucoBoard> charucoboard ;
  Ptr<aruco::Board> board ;

//  double totalTime;
//  int totalIterations;

  std::string cam_name_ ;

  Eigen::Affine3f T_ ;
};

class CameraCalib
{
public:
  CameraCalib(const CommandLineParser& parser, const std::string& baxter_cam_name, const std::string& baxter_cam_info_topic, const std::string& baxter_image_topic,
              const std::string& rs_cam_name, const std::string& rs_cam_info_topic, const std::string& rs_image_topic)
  {
    bd_realsense_.reset(new boardDetector (parser, rs_cam_name, rs_cam_info_topic, rs_image_topic));
    bd_cambaxter_.reset(new boardDetector (parser, baxter_cam_name, baxter_cam_info_topic, baxter_image_topic));
  }

  virtual ~CameraCalib()
  {
    if (runner_thread_)
      runner_thread_->join();
  }

  void start ()
  {
    bd_realsense_->start();
    bd_cambaxter_->start();

    runner_thread_.reset (new boost::thread(boost::bind(&CameraCalib::run, this)));
  }

private:
  void run ()
  {
    ros::Rate rate(500);
    Eigen::Affine3f rs_T, baxter_cam_T, T_rhand_to_rcam, T_rsLink_to_rsOpticalFrame;

    boost::unique_lock<boost::mutex> lock_rs (bd_realsense_->data_ready_mtx_);
    boost::unique_lock<boost::mutex> lock_cambaxter (bd_cambaxter_->data_ready_mtx_);

    baxter_cam_to_hand_.waitForTransform("/base", "/left_hand_camera", ros::Time(0), ros::Duration(10.0));
    rs_to_camera_link_.waitForTransform("/camera_link", "/camera_rgb_optical_frame", ros::Time(0), ros::Duration(10.0));


    static tf::TransformBroadcaster br;

    Eigen::Affine3f T = Eigen::Affine3f::Identity();

    while (ros::ok())
    {

      try
      {
        tf::StampedTransform T_rhand_to_rcam_stamp ;
        baxter_cam_to_hand_.lookupTransform("/base", "/left_hand_camera", ros::Time(0), T_rhand_to_rcam_stamp);
        T_rhand_to_rcam = Eigen::Affine3f::Identity();
        Eigen::Vector3f t (T_rhand_to_rcam_stamp.getOrigin().x(), T_rhand_to_rcam_stamp.getOrigin().y(), T_rhand_to_rcam_stamp.getOrigin().z());
        Eigen::Quaternionf r (T_rhand_to_rcam_stamp.getRotation().w(), T_rhand_to_rcam_stamp.getRotation().x(), T_rhand_to_rcam_stamp.getRotation().y(), T_rhand_to_rcam_stamp.getRotation().z());
        T_rhand_to_rcam.matrix().block<3,1>(0,3) = t;
        T_rhand_to_rcam.matrix().block<3,3>(0,0) = r.toRotationMatrix();
      }
      catch (tf::TransformException&) { }

      try
      {
        tf::StampedTransform T_rsLink_to_rsOpticalFrame_stamp ;
        rs_to_camera_link_.lookupTransform("/camera_link", "/camera_rgb_optical_frame", ros::Time(0), T_rsLink_to_rsOpticalFrame_stamp);
        T_rsLink_to_rsOpticalFrame = Eigen::Affine3f::Identity();
        Eigen::Vector3f t (T_rsLink_to_rsOpticalFrame_stamp.getOrigin().x(), T_rsLink_to_rsOpticalFrame_stamp.getOrigin().y(), T_rsLink_to_rsOpticalFrame_stamp.getOrigin().z());
        Eigen::Quaternionf r (T_rsLink_to_rsOpticalFrame_stamp.getRotation().w(), T_rsLink_to_rsOpticalFrame_stamp.getRotation().x(), T_rsLink_to_rsOpticalFrame_stamp.getRotation().y(), T_rsLink_to_rsOpticalFrame_stamp.getRotation().z());
        T_rsLink_to_rsOpticalFrame.matrix().block<3,1>(0,3) = t;
        T_rsLink_to_rsOpticalFrame.matrix().block<3,3>(0,0) = r.toRotationMatrix();
      }
      catch (tf::TransformException&) { }

      bool has_bd_rs = false;
      bool has_bd_cambaxter = false;
      has_bd_rs = bd_realsense_->getBoardPose(rs_T, lock_rs);
      has_bd_cambaxter = bd_cambaxter_->getBoardPose(baxter_cam_T, lock_cambaxter);

      if (has_bd_rs && has_bd_cambaxter)
      {
        //T = T_rhand_to_rcam * baxter_cam_T * (T_rsLink_to_rsOpticalFrame * rs_T).inverse() ;
        T = T_rhand_to_rcam * baxter_cam_T * (T_rsLink_to_rsOpticalFrame * rs_T).inverse() ;
      }

      tf::Transform transform;
      transform.setOrigin( tf::Vector3(T.translation().x(), T.translation().y(), T.translation().z()) );
      Eigen::Quaternionf r (T.rotation());
      tf::Quaternion q(r.x(), r.y(), r.z(), r.w());
      transform.setRotation(q);
      br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "base", "camera_link"));



      transform.setOrigin( tf::Vector3(baxter_cam_T.translation().x(), baxter_cam_T.translation().y(), baxter_cam_T.translation().z()) );
      r = Eigen::Quaternionf(baxter_cam_T.rotation());
      q = tf::Quaternion(r.x(), r.y(), r.z(), r.w());
      transform.setRotation(q);
      br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "left_hand_camera", "o1"));





      transform.setOrigin( tf::Vector3(rs_T.translation().x(), rs_T.translation().y(), rs_T.translation().z()) );
      r = Eigen::Quaternionf(rs_T.rotation());
      q = tf::Quaternion(r.x(), r.y(), r.z(), r.w());
      transform.setRotation(q);
      br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "camera_rgb_optical_frame", "o2"));

      rate.sleep();
    }
  }

  boost::shared_ptr<boardDetector> bd_realsense_;
  boost::shared_ptr<boardDetector> bd_cambaxter_;
  boost::shared_ptr<boost::thread> runner_thread_;

  tf::TransformListener baxter_cam_to_hand_;
  tf::TransformListener rs_to_camera_link_;
};

/**
 */
int main(int argc, char** argv)
{
  ros::init(argc, argv, "detect_board");
  CommandLineParser parser(argc, argv, keys);
  parser.about(about);

  if(argc < 6)
  {
    parser.printMessage();
    return 0;
  }

  CameraCalib cc (parser, "baxter_right_arm", "cameras/left_hand_camera/camera_info",
                          "cameras/left_hand_camera/image", "real_sense", "camera/rgb/camera_info", "camera/rgb/image_raw");
  cc.start();
  ros::spin();

  return 0;
}
