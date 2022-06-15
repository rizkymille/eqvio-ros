// Copyright (C) 2021 Pieter van Goor
// 
// This file is part of EqF VIO.
// 
// EqF VIO is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// EqF VIO is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with EqF VIO.  If not, see <http://www.gnu.org/licenses/>.

#include "geometry_msgs/PoseStamped.h"
#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/image_encodings.h"
#include "cv_bridge/cv_bridge.h"

#include "yaml-cpp/yaml.h"

#include "eqvio/VIOFilter.h"
#include "eqvio/VIOFilterSettings.h"

//#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "GIFT/PointFeatureTracker.h"
//#include "GIFT/Visualisation.h"

struct CallbackStruct {
    GIFT::PointFeatureTracker featureTracker;
    GIFT::GICameraPtr cameraPtr;
    VIOFilter filter;
    ros::Publisher pose_publisher;

    void callbackImu(const sensor_msgs::Imu& msg);
    void callbackImage(const sensor_msgs::ImageConstPtr& msg);
};

VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp);

int main(int argc, char* argv[]) {
    ros::init(argc, argv, "eqvio");
    ros::NodeHandle nh;
    
    //cv::namedWindow("feature");

    // Load configuration information
    std::string eqvioConfig_fname, giftConfig_fname, cameraIntrinsics_fname;
    bool configuration_flag = nh.getParam("/eqvio/eqvio_config", eqvioConfig_fname) &&
                              nh.getParam("/eqvio/gift_config", giftConfig_fname) &&
                              nh.getParam("/eqvio/camera_intrinsics", cameraIntrinsics_fname);
    ROS_ASSERT(configuration_flag);
    const YAML::Node eqvioConfig = YAML::LoadFile(eqvioConfig_fname);
    const YAML::Node giftConfig = YAML::LoadFile(giftConfig_fname);

    // Initialise the filter, camera, feature tracker
    CallbackStruct cbSys;
    VIOFilter::Settings filterSettings(eqvioConfig["eqf"]);
    cbSys.filter = VIOFilter(filterSettings);
    ROS_INFO_STREAM("EqVIO configured from\n" << eqvioConfig_fname);

    cbSys.cameraPtr = std::make_shared<GIFT::PinholeCamera>(cv::String(cameraIntrinsics_fname));
    ROS_INFO_STREAM("Camera configured from\n" << cameraIntrinsics_fname);

    cbSys.featureTracker = GIFT::PointFeatureTracker(cbSys.cameraPtr);
    cbSys.featureTracker.settings.configure(giftConfig["GIFT"]);
    ROS_INFO_STREAM("GIFT configured from\n" << giftConfig_fname);
    

    // Set up publishers and subscribers
    cbSys.pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("eqvio/pose", 50);

    ros::Subscriber subImu = nh.subscribe("/eqvio/imu", 100, &CallbackStruct::callbackImu, &cbSys);
    ros::Subscriber subImage = nh.subscribe("/eqvio/image", 5, &CallbackStruct::callbackImage, &cbSys);

    ros::spin();
    
    //cv::destroyWindow("feature");
    
    return 0;
}

void CallbackStruct::callbackImu(const sensor_msgs::Imu& msg){
    // Convert to IMU measurement

    ROS_DEBUG("IMU Message Received.");

    IMUVelocity imuVel;
    imuVel.stamp = msg.header.stamp.toSec();
    imuVel.gyr << msg.angular_velocity.x,msg.angular_velocity.y,msg.angular_velocity.z;
    imuVel.acc << msg.linear_acceleration.x,msg.linear_acceleration.y,msg.linear_acceleration.z;

    this->filter.processIMUData(imuVel);
}

void CallbackStruct::callbackImage(const sensor_msgs::ImageConstPtr& msg) {

    ROS_DEBUG("Image Message Received.");

    static VisionMeasurement visionData;

    cv_bridge::CvImagePtr cv_ptr;
    if (msg->encoding == "rgb8") {
    	cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC3);
    	cv::cvtColor(cv_ptr->image, cv_ptr->image, cv::COLOR_RGB2GRAY);
    }
    else {
    	cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_8UC1);
    }
    
    const VisionMeasurement& featurePrediction = filter.getFeaturePredictions(cameraPtr, msg->header.stamp.toSec());
    featureTracker.processImage(cv_ptr->image, featurePrediction.ocvCoordinates());
    visionData = convertGIFTFeatures(featureTracker.outputFeatures(), msg->header.stamp.toSec());
    visionData.cameraPtr = cameraPtr;
    filter.processVisionData(visionData);

    /*
    static cv::Mat show_img;
    show_img = GIFT::drawFeatureImage(cv_ptr->image, features);
    cv::imshow("feature", show_img);
    cv::waitKey(30);
    */

    // Write pose message
    VIOState estimatedState = filter.stateEstimate();
    geometry_msgs::PoseStamped poseMsg;
    poseMsg.header.frame_id = "map";
    poseMsg.header.stamp = msg->header.stamp;

    const Eigen::Quaterniond& attitude = estimatedState.sensor.pose.R.asQuaternion();
    const Eigen::Vector3d& position = estimatedState.sensor.pose.x;
    poseMsg.pose.orientation.w = attitude.w();
    poseMsg.pose.orientation.x = attitude.x();
    poseMsg.pose.orientation.y = attitude.y();
    poseMsg.pose.orientation.z = attitude.z();
    poseMsg.pose.position.x = position.x();
    poseMsg.pose.position.y = position.y();
    poseMsg.pose.position.z = position.z();

    pose_publisher.publish(poseMsg);
}

VisionMeasurement convertGIFTFeatures(const std::vector<GIFT::Feature>& GIFTFeatures, const double& stamp) {
    VisionMeasurement measurement;
    measurement.stamp = stamp;
    std::transform(
        GIFTFeatures.begin(), GIFTFeatures.end(),
        std::inserter(measurement.camCoordinates, measurement.camCoordinates.begin()),
        [](const GIFT::Feature& f) { return std::make_pair(f.idNumber, f.camCoordinatesEigen()); });
    return measurement;
}
