#include <ros/ros.h>
#include <ros/time.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp> // CV_AA

#include <math.h> // atan2
#include <sstream> // patch template to_string

#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Empty.h>
#include <ardrone_autonomy/Navdata.h>
#include <stdlib.h>


// default capture width and height
static const int FRAME_WIDTH = 640;
static const int FRAME_HEIGHT = 480;

// max number of objects to be detected in frame
static const int MAX_NUM_OBJECTS = 10; //50
// minimum and maximum object areai
static const int MIN_OBJECT_AREA = 50; // 5*5
static const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH/1.5;

static const int UPDATE_SPEED = 200; // process at 10hz
static const float AREA_DIST_CONST = 35.623f;

static const float PI = 3.1415;

class VisionSensor {

    ros::Publisher learner_pub;
    ros::Publisher not_visible_pub;
    ros::Publisher controller_pub;
    ros::Subscriber height_sub;

    image_transport::ImageTransport it_;
    image_transport::Subscriber video_sub;
    //image_transport::Publisher image_pub_;

    cv_bridge::CvImagePtr latest_image;

    float hover_distance;

    int h_min_;
    int h_max_;
    int s_min_;
    int s_max_;
    int v_min_;
    int v_max_;

    float height;
    float width;
    float last;

    bool is_first_img;

public:
    VisionSensor(ros::NodeHandle &nh_):it_(nh_)
    {
        is_first_img = true;
        //latest_image = 0;
        // green
        h_min_ = 26;
        h_max_ = 124;
        s_min_ = 74;
        s_max_ = 240;
        v_min_ = 50;
        v_max_ = 255;
        // orange
        //h_min_ = 0;
        //h_max_ = 200;
        //s_min_ = 170;
        //s_max_ = 255;
        //v_min_ = 0;
        //v_max_ = 255;

        height = width = -1.0f;
        last = 0.0f;
        hover_distance = 2.0f;

        //nh_.getParam("v_controller/hover_distance", hover_distance);

        learner_pub = nh_.advertise<std_msgs::Float32MultiArray>("v_controller/state", 2);
        not_visible_pub = nh_.advertise<std_msgs::Bool>("v_controller/visible", 1);
        controller_pub = nh_.advertise<std_msgs::Float32MultiArray>("v_controller/control_state", 1);

        video_sub = it_.subscribe("/ardrone/image_raw", 1, &VisionSensor::receive_image_cb, this);
        height_sub = nh_.subscribe("/ardrone/navdata", 1, &VisionSensor::receive_height_cb, this);

        ros::spinOnce();
    }

    bool isFirstImgLoaded()
    {
        return is_first_img;
    }

    void start()
    {
        ros::Rate loop_rate(UPDATE_SPEED);

        while (ros::ok())
        {
            if (!is_first_img)
                processing_function();
            ros::spinOnce();
            loop_rate.sleep();
        }
    }

    void receive_image_cb(const sensor_msgs::ImageConstPtr& msg)
    {
        try
        {
            // OpenCV expects color images to use BGR channel order
            latest_image = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);

            if (is_first_img) {
                height = latest_image->image.size().height;
                width = latest_image->image.size().width;
                is_first_img = false;
            }
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }
    }

    void receive_height_cb(const ardrone_autonomy::Navdata& msg)
    {
        float h = float(msg.altd) / 1000.0f;
        std_msgs::Float32MultiArray hArray;
        hArray.data.push_back(h);
        controller_pub.publish(hArray);
    }

    float sign(float value)
    {
        if (value > 0.0f) return 2.0f;
        if (value < 0.0f) return -2.0f;
        return 0.0f;
    }

    void processing_function()
    {
        ros::WallTime startt = ros::WallTime::now();
        //while (true)
        //    if (ros::ok() && latest_image != 0) break;

        float x = 0.0f, y = 0.0f, distance = 0.0f;
        find_target(latest_image, x, y, distance);

        std_msgs::Float32MultiArray pose;
        pose.data.push_back(x);
        pose.data.push_back(y);

        learner_pub.publish(pose);

        ros::WallTime endt = ros::WallTime::now();
        ros::WallDuration diff = endt - startt;
        ROS_DEBUG_STREAM_NAMED("vertical_control", "Object Detection duration " << diff.toSec());
    }

    void find_target(cv_bridge::CvImagePtr img, float &x, float &y, float &distance)
    {
        // Convert to HSV
        cv::Mat hsv;
        cv::cvtColor(img->image, hsv, CV_BGR2HSV);

        // Use smaller image for faster processing
        cv::Mat hsv_small(cv::Size(160,120), hsv.type());
        cv::resize(hsv, hsv_small, hsv_small.size());
 
        // filter for color
        cv::Mat color_filter;
        cv::inRange(hsv_small, cv::Scalar(h_min_, s_min_, v_min_), cv::Scalar(h_max_, s_max_, v_max_), color_filter);

        // eliminate noise
        morphOps(color_filter);

        std::string text;
        trackFilteredObject(color_filter, x, y, distance, text);

        cv::putText(img->image, text, cv::Point(0,50), 2, 1, cv::Scalar(0,255,0), 2);
        cv::imshow("threshold", color_filter);
        cv::imshow("frame", img->image);
        cv::waitKey(5);
    }


    void trackFilteredObject(cv::Mat &threshold, float &x, float &y, float &distance, std::string &text)
    {
        cv::Mat temp = threshold.clone();
        bool object_found = false;
        std_msgs::Bool b;
        float ref_area = 0.0f;

        // find contours of filtered image using openCV findContours functions
        // two vectors needed for output of findContours
        cv::vector< cv::vector<cv::Point> > contours;
        cv::vector<cv::Vec4i> hierarchy;
        cv::findContours(temp, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
        // use moments method to find our filtered object
        if (hierarchy.size() > 0)
        {
            int numObjects = hierarchy.size();
            // if number of objects greater than MAX_NUM_OBJECTS we have a noisy filter
            if (numObjects <= MAX_NUM_OBJECTS)
            {
                //int x = 0, y = 0;
                for (int index = 0; index >= 0; index = hierarchy[index][0])
                {
                    cv::Moments moment = cv::moments((cv::Mat)contours[index]);
                    double area = moment.m00;
                    //double area = cv::contourArea(contours[index]);

                    // if the area is less than 20px by 20px then it is probably just noise
                    // if the area is the same as the 3/2 of the image size, probably just a bad filter
                    // we only want the object with the largest area so we safe a reference area each
                    // iteration and compare it to the area in the next iteration.
                    if (area > MIN_OBJECT_AREA && area < MAX_OBJECT_AREA && area > ref_area)
                    //if (area > ref_area)
                    {
                        x = moment.m10/area;
                        y = moment.m01/area;
                        object_found = true;
                        ref_area = area;
                    }
                }
                // let user know you found an object
                if (object_found)
                {
                    float px = x / threshold.size().width;
                    float py = y / threshold.size().height;
                    x = ceil(width * px) - 12.0f;
                    y = ceil(height * py) - 12.0f;
                    distance = AREA_DIST_CONST * 1.0f / sqrt(ref_area);
                    b.data = 1;
                    not_visible_pub.publish(b);
                    rescale(x, y);
                    last = sign(y);
                    text = "Largest Contour found";
                    return; // ends function when object is found
                    //cv::putText(frame, "Tracking Object", cv::Point(0,50), 2, 1, cv::Scalar(0,255,0), 2);
                }
            }
        }

        // object not found
        x = 0.0f;
        y = last;
        distance = hover_distance;
        b.data = 0;
        not_visible_pub.publish(b);
        text = "Nothing found";
    }

    void rescale(float &x, float &y)
    {
        float width_div = float(width) / 4.0f;
        float height_div = float(height) / 3.0f;
        x = (x - width/2) / width_div;
        y = (y - height/2) / height_div;
    }

    void morphOps(cv::Mat &threshold)
    {
      // create structuring element that will be used to dilate and erode image
      // the element chosen is a 3px by 3px rectangle
      cv::Mat erode_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,2));
      // dilate with larger element so make sure object is nicely visible
      cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,2)); //8,8
      cv::Mat dilate_kernel4 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4,4)); //8,8

      cv::erode(threshold, threshold, erode_kernel);
      cv::erode(threshold, threshold, erode_kernel);
      cv::erode(threshold, threshold, erode_kernel);
      cv::dilate(threshold, threshold, dilate_kernel);
      cv::dilate(threshold, threshold, dilate_kernel4);

    }

};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "real_sensors");
    ros::NodeHandle nh;

    VisionSensor vs(nh);
    //vs.start();

    ros::Rate loop_rate(UPDATE_SPEED);
    
    while (ros::ok())
    {
        if (!vs.isFirstImgLoaded())
            vs.processing_function();
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}
