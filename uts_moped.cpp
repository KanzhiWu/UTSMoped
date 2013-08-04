/*
 * wu_moped.cpp
 *
 *  Created on: Jun 23, 2013
 *      Author: wu
 */


#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <message_filters/subscriber.h>
#include <image_transport/subscriber_filter.h>
#include <pr_msgs/ObjectPose.h>
#include <pr_msgs/ObjectPoseList.h>
#include <pr_msgs/Enable.h>
#include <cv_bridge/CvBridge.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <fstream>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>

#include <moped.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>
#include <boost/bind.hpp>
#include <string>
#include <omp.h>
#include <dirent.h>

#include <time.h>

#define foreach( i, c ) for( typeof((c).begin()) i##_hid=(c).begin(), *i##_hid2=((typeof((c).begin())*)1); i##_hid2 && i##_hid!=(c).end(); ++i##_hid) for( typeof( *(c).begin() ) &i=*i##_hid, *i##_hid3=(typeof( *(c).begin() )*)(i##_hid2=NULL); !i##_hid3 ; ++i##_hid3, ++i##_hid2)
#define eforeach( i, it, c ) for( typeof((c).begin()) it=(c).begin(), i##_hid = (c).begin(), *i##_hid2=((typeof((c).begin())*)1); i##_hid2 && it!=(c).end(); (it==i##_hid)?++it,++i##_hid:i##_hid=it) for( typeof(*(c).begin()) &i=*it, *i##_hid3=(typeof( *(c).begin() )*)(i##_hid2=NULL); !i##_hid3 ; ++i##_hid3, ++i##_hid2)
using namespace std;

using namespace MopedNS;


string fix_param_name(string s) {
	using namespace boost::algorithm;
	return replace_all_copy(replace_all_copy(replace_all_copy(s,":","_"),"-","_"),".","_");
}



class WuROS{

private:
	image_transport::Subscriber moped_sub;
	ros::Publisher moped_pub;

	Moped moped;
	Pt<4> intrinsicLinearCalibration;
	Pt<4> intrinsicNonlinearCalibration;

	int Enabled;
	ros::ServiceServer moped_enable;

	ros::NodeHandle nh_;
	image_transport::ImageTransport it_;

	message_filters::Subscriber< sensor_msgs::Image > rgbImage;
	message_filters::Subscriber< sensor_msgs::Image > depthImage;
	message_filters::Subscriber< sensor_msgs::PointCloud2 > cloud;
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::PointCloud2> wuROSsync;
	message_filters::Synchronizer<wuROSsync> sync;

public:
	WuROS():it_(nh_),
	rgbImage(nh_, "/camera/rgb/image_color", 1),
	depthImage(nh_, "/camera/depth_registered/image_rect_raw", 1),
	cloud(nh_, "/camera/depth_registered/points", 1),
	sync(wuROSsync(10), rgbImage, depthImage, cloud) {
		Enabled = 1;
		ros::NodeHandle pnh("~");
		string modelsPath;
		pnh.param("models_path", modelsPath, string("/home/wu/ros_f_ws/sandbox/uts_moped/models") );		//Add model path
		DIR *dp;
		struct dirent *dirp;
		if((dp  = opendir(modelsPath.c_str())) ==  NULL)
			throw string("Error opening \"") + modelsPath + "\"";
		vector<string> fileNames;
		while((dirp = readdir(dp)) != NULL) {
			string fileName =  modelsPath + "/" + string(dirp->d_name);
			if( fileName.rfind(".moped.xml") == string::npos )
				continue;
			fileNames.push_back( fileName );
		}

		#pragma omp parallel for
		for(int i=0; i<(int)fileNames.size(); i++) {
			sXML XMLModel;
			XMLModel.fromFile(fileNames[i]);
			#pragma omp critical(addModel)
			moped.addModel(XMLModel);
		}
		closedir(dp);

		//Set input and output topic name
		string inputImageTopicName;
		string outputObjectListTopicName;
		string EnableSrvName;
		pnh.param("input_image_topic_name", inputImageTopicName, std::string("/camera/rgb/image_rect_color"));
		pnh.param("output_object_list_topic_name", outputObjectListTopicName, std::string("/object_poses"));
		pnh.param("enable_service_name", EnableSrvName, std::string("/Enable"));

		moped_pub = nh_.advertise<pr_msgs::ObjectPoseList>( "/object_poses", 100 );
		sync.registerCallback( boost::bind( &WuROS::callback, this, _1, _2, _3 ) );
		moped_enable = nh_.advertiseService( "/Enable", &WuROS::EnableMoped, this );

		//Camera parameters
		double d1, d2, d3, d4;
		nh_.param("KK_fx", d1, 529.21);
		nh_.param("KK_fy", d2, 525.56);
		nh_.param("KK_cx", d3, 328.94);
		nh_.param("KK_cy", d4, 267.48);
		intrinsicLinearCalibration.init(d1, d2, d3, d4);

		nh_.param("kc_k1", d1, 0.264);
		nh_.param("kc_k2", d2, -0.328);
		nh_.param("kc_p1", d3, 1e-12);
		nh_.param("kc_p2", d4, 1e-12);
		intrinsicNonlinearCalibration.init(d1, d2, d3, d4);

		// read the config.hpp, load algorithm steps and parameters
		map<string,string> config = moped.getConfig();
		foreach( value, config ) {
			nh_.param( fix_param_name(value.first), value.second, value.second);
		}
		moped.setConfig(config);

		ros::Rate loop_rate(60);
	}

	bool EnableMoped(pr_msgs::Enable::Request& Req, pr_msgs::Enable::Response& Resp){
		Enabled = Req.Enable;
		Resp.ok = true;
		return true;
	}

	void imageinfo( cv::Mat image ){
		cout << "image info: " << endl;
		cout << "\t" << image.cols << image.rows << endl;
		for ( int i = 0; i < image.rows; i ++ ) {
			for ( int j = 0; j < image.cols; j ++ ) {
				cout << (int)image.at<uchar>(i, j) << ", ";
			}
			getchar();
			cout << endl;
		}

	}

	void callback(const sensor_msgs::ImageConstPtr &rgbImage, const sensor_msgs::ImageConstPtr &depthImage, const sensor_msgs::PointCloud2ConstPtr &cloud) {
		if (Enabled) {
			struct timeval start, end;
			unsigned long diff;
			gettimeofday(&start, NULL);

			// load rgb image from the message
			cv_bridge::CvImagePtr cv_ptr;
			try {
				cv_ptr = cv_bridge::toCvCopy( rgbImage, sensor_msgs::image_encodings::BGR8 );
			}
			catch ( cv_bridge::Exception& e ){
				ROS_ERROR( "Exception: %s", e.what() );
			}
			sensor_msgs::CvBridge bridge;
			IplImage *gs = bridge.imgMsgToCv( rgbImage, "mono8" );
			cv::Mat wuRgbImage = cv_ptr->image;
			vector<SP_Image> images;
			SP_Image mopedImage( new Image );
			mopedImage->name = "ROS_Image";
			mopedImage->intrinsicLinearCalibration = intrinsicLinearCalibration;
			mopedImage->intrinsicNonlinearCalibration = intrinsicNonlinearCalibration;
			mopedImage->cameraPose.translation.init( 0.,0.,0. );
			mopedImage->cameraPose.rotation.init( 0.,0.,0.,1. );
			mopedImage->width = wuRgbImage.cols;
			mopedImage->height = wuRgbImage.rows;
			mopedImage->data.resize( wuRgbImage.cols * wuRgbImage.rows );
			for (int y = 0; y < gs->height; y++)
				memcpy( &mopedImage->data[y*gs->width], &gs->imageData[y*gs->widthStep], gs->width );
			images.push_back( mopedImage );

			// load depth image from the message
			cv_bridge::CvImagePtr depth_ptr;
			try {
				depth_ptr = cv_bridge::toCvCopy( depthImage, sensor_msgs::image_encodings::TYPE_16UC1 );
			}
			catch ( cv_bridge::Exception& e ) {
				ROS_ERROR( "Exception: %s", e.what() );
			}
			cv::Mat wuDepthImage = depth_ptr->image;
			vector<SP_DepthImage> depthImages;
			SP_DepthImage mopedDepthImage( new DepthImage );
			mopedDepthImage->name = "ROS_DepthImage";
			mopedDepthImage->intrinsicLinearCalibration = intrinsicLinearCalibration;
			mopedDepthImage->intrinsicNonlinearCalibration = intrinsicNonlinearCalibration;
			mopedDepthImage->cameraPose.translation.init( 0., 0., 0. );
			mopedDepthImage->cameraPose.rotation.init( 0., 0., 0., 1. );
			mopedDepthImage->width = wuDepthImage.cols;
			mopedDepthImage->height = wuDepthImage.rows;
			uint16_t* depthDataPtr = (uint16_t*)wuDepthImage.data;
			for ( int i = 0; i < wuDepthImage.rows; i ++ ) {
				for ( int j = 0; j < wuDepthImage.cols; j ++ ) {
					mopedDepthImage->data.push_back(depthDataPtr[i*wuDepthImage.cols+j]);
				}
			}
			depthImages.push_back( mopedDepthImage );

			// load pointcloud2 from the message
			sensor_msgs::PointCloud2 cloudData = *cloud;
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPclPtr( new pcl::PointCloud<pcl::PointXYZ> );
			pcl::fromROSMsg( cloudData, *cloudPclPtr );
//			pcl::PointCloud<pcl::PointXYZ> cloudPcl = *cloudPclPtr;

			list<SP_Object> objects;
			moped.processImages( images, depthImages, cloudPclPtr, objects );
			pr_msgs::ObjectPoseList outList;
			outList.header.seq = rgbImage->header.seq;
			outList.header.frame_id = "/objdet_cam";
			outList.header.stamp = rgbImage->header.stamp;
			outList.originalTimeStamp = rgbImage->header.stamp;

			foreach( object, objects ) {
				pr_msgs::ObjectPose out;
				out.name = object->model->name;
				out.pose.position.x = object->pose.translation[0];
				out.pose.position.y = object->pose.translation[1];
				out.pose.position.z = object->pose.translation[2];
				object->pose.rotation.norm();

				// Not really necessary, but this way we always display the same half of the quaternion hypersphere
				float flip = object->pose.rotation[0] + object->pose.rotation[1] + object->pose.rotation[2] + object->pose.rotation[3];
				if( flip < 0 ) {
					object->pose.rotation[0] *= -1;
					object->pose.rotation[1] *= -1;
					object->pose.rotation[2] *= -1;
					object->pose.rotation[3] *= -1;
				}
				out.pose.orientation.x = object->pose.rotation[0];
				out.pose.orientation.y = object->pose.rotation[1];
				out.pose.orientation.z = object->pose.rotation[2];
				out.pose.orientation.w = object->pose.rotation[3];
				out.mean_quality = object->score;
				out.used_points = 10;

				list<Pt<2> > hull = object->getObjectHull((Image &) images[0]);
				foreach( pt, hull) {
				   out.convex_hull_x.push_back( (int) pt[0] );
				   out.convex_hull_y.push_back( (int) pt[1] );
				}
				outList.object_list.push_back(out);
			}

			moped_pub.publish(outList);
			gettimeofday(&end, NULL);
			diff = 1000000*(end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec);
			cout << "--Time is: " << (double)diff/1000000  << "s" << endl;
/*			int observeCnt;
			foreach( object, objects ) {
				string DescriptorType = object->model->IPs.begin()->first;
				int featNumber = object->model->IPs[DescriptorType].size();
				int ptIndex;
				MopedNS::Pt<3> pt3It;
				MopedNS::Pt<2> pt2It;

				observeCnt = 0;
				for ( ptIndex = 0; ptIndex < featNumber; ptIndex ++ ){
					if ( object->model->IPs[DescriptorType][ptIndex].observeFlag == true ) {
						observeCnt ++;
					}
				}
				cout << object->model->name << ": " << observeCnt << endl;
			}
			cout << endl;*/

			// Display some info

/*			clog << " Found " << objects.size() << " objects" << endl;
			cv::RNG rng( 0xFFFFFFFF );
			int observeCnt;
			double varphi, theta, psi;
			varphi = 0.0; theta = 0.0; psi = 0.0;
			foreach( object, objects ) {
				string DescriptorType = object->model->IPs.begin()->first;
				int icolor = (unsigned)rng;
				clog << " Found " << object->model->name << " at " << object->pose << " with score " << object->score << endl;

				varphi 	= 2*( object->pose.rotation[3]*object->pose.rotation[0] + object->pose.rotation[1]*object->pose.rotation[2] )
						  /( 1-2*(pow(object->pose.rotation[0], 2)+pow(object->pose.rotation[1], 2)) );
				theta 	= 2*( object->pose.rotation[3]*object->pose.rotation[1]-object->pose.rotation[2]*object->pose.rotation[0] );
				psi 	= 2*( object->pose.rotation[3]*object->pose.rotation[2] + object->pose.rotation[0]*object->pose.rotation[1] )
						  /( 1-2*(pow(object->pose.rotation[1], 2)+pow(object->pose.rotation[2], 2)) );
				varphi = atan( varphi ) * 180/PI;
				theta = asin( theta ) * 180/PI;
				psi = atan( psi ) * 180/PI;
				cout << "Pose Eular angle: " << varphi << ", " << theta << ", " << psi << endl;
				int featNumber = object->model->IPs[DescriptorType].size();
				int ptIndex;
				MopedNS::Pt<3> pt3It;
				MopedNS::Pt<2> pt2It;

				observeCnt = 0;
				for ( ptIndex = 0; ptIndex < featNumber; ptIndex ++ ){
					if ( object->model->IPs[DescriptorType][ptIndex].observeFlag == true ) {
						observeCnt ++;
						pt3It = object->model->IPs[DescriptorType][ptIndex].coord3D;
						pt2It = project(object->pose, pt3It, *mopedImage);
						cv::circle(wuRgbImage, cv::Point(pt2It[0], pt2It[1]), 4, cv::Scalar( icolor&255, (icolor>>8)&255, (icolor>>16)&255 ), 1.5 );
					}
				}
				cout << "In object " << object->model->name << ", observed features: " << observeCnt << endl;
			}
			cv::imshow( "image", wuRgbImage );
			cv::imshow( "depthImage", wuDepthImage );
			cv::waitKey(0);
			*/
		}
	}

};

int main(int argc, char** argv) {
	try {
		omp_set_num_threads(4);
		ros::init(argc, argv, "wu_moped");
		WuROS wuROS;
		ros::spin();
		ROS_INFO("Done!");
	}
	catch( string s ) {
		cerr << "ERROR " << s << endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
