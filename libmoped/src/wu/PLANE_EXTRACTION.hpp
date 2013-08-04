/*
 * DEPTH_VERIFICATION.hpp
 *
 *  Created on: Jul 18, 2013
 *      Author: wu
 */

#pragma once


namespace MopedNS {
	class PLANE_EXTRACTION:public MopedAlg {
		double param1;
		double param2;
		double param3;
		double param4;
		double param5;

	public:
		PLANE_EXTRACTION(double param1, double param2, double param3, double param4, double param5)
		:param1(param1), param2(param2), param3(param3), param4(param4), param5(param5) {
		}
		
		void getConfig( map<string, string> &config ) const {
			GET_CONFIG( param1 );
			GET_CONFIG( param2 );
			GET_CONFIG( param3 );
			GET_CONFIG( param4 );
			GET_CONFIG( param5 );
		}
		
		void setConfig( map<string, string> &config ) {
			SET_CONFIG( param1 );
			SET_CONFIG( param2 );
			SET_CONFIG( param3 );
			SET_CONFIG( param4 );
			SET_CONFIG( param5 );
		}
		
		void process( FrameData &frameData ) {
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = frameData.cloudPclPtr;
			// cloudMask[height][width]
			vector<vector<bool> > &cloudMask = frameData.cloudMask;
			pcl::PointCloud<pcl::PointXYZ>::Ptr trueCloud(new pcl::PointCloud<pcl::PointXYZ>);
			
			cloudMask.resize( cloud->height );
			int pointsNo = 0;
			for ( int i = 0; i < cloud->height; i ++ ) {
				cloudMask[i].resize( cloud->width );
				for ( int j = 0; j < cloud->width; j ++ ) {
					cloudMask[i][j] = true;
					if ( cloud->points[i*cloud->width+j].x != NULL &&
					     cloud->points[i*cloud->width+j].y != NULL &&
					     cloud->points[i*cloud->width+j].z != NULL ) {
						pcl::PointXYZ tmppt;
						tmppt.x = cloud->points[i*cloud->width+j].x;
						tmppt.y = cloud->points[i*cloud->width+j].y;
						tmppt.z = cloud->points[i*cloud->width+j].z;
						trueCloud->points.push_back(tmppt);
						pointsNo ++;
//						cloudMask[i][j] = true;		
					}
				}
			}
			trueCloud->width = trueCloud->points.size();
			trueCloud->height = 1;
			
			// start extracting support plane
			
			if ( (float)pointsNo/(cloud->width*cloud->height) > 0.5 ) {
//				std::cout << "PointCloud before filtering has: " << trueCloud->points.size () << " data points." << std::endl; 
				pcl::VoxelGrid<pcl::PointXYZ> vg;
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_f(new pcl::PointCloud<pcl::PointXYZ>),
				                                    cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
				vg.setInputCloud(trueCloud);
				vg.setLeafSize(0.01f, 0.01f, 0.01f);
				vg.filter(*cloud_filtered);
//				std::cout << "PointCloud after filtering has: " << cloud_filtered->points.size ()  << " data points." << std::endl;
				pcl::SACSegmentation<pcl::PointXYZ> seg;
				pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
				pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane (new pcl::PointCloud<pcl::PointXYZ> ());
				pcl::PCDWriter writer;
				seg.setOptimizeCoefficients (true);
				seg.setModelType (pcl::SACMODEL_PLANE);
				seg.setMethodType (pcl::SAC_RANSAC);
				// this should be param1 and param2
				seg.setMaxIterations (100);
				seg.setDistanceThreshold (0.02);
				
				int i=0, nr_points = (int) cloud_filtered->points.size ();
				// The largest patch of plane is needed only
				seg.setInputCloud (cloud_filtered);
				seg.segment (*inliers, *coefficients);
				if (inliers->indices.size () == 0) {
//					cout << "Could not estimate a planar model for the given dataset." << endl;
				}
				// Extract the planar inliers from the input cloud
				else {
					pcl::ExtractIndices<pcl::PointXYZ> extract;
					extract.setInputCloud(cloud_filtered);
					extract.setIndices(inliers);
					extract.setNegative(false);
					// Get the points associated with the planar surface
					extract.filter (*cloud_plane);
//					std::cout << "PointCloud representing the planar component: " << cloud_plane->points.size() << " data points." << std::endl;
					// Get the support plane equation
					int it = 0;
					pcl::PointXYZ nvecall(.0, .0, .0);
					pcl::PointXYZ centerpt(.0, .0, .0);
					// iteration no is param3
					while (it < 1000) {
						int idx1 = rand()%(cloud_plane->points.size());
						int idx2 = rand()%(cloud_plane->points.size());
						int idx3 = rand()%(cloud_plane->points.size());
						pcl::PointXYZ vec1, vec2, nvec;
						if ( idx1 != idx2 && idx2 != idx3 && idx1 != idx3) {
							vec1.x = cloud_plane->points[idx1].x - cloud_plane->points[idx2].x; 
							vec1.y = cloud_plane->points[idx1].y - cloud_plane->points[idx2].y; 
							vec1.z = cloud_plane->points[idx1].z - cloud_plane->points[idx2].z;
							vec2.x = cloud_plane->points[idx1].x - cloud_plane->points[idx3].x;
							vec2.y = cloud_plane->points[idx1].y - cloud_plane->points[idx3].y;
							vec2.z = cloud_plane->points[idx1].z - cloud_plane->points[idx3].z;
							nvec.x = vec1.y*vec2.z - vec1.z*vec2.y;
							nvec.y = vec1.z*vec2.x - vec1.x*vec2.z;
							nvec.z = vec1.x*vec2.y - vec1.y*vec2.x;
							double vecnorm = sqrt(nvec.x*nvec.x + nvec.y*nvec.y + nvec.z*nvec.z);
							nvec.x = nvec.x/vecnorm; nvec.y = nvec.y/vecnorm; nvec.z = nvec.z/vecnorm;
							if (nvec.y < 0) {
								nvec.x *= -1.; nvec.y *= -1.; nvec.z *= -1.;
							}								
						}
						nvecall.x += nvec.x;
						nvecall.y += nvec.y;
						nvecall.z += nvec.z;
						centerpt.x += cloud_plane->points[idx1].x; 
						centerpt.x += cloud_plane->points[idx2].x; 
						centerpt.x += cloud_plane->points[idx3].x;
						centerpt.y += cloud_plane->points[idx1].y; 
						centerpt.y += cloud_plane->points[idx2].y; 
						centerpt.y += cloud_plane->points[idx3].y;
						centerpt.z += cloud_plane->points[idx1].z; 
						centerpt.z += cloud_plane->points[idx2].z; 
						centerpt.z += cloud_plane->points[idx3].z;
						it ++;
					}
					nvecall.x /= 1000; nvecall.y /= 1000; nvecall.z /= 1000;
					centerpt.x /= 3000; centerpt.y /= 3000; centerpt.z /= 3000;
					double shift = nvecall.x*centerpt.x +
					               nvecall.y*centerpt.y +
					               nvecall.z*centerpt.z;
					double normd = sqrt( nvecall.x*nvecall.x +
					                     nvecall.y*nvecall.y +
					                     nvecall.z*nvecall.z );
//					cv::Mat cloudMaskImg( cloud->height, cloud->width, CV_8UC1, cv::Scalar::all(255) );
					for ( int i = 0; i < cloud->height; i ++ ) {
						for ( int j = 0; j < cloud->width; j ++ ) {
							if ( cloud->points[i*cloud->width+j].x != NULL &&
								 cloud->points[i*cloud->width+j].y != NULL &&
								 cloud->points[i*cloud->width+j].z != NULL ) {
								double dist = nvecall.x*cloud->points[i*cloud->width+j].x +
								              nvecall.y*cloud->points[i*cloud->width+j].y +
								              nvecall.z*cloud->points[i*cloud->width+j].z -
								              shift;
								dist /= normd;
								// this is param4
								if (dist > -0.01) {
//									cloudMaskImg.at<uchar>(i, j) = 0;
									cloudMask[i][j] = false;
								}
							}

						}
					}
//					cv::imshow( "cloudMask", cloudMaskImg );
//					cv::waitKey(0);				
				}				
			}
		}
	};

};
