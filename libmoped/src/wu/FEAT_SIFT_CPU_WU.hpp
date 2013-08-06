/*
 * FEAT_SIFT_CPU.hpp
 *
 *  Created on: Jun 23, 2013
 *      Author: wu
 */

#pragma once

#include <siftfast.h>

extern int DoubleImSize;

namespace MopedNS {

	class FEAT_SIFT_CPU_WU:public MopedAlg {

		string ScaleOrigin;
		
		bool CloudDepthExist( pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, double row, double col ) {
			double x = cloud->points[row*cloud->width+col].x*100;
			double y = cloud->points[row*cloud->width+col].y*100;
			double z = cloud->points[row*cloud->width+col].z*100;
			if ( x > -500.0 && x < 500.0 &&
			     y > -500.0 && y < 500.0 &&
			     z > 40.0 && z < 300.0 ) {
				return true;
			}
			else
				return false;
		}

	public:

		FEAT_SIFT_CPU_WU( string ScaleOrigin )
		: ScaleOrigin(ScaleOrigin) {
		}

		void getConfig( map<string,string> &config ) const {

			GET_CONFIG( ScaleOrigin );
		}

		void setConfig( map<string,string> &config ) {

			SET_CONFIG( ScaleOrigin );
			if( ScaleOrigin=="-1" )
				DoubleImSize=1;
			else
				DoubleImSize=0;
		}

		void process( FrameData &frameData ) {
			vector<vector<bool> > cloudMask = frameData.cloudMask;
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = frameData.cloudPclPtr;
			
//			cout << "FEAT_SIFT_CPU\n";
			for( int i=0; i<(int)frameData.images.size(); i++) {
				Image *img = frameData.images[i].get();
				vector<FrameData::DetectedFeature> &detectedFeatures = frameData.detectedFeatures[_stepName];
				// Convert to a floating point image with pixels in range [0,1].
				SFImage image = CreateImage(img->height, img->width);
				cv::Mat cvImage( img->height, img->width, CV_8UC1 );
				for (int y = 0; y < img->height; y++) {
					for (int x = 0; x < img->width; x++) {
						image->pixels[y*image->stride+x] = ((float) img->data[img->width*y+x]) * 1./255.;
						cvImage.at<uchar>(y, x) =  (float) img->data[img->width*y+x];
					}
				}				
				// GetKeypoints uses ./libmoped/libs/libsiftfast-1.1-src/libsiftfast.cpp for detail
				Keypoint keypts = GetKeypoints(image);
				Keypoint key = keypts;
				while (key) {
					double m = key->row, n = key->col;
					//bool CloudFlag = CloudDepthExist( cloud, m, n );
					if ( cloudMask[m][n] == true /*&& CloudFlag == true*/ ) {
						bool CloudFlag = CloudDepthExist( cloud, m, n );
						if ( CloudFlag == true ) { 
							detectedFeatures.resize(detectedFeatures.size()+1);
							detectedFeatures.back().imageIdx = i;
							detectedFeatures.back().descriptor.resize(128);
							for (int x=0; x<128; x++) {
								detectedFeatures.back().descriptor[x] = key->descrip[x];
							}
							detectedFeatures.back().coord2D[0] = key->col;
							detectedFeatures.back().coord2D[1] = key->row;
//							cv::Point pt;
//							pt.x = key->col;
//							pt.y = key->row;
//							cv::circle( cvImage, pt, 5, cv::Scalar::all(0), 2 );
							detectedFeatures.back().cloud3D[0] = cloud->points[m*cloud->width+n].x*100;
							detectedFeatures.back().cloud3D[1] = cloud->points[m*cloud->width+n].y*100;
							detectedFeatures.back().cloud3D[2] = cloud->points[m*cloud->width+n].z*100;
						}
					}
					key = key->next;
				}	
				FreeKeypoints(keypts);
				DestroyAllImages();   // we can't destroy just one!
//				cv::imshow( "feat", cvImage );
//				cv::waitKey( 10 );
			}
		}
	};
};
