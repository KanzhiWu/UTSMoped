#pragma once
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>

namespace MopedNS {

	class POSE_DEPTH_CPU_WU :public MopedAlg {

		int MaxRANSACTests; 		// e.g. 100
		int MaxObjectsPerCluster; 	// e.g. 4
		int NPtsAlign; 			// e.g. 5
		int MinNPtsObject; 		// e.g. 8
		Float ErrorThreshold; 		// e.g. 5

		struct Match3DData {
			Image *image;
			Pt<3> model3d;
			Pt<3> obser3d;
			Pt<2> image2d;
			double dist;
			bool operator < (const Match3DData& match) const {
				return dist > match.dist;
			}
		};
		
		bool randSample( vector<Match3DData *> &samples, const vector<Match3DData *> &cluster, unsigned int nSamples) {
			map<pair<Image *, Pt<3> >, int> used;
			deque<pair<Float, Match3DData *> > randomSamples;
			foreach( match, cluster )
				randomSamples.push_back( make_pair( (Float)rand(), match ) );
			sort( randomSamples.begin(), randomSamples.end() );
			while( used.size() < nSamples && !randomSamples.empty() ) {
				pair<Image *, Pt<3> > imageAndPoint( randomSamples.front().second->image, randomSamples.front().second->obser3d );
				if( !used[ imageAndPoint ]++ )
					samples.push_back( randomSamples.front().second );
				randomSamples.pop_front();
			}
			return used.size() == nSamples;
		}

		void testAllPoints( vector<Match3DData *> &consistentCorresp, const Pose &pose, const vector<Match3DData *> &testPoints, const Float ErrorThreshold ) {
			consistentCorresp.clear();
			foreach( corresp, testPoints ) {
				Pt<3> p = project3d( pose, corresp->model3d, *corresp->image );
				p -= corresp->obser3d;
				Float projectionError = p[0]*p[0]+p[1]*p[1]+p[2]*p[2];
				if( projectionError < ErrorThreshold )
					consistentCorresp.push_back(corresp);
			}
		}

		void initPose( Pose &pose, const vector<Match3DData *> &samples ) {
			pose.rotation.init( (rand()&255)/256., (rand()&255)/256., (rand()&255)/256., (rand()&255)/256. );
			pose.translation.init( 0.,0.,0.5 );
		}
		
		bool DistSample( vector<Match3DData *> samples ) {
			int num = samples.size();
			for ( int i = 0; i < num; i ++ ) {
				for ( int j = i+1; j < num; j ++ ) {
					Pt<3> pt1 = samples[i]->obser3d; Pt<3> pt2 = samples[j]->model3d;
					double dist = sqrt(	(pt1[0]-pt2[0])*(pt1[0]-pt2[0]) +
											(pt1[1]-pt2[1])*(pt1[1]-pt2[1]) +
											(pt1[2]-pt2[2])*(pt1[2]-pt2[2]) );
					if ( dist > 0.5 )
						return true;
				}
			}
			return true;
		}
		
		void NormalCalc( vector<Pt<3> > pts, Pt<3> &normal ) {
			Pt<3> v1, v2;
			v1[0] = pts[0][0] - pts[1][0];
			v1[1] = pts[0][1] - pts[1][1];
			v1[2] = pts[0][2] - pts[1][2];
			v2[0] = pts[0][0] - pts[2][0];
			v2[1] = pts[0][1] - pts[2][1];
			v2[2] = pts[0][2] - pts[2][2];
			normal[0] = v1[1]*v2[2] - v1[2]*v2[1];
			normal[1] = v1[2]*v2[0] - v1[0]*v2[2];
			normal[2] = v1[0]*v2[1] - v1[1]*v2[0];
		}
		
		bool PlanarSamples( vector<Match3DData *> samples ) {
			vector<Pt<3> > normalPts;
			normalPts.resize(3);
			for ( int i = 0; i < 3; i ++ ) 
				normalPts[i] = samples[i]->model3d;
			Pt<3> normal;
			NormalCalc( normalPts, normal );
			double dNormal = sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
			Pt<3> testPt = samples[3]->model3d;
			double dist = abs( (normal[0]*testPt[0] +
			                     normal[1]*testPt[1] +
			                     normal[2]*testPt[2])/(dNormal*dNormal) );
			if ( dist > 0.5 )
				return true;
			else
				return false;
		}
		
		void Match2Matrix( vector<Match3DData *> samples, Eigen::MatrixXd &matModel, Eigen::MatrixXd &matObser ) {
			for ( int i = 0; i < (int)samples.size(); i ++ ) {
				matModel(i, 0) = samples[i]->model3d[0];
				matModel(i, 1) = samples[i]->model3d[1];
				matModel(i, 2) = samples[i]->model3d[2];
				matObser(i, 0) = samples[i]->obser3d[0];
				matObser(i, 1) = samples[i]->obser3d[1];
				matObser(i, 2) = samples[i]->obser3d[2];
			}
		}
		
		void AveMatrix( Eigen::MatrixXd mat, Eigen::Vector3d &ave ) {
			double sumx = 0.0, sumy = 0.0, sumz = 0.0;
			for ( int i = 0; i < mat.rows(); i ++ ) {
				sumx += mat(i, 0);
				sumy += mat(i, 1);
				sumz += mat(i, 2);
			}
			ave(0, 0) = sumx/mat.rows();
			ave(1, 0) = sumy/mat.rows();
			ave(2, 0) = sumz/mat.rows();			
		}
		
		void MatrixNormalize( Eigen::MatrixXd &mat ) {
			double sumx = 0.0, sumy = 0.0, sumz = 0.0;
			for ( int i = 0; i < mat.rows(); i ++ ) {
				sumx += mat(i, 0);
				sumy += mat(i, 1);
				sumz += mat(i, 2);
			}
			double avex = sumx/mat.rows();
			double avey = sumy/mat.rows();
			double avez = sumz/mat.rows();
			for ( int i = 0; i < mat.rows(); i ++ ) {
				mat(i, 0) -= avex;
				mat(i, 1) -= avey;
				mat(i, 2) -= avez;
			}
		}
		
		double SumNorm( Eigen::MatrixXd mat ) {
			double sum = 0.0;
			for ( int i = 0; i < mat.rows(); i ++ ) {
				sum += sqrt( mat(i, 0)*mat(i, 0) +
				             mat(i, 1)*mat(i, 1) +
				             mat(i, 2)*mat(i, 2) );
			}
			return sum;
		}		
		
		bool PoseDepth( vector<Match3DData *> samples, Pose &pose ) {
			int ptNum = samples.size();
			Eigen::MatrixXd matModel(ptNum, 3), matObser(ptNum, 3);	
			Match2Matrix( samples, matModel, matObser );
			Eigen::Vector3d aveModel, aveObser;
			AveMatrix( matModel, aveModel );
			AveMatrix( matObser, aveObser );
			MatrixNormalize( matModel ); MatrixNormalize( matObser );
			double sumModel = SumNorm( matModel );
			double sumObser = SumNorm( matObser );		

			Eigen::Matrix3d sumProduct;
			Eigen::Matrix3d rotation;
			Eigen::Vector3d translation;
			Eigen::MatrixXd svdu, svdv;
			for ( int i = 0; i < matModel.rows(); i ++ ) {
				Eigen::MatrixXd vecModel(3,1);
				vecModel(0,0) = matModel(i,0);
				vecModel(1,0) = matModel(i,1);
				vecModel(2,0) = matModel(i,2);
				Eigen::MatrixXd vecObser(1,3);
				vecObser(0,0) = matObser(i,0);
				vecObser(0,1) = matObser(i,1);
				vecObser(0,2) = matObser(i,2);
				Eigen::Matrix3d product;
				product = vecModel * vecObser;
				sumProduct = sumProduct + product;	
			}
			Eigen::JacobiSVD<Eigen::MatrixXd> svd(sumProduct, Eigen::ComputeThinU | Eigen::ComputeThinV);
			svdu = svd.matrixU();
			svdv = svd.matrixV();
			Eigen::MatrixXd tmp = svdv*(svdu.transpose());
			double det = tmp.determinant();
			if ( abs(det-1) < 0.1 ) {
				rotation = svdv * (svdu.transpose());
			}
			else {
				svdv(0,2) *= -1;
				svdv(1,2) *= -1;
				svdv(2,2) *= -1;
				rotation = svdv * (svdu.transpose());
			}

			Eigen::MatrixXd matModelT = matModel.transpose();
			Eigen::MatrixXd matObserE = rotation*matModelT;
			Eigen::MatrixXd matObserT = matObser.transpose();
			Eigen::MatrixXd errorET = matObserE - matObserT;
			double rmse = 0.0;
			for ( int i = 0; i < errorET.rows(); i ++ ) {
				for ( int j = 0; j < errorET.cols(); j ++ ) {
					errorET(i,j) *= errorET(i,j);
					rmse += errorET(i,j);
				}
			}
			rmse = sqrt( rmse );
//			if ( rmse > 10 )
//				return false;
			translation = aveObser - rotation * aveModel;
	//		cout << "\nRotation: \n" << rotation << "\nTranslation: \n" << translation << "\n";

			// rotation matrix to quaternion
			Pt<4> quat;

			double tr = rotation(0,0)+rotation(1,1)+rotation(2,2);
			if ( tr > 0 ) {
				double S = sqrt(tr+1.0)*2;
				quat[0] = (rotation(2,1)-rotation(1,2))/S;
				quat[1] = (rotation(0,2)-rotation(2,0))/S;
				quat[2] = (rotation(1,0)-rotation(0,1))/S;
				quat[3] = S/4;
			}
			else if ( rotation(0,0) >= rotation(1,1) && rotation(0,0) >= rotation(2,2) ) {
				double S = sqrt(1.0+rotation(0,0)-rotation(1,1)-rotation(2,2))*2;
				quat[0] = S/4;
				quat[1] = (rotation(0,1)+rotation(1,0))/S;
				quat[2] = (rotation(0,2)+rotation(2,0))/S;
				quat[3] = (rotation(2,1)-rotation(1,2))/S;
			}
			else if ( rotation(1,1) >= rotation(2,2) ) {
				double S = sqrt(1.0+rotation(1,1)-rotation(0,0)-rotation(2,2))*2;
				quat[0] = (rotation(0,1)+rotation(1,0))/S;
				quat[1] = S/4;
				quat[2] = (rotation(1,2)+rotation(2,1))/S;
				quat[3] = (rotation(0,2)-rotation(2,0))/S;
			}
			else {
				double S = sqrt(1.0+rotation(2,2)-rotation(0,0)-rotation(1,1))*2;
				quat[0] = (rotation(0,2)+rotation(2,0))/S;
				quat[1] = (rotation(1,2)+rotation(2,1))/S;
				quat[2] = S/4;
				quat[3] = (rotation(1,0)-rotation(0,1))/S;
			}

			double nQuat = 0.0;
			for ( int i = 0; i < 4; i ++ ) {
				nQuat += quat[i]*quat[i];
			}
			nQuat = sqrt(nQuat);
			for ( int i = 0; i < 4; i ++ ) {
				quat[i] /= nQuat;
			}
			Pt<3> trans;
			trans[0] = translation(0,0);
			trans[1] = translation(1,0);
			trans[2] = translation(2,0);
			pose.translation = trans;
			pose.rotation = quat;
			return true;
		}
		
		void OutputMatch( vector<Match3DData *> data ) {
			for ( int i = 0; i < data.size(); i ++ ) 
				cout << data[i]->obser3d << " " << data[i]->model3d << endl;
			getchar();
		}
		
		void VisualizeMatch( vector<Match3DData *> data ) {
			int ptNum = data.size();
			Eigen::MatrixXd matModel(ptNum, 3), matObser(ptNum, 3);	
			Match2Matrix( data, matModel, matObser );
			Eigen::Vector3d aveModel, aveObser;
			AveMatrix( matModel, aveModel );
			AveMatrix( matObser, aveObser );
			MatrixNormalize( matModel ); MatrixNormalize( matObser );			
			int dataNum = data.size();
			pcl::PointCloud<pcl::PointXYZRGB> cloud;
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPtr;
			cloud.width = dataNum*2;
			cloud.height = 1;
			cloud.is_dense = false;
			cloud.points.resize( dataNum*2 );
			// define 2 colors
			uint8_t rmdl(255), gmdl(0), bmdl(0);
			uint32_t colorMdl = ( static_cast<uint32_t>(rmdl) << 16 |
			                      static_cast<uint32_t>(gmdl) << 8  |
			                      static_cast<uint32_t>(bmdl) );
			uint8_t robs(0), gobs(255), bobs(0);
			uint32_t colorObs = ( static_cast<uint32_t>(robs) << 16 |
								  static_cast<uint32_t>(gobs) << 8  |
								  static_cast<uint32_t>(bobs));
			// assign value to each point
			for ( int i = 0; i < data.size(); i ++ ) {
				cloud.points[2*i].x = matObser(i, 0);
				cloud.points[2*i].y = matObser(i, 1);
				cloud.points[2*i].z = matObser(i, 2);
				cloud.points[2*i].rgb = *reinterpret_cast<float*>(&colorMdl);
				
				cloud.points[2*i+1].x = matModel(i, 0);
				cloud.points[2*i+1].y = matModel(i, 1);
				cloud.points[2*i+1].z = matModel(i, 2);
				cloud.points[2*i+1].rgb = *reinterpret_cast<float*>(&colorObs);
			}
			
			cloudPtr.reset( new pcl::PointCloud<pcl::PointXYZRGB>(cloud) );
			boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer( new pcl:: visualization::PCLVisualizer("matched_point") );
			viewer->setBackgroundColor(255, 255, 255);
			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloudPtr);
			viewer->addPointCloud<pcl::PointXYZRGB>(cloudPtr, rgb, "matched_point");
			viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "matched_point");
			viewer->addCoordinateSystem(2.0);
			viewer->setCameraPosition(0.0, 0.0, -50.0, 0.0, 0.0, 0.0);
			while (!viewer->wasStopped ()) {
				viewer->spinOnce(100);
				boost::this_thread::sleep(boost::posix_time::microseconds (1000000));
			}			
		}
		
		void VisualizeReproMatch( vector<Match3DData *> data, Pose pose ) {
			Image *img = data[0]->image;
			cv::Mat cvImage( img->height, img->width, CV_8UC1 );
			for (int y = 0; y < img->height; y++) 
				for (int x = 0; x < img->width; x++) 
					cvImage.at<uchar>(y, x) = (float)img->data[img->width*y+x];				
			
			int dataNum = data.size();
			vector<Pt<3> > obserPts, modelPts;
			vector<Pt<2> > imagePts;
			vector<Pt<3> > nobserPts;
			imagePts.resize( dataNum );
			nobserPts.resize( dataNum );
			obserPts.resize( dataNum );
			modelPts.resize( dataNum );
			for ( int i = 0; i < dataNum; i ++ ) {
				for ( int j = 0; j < 3; j ++ ) {
					obserPts[i][j] = data[i]->obser3d[j];
					modelPts[i][j] = data[i]->model3d[j];
				}
			}
			for ( int i = 0; i < dataNum; i ++ ) {
				for ( int j = 0; j < 2; j ++ ) {
					imagePts[i][j] = data[i]->image2d[j];
					
				}					
				cv::Point pt;
				pt.x = imagePts[i][0];
				pt.y = imagePts[i][1];
				cv::circle( cvImage, pt, 5, cv::Scalar::all(0), 2 );
			}
			
			cv::imshow( "image", cvImage );
			cv::waitKey(10);
			
		}
		
		void VisualizeFeature2d( vector<Match3DData *> data ) {
			int matchNum = data.size();
			Image *img = data[0]->image;
			cv::Mat cvImage( img->height, img->width, CV_8UC1 );
			for (int y = 0; y < img->height; y++) 
				for (int x = 0; x < img->width; x++) 
					cvImage.at<uchar>(y, x) = (float)img->data[img->width*y+x];	
		
			for ( int i = 0; i < matchNum; i ++ ) {
				cout << data[i]->image2d << " ";
				cv::Point2f pt;
				pt.x = data[i]->image2d[0];
				pt.y = data[i]->image2d[1];
				cv::circle( cvImage, pt, 5, cv::Scalar::all(0), 2 );				
			}
			cv::imshow( "features", cvImage );
			cv::waitKey(0);	
			cout << endl;
		
		}
		
		bool RelativeDistanceSamples( vector<Match3DData *> data ) {
			int dataNum = data.size();
			for ( int i = 0; i < dataNum; i ++ ) {
				for ( int j = i+1; j < dataNum; j ++ ) {
					Pt<3> pti = data[i]->obser3d;
					Pt<3> ptj = data[j]->obser3d;
					if ( pti.euclDist(ptj) < 2.0 )
						return false;
				}
			}
			return true;
		}
		
		int MinVector( vector<double> data ) {
			double min = INT_MAX;
			int minidx;
			for ( int i = 0; i < data.size(); i ++ ) {
				if ( data[i] < min ) {
					min = data[i];
					minidx = i;
				}
			}
			return minidx;
		}
		
		void GetSeeds( vector<Pt<2> > image2ds, double &distance, vector<int> &seeds ) {
			// set the seeds value
			vector<int> idxs;
			for ( int i = 0; i < image2ds.size(); i ++ )
				idxs.push_back(i);
			for ( int i = 0; i < 4; i ++ ) {
				int idx = rand()%(idxs.size());
				seeds.push_back(idxs[idx]);
				idxs.erase(idxs.begin()+idx);
			}
			// just calculate the distance between seeds
			double minDist = INT_MAX;
			for ( int i = 0; i < seeds.size(); i ++ ) {
				for ( int j = i+1; j < seeds.size(); j ++ ) {
					double dist = image2ds[seeds[i]].euclDist( image2ds[seeds[j]] );
					if ( dist < minDist )
						minDist = dist;
				}
			}
			distance = minDist;
			
			/*
			// KMeans part
			vector<vector<int> > cluster;
			cluster.resize( 4 );
			for ( int i = 0; i < 4; i ++ )
				cluster[i].push_back(seeds[i]);
			// assign other features using distance
			for ( int i = 0; i < idxs.size(); i ++ ) {
				// get the distance to 4 centers
				vector<double> dists;
				dists.resize( seeds.size() );
				for ( int j = 0; j < seeds.size(); j ++ )
					dists[j] = image2ds[idxs[i]].euclDist(image2ds[seeds[j]]);
				int closeSeed = MinVector( dists );
				cluster[closeSeed].push_back( idxs[i] );
			}
			*/	
		}
		
		struct CmpSeedsDists {
			bool operator()(const std::pair<double, vector<int> > &left, const std::pair<double, vector<int> > &right) {
				return left.first > right.first;
			}
		};
		
		void OrderedRandomSample( vector<Match3DData *> &samples, vector<Match3DData> cluster, int nSamples) {
			if ( nSamples == 4 ) {
				for ( int i = 0; i < 4; i ++ )
					samples.push_back( &cluster[i] );
			}
			else {
				
				Image *img = cluster[0].image;
				cv::Mat cvImage( img->height, img->width, CV_8UC1 );
				for (int y = 0; y < img->height; y++) 
					for (int x = 0; x < img->width; x++) 
						cvImage.at<uchar>(y, x) = (float)img->data[img->width*y+x];					
				// get the features from the cluster
				vector<Pt<2> > image2ds;
				vector<Match3DData *> subCluster;
				for ( int i = 0; i < nSamples; i ++ ) {
					image2ds.push_back( cluster[i].image2d );
					subCluster.push_back( &cluster[i] );
					cv::Point2f pt;
					pt.x = cluster[i].image2d[0];
					pt.y = cluster[i].image2d[1];
					cv::circle( cvImage, pt, 4, cv::Scalar::all(0), 2 );					
				}
				// get the relative distance matrix
				vector<int> distSeeds;
				int seed1, seed2;
				double maxdist = 0.0;
				Eigen::MatrixXd distsMat( nSamples, nSamples );
				for ( int i = 0; i < subCluster.size(); i ++ ) {
					for ( int j = i; j < subCluster.size(); j ++ ) {
						if ( j == i )
							distsMat(i, j) = 0.0;
						else {
							double dist = subCluster[i]->obser3d.euclDist( subCluster[j]->obser3d );
							distsMat(i, j) = dist;
							distsMat(j, i) = dist;
							if ( dist > maxdist ) {
								maxdist = dist;
								seed1 = i;
								seed2 = j;
							}
						}
					}
				}
				maxdist = 0.0;
				int seed3;
				distSeeds.push_back( seed1 );
				distSeeds.push_back( seed2 );
				for ( int i = 0; i < subCluster.size(); i ++ ) {
					if ( i != seed1 && i != seed2 ) {
						double dist = distsMat(i, seed1) + distsMat(i, seed2);
						if ( dist > maxdist ) {
							seed3 = i;
							maxdist = dist;
						}
					}
				}
				distSeeds.push_back(seed3);
				int seed4;
				maxdist = 0.0;
				for ( int i = 0; i < subCluster.size(); i ++ ) {
					if ( i != seed1 && i != seed2 && i != seed3 ) {
						double dist = distsMat(i, seed1) +
						               distsMat(i, seed2) + 
						               distsMat(i, seed3);
						if ( dist > maxdist ) {
							seed4 = i;
							maxdist = dist;
						}
 					}
				}
				distSeeds.push_back(seed4);
				for ( int i = 0; i < distSeeds.size(); i ++ ) {
					int idx = distSeeds[i];
					samples.push_back(subCluster[idx]);
					cv::Point2f pt;
					pt.x = subCluster[idx]->image2d[0];
					pt.y = subCluster[idx]->image2d[1];
					cv::circle( cvImage, pt, 6, cv::Scalar::all(255), 2 );							
				}
				/*
				// k-means multiple times and select the cluster
				// result which has the largest inter-cluster difference
				// seeds maybe the same in the RANDOM selection
				// *********************************************
				// used check should be further implemented
				int nKMeans = 20;
				vector<pair<double, vector<int> > > seedDists;
				for ( int i = 0; i < nKMeans; i ++ ) {
					double distance;
					vector<int> seeds;
					GetSeeds( image2ds, distance, seeds );
					seedDists.push_back(make_pair(distance, seeds));
				}
				std::sort( seedDists.begin(), seedDists.end(), CmpSeedsDists() );	

				for ( int i = 0; i < seedDists[0].second.size(); i ++ ) {
					int idx = seedDists[0].second[i];
					samples.push_back( subCluster[idx] );
					
					cv::Point2f pt;
					pt.x = subCluster[idx]->image2d[0];
					pt.y = subCluster[idx]->image2d[1];
					cv::circle( cvImage, pt, 6, cv::Scalar::all(255), 2 );						
															
				}
				*/
				//cout << "\n*********************************************\n";
				cv::imshow( "features", cvImage );
				cv::waitKey(10);									
			}			
		}		
		
		


		
		
		
		bool OrderedRANSAC( Pose &pose, const vector<Match3DData *> &cluster ) {
			// calculate the center of the cluster
			Pt<3> clusterCenter;
			clusterCenter.init(0.0, 0.0, 0.0);
			int matchNum = cluster.size();
			for ( int i = 0; i < matchNum; i ++ ) {
				for ( int j = 0; j < 3; j ++ )
					clusterCenter[j] += cluster[i]->obser3d[j];
			}
			for ( int i = 0; i < 3; i ++ )
				clusterCenter[i] /= matchNum;
			vector<Match3DData> clusterClone;
			for ( int i = 0; i < matchNum; i ++ ) {
				cluster[i]->dist = sqrt( (cluster[i]->obser3d[0]-clusterCenter[0])*(cluster[i]->obser3d[0]-clusterCenter[0]) +
										 (cluster[i]->obser3d[1]-clusterCenter[1])*(cluster[i]->obser3d[1]-clusterCenter[1]) +
										 (cluster[i]->obser3d[2]-clusterCenter[2])*(cluster[i]->obser3d[2]-clusterCenter[2]));
				clusterClone.push_back( *cluster[i] );
			}
			std::sort( clusterClone.begin(), clusterClone.end() );
			
			// SVD based pose estimation
			Pose ePose;
			// check the number of remain features
			while ( clusterClone.size() > 20 ) {
				// select top 4 features in clusterClone				
				vector<Match3DData *> samples;
				OrderedRandomSample( samples, clusterClone, 8 );
				if ( RelativeDistanceSamples(samples) ) {
					// pose estimation using SVD
					initPose( ePose, samples );
					PoseDepth( samples, ePose );
					vector<Match3DData *> consistent;
					testAllPoints( consistent, ePose, cluster, ErrorThreshold );
					if ( (int)consistent.size() > clusterClone.size()*0.8 ) {
						PoseDepth( consistent, pose );
						VisualizeReproMatch( consistent, pose );					
						return true;
					}	
					clusterClone.erase(clusterClone.begin());
					clusterClone.erase(clusterClone.begin());					
				}
				else {
					// pop out the first 2 elements in clusterClone
					clusterClone.erase(clusterClone.begin());
					clusterClone.erase(clusterClone.begin());					
				}
			}
		}
		
		
		bool RANSAC( Pose &pose, const vector<Match3DData *> &cluster ) {
			vector<Match3DData *> samples;
			for ( int nIters = 0; nIters < MaxRANSACTests; nIters ++) {
				samples.clear();
				if( !randSample( samples, cluster, NPtsAlign ) ) 
					return false;
				if ( DistSample( samples ) ) {
					initPose( pose, samples );
					/*					
					if (PoseDepth( samples, PlanarFlag, pose ))
						return true;
					*/
					if ( PoseDepth( samples, pose ) ) {
						vector<Match3DData *> consistent;
						testAllPoints( consistent, pose, cluster, ErrorThreshold );
						if ( (int)consistent.size() > MinNPtsObject ) {
							cout << "...\n";
							//OutputMatch( consistent );
//							VisualizeMatch( consistent );
							PoseDepth( consistent, pose );
							VisualizeReproMatch( consistent, pose );					
							return true;
						}						
					}
				}
			}
			return false;
		}

		void preprocessAllMatches( 	vector<vector< Match3DData > > &optData,
									const vector< vector< FrameData::Match > > &matches,
									const vector< SP_Image > &images ) {
			optData.resize( matches.size() );
			for ( int model = 0; model < (int)matches.size(); model ++ )
				optData[model].resize( matches[model].size() );
			vector< pair<int,int> > tasks;
			tasks.reserve(1000);
			for(int model=0; model<(int)matches.size(); model++ )
				for(int match=0; match<(int)matches[model].size(); match++ )
					tasks.push_back( make_pair(model, match) );

			#pragma omp parallel for
			for(int task=0; task<(int)tasks.size(); task++) {
				int model=tasks[task].first;
				int match=tasks[task].second;
				const SP_Image &image = images[matches[model][match].imageIdx];

				optData[model][match].model3d = matches[model][match].coord3D;
				optData[model][match].obser3d = matches[model][match].cloud3D;
				optData[model][match].image2d = matches[model][match].coord2D;
				optData[model][match].image = image.get();				
			}
		}

		void outputcl(  vector<Match3DData *> cl) {
			ofstream out;
			out.open( "cl1.txt" );
			for ( int i = 0; i < (int)cl.size(); i ++ ) {
				out << cl[i]->model3d << "\t" << cl[i]->obser3d << "\n";
			}
			out.close();
		}

	public:

		POSE_DEPTH_CPU_WU( int MaxRANSACTests, int MaxObjectsPerCluster, int NPtsAlign, int MinNPtsObject, Float ErrorThreshold )
		: MaxRANSACTests(MaxRANSACTests), MaxObjectsPerCluster(MaxObjectsPerCluster), NPtsAlign(NPtsAlign),
		  MinNPtsObject(MinNPtsObject), ErrorThreshold(ErrorThreshold) {
		}

		void getConfig( map<string,string> &config ) const {
			GET_CONFIG( MaxRANSACTests );
			GET_CONFIG( MaxObjectsPerCluster );
			GET_CONFIG( NPtsAlign );
			GET_CONFIG( MinNPtsObject );
			GET_CONFIG( ErrorThreshold );
		}

		void setConfig( map<string,string> &config ) {
			SET_CONFIG( MaxRANSACTests );
			SET_CONFIG( MaxObjectsPerCluster );
			SET_CONFIG( NPtsAlign );
			SET_CONFIG( MinNPtsObject );
			SET_CONFIG( ErrorThreshold );
		}

		void process( FrameData &frameData ) {
			int NObjectsCluster = MaxObjectsPerCluster;
			string DescriptorType = "SIFT";
			vector< SP_Image > &images = frameData.images;
			vector< SP_DepthImage > &depthImages = frameData.depthImages ;
			vector< vector< FrameData::Match > > &matches = frameData.matches;
			vector< vector< FrameData::Cluster > > &clusters = frameData.clusters;
			vector< vector< Match3DData > > matchData;
			preprocessAllMatches( matchData, matches, images );
			vector< pair<int,int> > tasks;
			tasks.reserve(1000);
			int model;
			for( model=0; model<(int)clusters.size(); model++ ){
				for(int cluster=0; cluster<(int)clusters[model].size(); cluster++ )
					for(int obj=0; obj<NObjectsCluster; obj++)
						tasks.push_back( make_pair(model, cluster) );
			}

			//#pragma omp parallel for
			for(int task = 0; task < (int)tasks.size(); task ++) {
				int model = tasks[task].first;
				int cluster = tasks[task].second;
				vector<Match3DData *> cl;
				foreach( point, clusters[model][cluster] ) 
					cl.push_back( & matchData[model][point] );
//				OutputMatch(cl);
//				outputcl(cl);
				Pose pose;
//				bool found = RANSAC( pose, cl );
				bool found = OrderedRANSAC( pose, cl );
				if ( found > 0 ) {
					#pragma omp critical(POSE)
					{
						SP_Object obj(new Object);
						frameData.objects->push_back(obj);
						obj->pose = pose;
						obj->model = (*models)[model];
					}
				}
			}
			if( _stepName == "POSE" )
				frameData.oldObjects = *frameData.objects;
		}
	};
};
