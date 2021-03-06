#pragma once
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <algorithm>
#include <lm.h>

#ifdef USE_DOUBLE_PRECISION
    #define LEVMAR_DIF dlevmar_dif
#else
    #define LEVMAR_DIF slevmar_dif
#endif

namespace MopedNS {
	class NO_RANSAC_POSE_DEPTH_CPU_WU :public MopedAlg {
		double param1;
		double param2;
		int maxLMTests;
		
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
		
		void OutputMatch( vector<Match3DData *> data ) {
			for ( int i = 0; i < data.size(); i ++ ) 
				cout << data[i]->obser3d << " " << data[i]->model3d << endl;
			getchar();
		}		
				
		void outputcl(  vector<Match3DData *> cl) {
			ofstream out;
			out.open( "mediansvd.txt" );
			for ( int i = 0; i < (int)cl.size(); i ++ ) {
				out << cl[i]->model3d << "\t" << cl[i]->obser3d << "\n";
			}
			out.close();
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
		
		bool PoseSVD( Pose &pose, vector<Match3DData *> samples ) {
			//VisualizeMatch( samples );
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

		void initPose( Pose &pose ) {
			pose.rotation.init( (rand()&255)/256., (rand()&255)/256., (rand()&255)/256., (rand()&255)/256. );
			pose.translation.init( 0.,0.,0.5 );
		}		

		void testAllPoints( vector<Match3DData *> &consistentCorresp, const Pose &pose, const vector<Match3DData *> &testPoints, const Float ErrorThreshold ) {
			consistentCorresp.clear();
			foreach( corresp, testPoints ) {
				Pt<2> p = project( pose, corresp->model3d, *corresp->image );
				p -= corresp->image2d;
				Float projectionError = p[0]*p[0]+p[1]*p[1];
				if( projectionError < ErrorThreshold )
					consistentCorresp.push_back(corresp);
			}
		}
	
		bool PoseEstimation( Pose &pose, const vector<Match3DData *> &samples ) {
			initPose( pose );
			bool PoseFlag = PoseSVD( pose, samples);
			if ( PoseFlag == false )
				return false;
			else
				return true;
				int LMIterations = optimizeCamera( pose, samples, MaxLMTests );
				if( LMIterations == -1 ) continue;

				vector<LmData *> consistent;
				testAllPoints( consistent, pose, cluster, 5.0 );

				if ( (int)consistent.size() > MinNPtsObject ) {
					optimizeCamera( pose, consistent, MaxLMTests );
					return true;
				}
			}
			return false;
		}		

		static void lmFuncQuat(Float *lmPose, Float *pts2D, int nPose, int nPts2D, void *data) {
			vector<Match3DData *> &lmData = *(vector<Match3DData *> *)data;
			Pose pose;
			pose.rotation.init( lmPose );
			pose.rotation.norm();
			pose.translation.init( lmPose + 4 );
			TransformMatrix PoseTM;
			PoseTM.init( pose );
			for( int i=0; i<nPts2D/2; i++ ) {
				Pt<3> p3D;
				PoseTM.transform( p3D, lmData[i]->model3d );
				lmData[i]->image->TM.inverseTransform( p3D, p3D );
				Pt<2> p;
				p[0] = p3D[0]/p3D[2] * lmData[i]->image->intrinsicLinearCalibration[0] + lmData[i]->image->intrinsicLinearCalibration[2];
				p[1] = p3D[1]/p3D[2] * lmData[i]->image->intrinsicLinearCalibration[1] + lmData[i]->image->intrinsicLinearCalibration[3];
				if( p3D[2] < 0 ) {
					pts2D[2*i  ] = -p3D[2] + 10;
					pts2D[2*i+1] = -p3D[2] + 10;
				} 
				else {
					pts2D[2*i]   = p[0] - lmData[i]->image2d[0];
					pts2D[2*i+1] = p[1] - lmData[i]->image2d[1];
					pts2D[2*i]   *= pts2D[2*i];
					pts2D[2*i+1] *= pts2D[2*i+1];
				}
			}
		}

		bool OptimizeCamera( Pose &pose, const vector<Match3DData *> &samples, const int maxLMTests ) {
			// set up vector for LM
			Float camPoseLM[7] = {
				pose.rotation[0], pose.rotation[1], pose.rotation[2], pose.rotation[3],
				pose.translation[0], pose.translation[1], pose.translation[2] };
			// LM expects pts2D as a single vector
			vector<Float> pts2D( samples.size()*2, 0 );
			Float info[LM_INFO_SZ];
			// call levmar
			int retValue = LEVMAR_DIF(lmFuncQuat, camPoseLM, &pts2D[0], 7, samples.size()*2, maxLMTests,
							NULL, info, NULL, NULL, (void *)&samples);
			if( retValue < 0 ) 
				return false;
			pose.rotation.init( camPoseLM );
			pose.translation.init( camPoseLM + 4 );
			pose.rotation.norm();
			return true;
		}

		void VisualizeMatch( vector<Match3DData *> data ) {
			Image *img = data[0]->image;
			cv::Mat cvImage( img->height, img->width, CV_8UC1 );
			for (int y = 0; y < img->height; y++) 
				for (int x = 0; x < img->width; x++) 
					cvImage.at<uchar>(y, x) = (float)img->data[img->width*y+x];				
			
			int dataNum = data.size();

			for ( int i = 0; i < dataNum; i ++ ) {
				cv::Point pt;
				pt.x = data[i]->image2d[0];
				pt.y = data[i]->image2d[1];
				cv::circle( cvImage, pt, 5, cv::Scalar::all(0), 2 );
			}
			
			cv::imshow( "image", cvImage );
			cv::waitKey(10);			
		}
		
		
	public:
		NO_RANSAC_POSE_DEPTH_CPU_WU(double param1, double param2, int maxLMTests)
		:param1(param1), param2(param2), maxLMTests(maxLMTests) {
		}
		
		void getConfig( map<string, string> &config ) const {
			GET_CONFIG( param1 );
			GET_CONFIG( param2 );
			GET_CONFIG( maxLMTests );			
		}
		
		void setConfig( map<string, string> &config ) {
			SET_CONFIG( param1 );
			SET_CONFIG( param2 );
			SET_CONFIG( maxLMTests );						
		}
		
		void process( FrameData &frameData ) {
			
			string DescriptorType = "SIFT";
			vector< SP_Image > &images = frameData.images;
			vector< SP_DepthImage > &depthImages = frameData.depthImages ;
			vector< vector< FrameData::Match > > &matches = frameData.matches;
			vector< vector< FrameData::Cluster > > &clusters = frameData.clusters;
			vector< vector< Match3DData > > matchData;
			preprocessAllMatches( matchData, matches, images );	
			vector< pair<int,int> > tasks;
			tasks.reserve(1000);
			for( int model = 0; model < (int)clusters.size(); model ++ ){
				for(int cluster = 0; cluster < (int)clusters[model].size(); cluster ++ )
					tasks.push_back( make_pair(model, cluster) );
			}
//			#pragma omp parallel for
			for(int task = 0; task < (int)tasks.size(); task ++) {
				int model = tasks[task].first;
				int cluster = tasks[task].second;
				cout << "model: " << model << ", cluster: " << cluster << endl;
				vector<Match3DData *> cl;
				foreach( point, clusters[model][cluster] ) 
					cl.push_back( & matchData[model][point] );
//				OutputMatch(cl);
//				outputcl(cl);
				Pose pose;
				bool found = MedianSVD( pose, cl );
				if ( found == true ) {
//					bool opt = OptimizeCamera( pose, cl, maxLMTests );
//					#pragma omp critical(POSE)
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
