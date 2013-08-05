/*
 * DEPTH_VERIFICATION.hpp
 *
 *  Created on: Jul 30, 2013
 *      Author: wu
 */
 
#pragma once
#include <Eigen/Dense>
#include <Eigen/SVD>


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
		};

		// This function populates "samples" with nSamples references to object correspondences
		// The samples are randomly choosen, and aren't repeated
		bool randSample( vector<Match3DData *> &samples, const vector<Match3DData *> &cluster, unsigned int nSamples) {
			// Do not add a correspondence of the same image at the same coordinate
			map<pair<Image *, Pt<3> >, int> used;
			// Create a vector of samples prefixed with a random int. The int has preference over the pointer when sorting the vector.
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
			cout << "test all points ...\n";
			consistentCorresp.clear();
			foreach( corresp, testPoints ) {
				Pt<3> p = project3d( pose, corresp->model3d, *corresp->image );
				cout << " " << corresp->obser3d << " " << p;
				p -= corresp->obser3d;
				Float projectionError = p[0]*p[0]+p[1]*p[1]+p[2]*p[2];
				cout << " " << projectionError << endl;
				if( projectionError < ErrorThreshold )
					consistentCorresp.push_back(corresp);
			}
		}

		void initPose( Pose &pose, const vector<Match3DData *> &samples ) {
			pose.rotation.init( (rand()&255)/256., (rand()&255)/256., (rand()&255)/256., (rand()&255)/256. );
			pose.translation.init( 0.,0.,0.5 );
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
			int numSamples = (int)samples.size();
			vector<Pt<3> > pts;
			pts.resize( numSamples );
			for ( int i = 0; i < numSamples; i ++ ) {
				Pt<3> pt = samples[i]->model3d;
				pts[i] = pt;
			}
			map<int, int> used;
			vector<Pt<3> > normalPts;
			int idx = 0;
			while (idx < 3) {
				int seed = rand()%numSamples;
				if ( !used[seed] ++ ) {
					normalPts.push_back( pts[seed] );
					idx ++;
				}
			}
			Pt<3> normal;
			NormalCalc( normalPts, normal );
			double proDist = 0.;
			double lNormal = sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
			for ( int i = 0; i < numSamples; i ++ ) {
				if (!used[i] ++) {
					double dist =abs(( normal[0]*pts[i][0] +
										normal[1]*pts[i][1] +
										normal[2]*pts[i][2])/(lNormal*lNormal) );
					proDist += dist;
				}
			}
			double DistThreshold = 5.0;
			if ( proDist > DistThreshold )
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
		
		void PoseDepth( vector<Match3DData *> samples, bool flag, Pose &pose ) {
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
				Eigen::JacobiSVD<Eigen::MatrixXd> svd(sumProduct, Eigen::ComputeThinU | Eigen::ComputeThinV);
				svdu = svd.matrixU();
				svdv = svd.matrixV();
								
			}
			if ( flag == false ) {
				rotation = svdv * (svdu.transpose());
			}
			else {
				svdv(0,2) *= -1;
				svdv(1,2) *= -1;
				svdv(2,2) *= -1;
				rotation = svdv * (svdu.transpose());
			}
			translation = aveObser - rotation * aveModel;
			Pt<4> rot;
			double tr = rotation(0,0) + rotation(1,1) + rotation(2,2);
			if ( tr > 0 ) {
				double S = sqrt(tr+1.0)*2;
				rot[0] = (rotation(2,1) - rotation(1,2))/S;
				rot[1] = (rotation(0,2) - rotation(2,0))/S;
				rot[2] = (rotation(1,0) - rotation(0,1))/S;
				rot[3] = 0.25*S;
			}
			else if ( rotation(0,0)>rotation(1,1) &&
			           rotation(0,0)>rotation(2,2) ) {
				double S = sqrt(1.0+rotation(0,0)-rotation(1,1)-rotation(2,2))*2;
				rot[0] = 0.25*S;
				rot[1] = (rotation(0,1)+rotation(1,0))/S;
				rot[2] = (rotation(0,2)+rotation(2,0))/S;
				rot[3] = (rotation(2,1)-rotation(1,2))/S;
			}
			else if ( rotation(1,1) > rotation(2,2) ) {
				double S = sqrt(1.0+rotation(1,1)-rotation(0,0)-rotation(2,2))*2;
				rot[0] = (rotation(0,1)+rotation(1,0))/S;
				rot[1] = 0.25*S;
				rot[2] = (rotation(1,2)+rotation(2,1))/S;
				rot[3] = (rotation(0,2)-rotation(2,0))/S;
			}
			else {
				double S = sqrt(1.0+rotation(2,2)-rotation(0,0)-rotation(1,1))*2;
				rot[0] = (rotation(0,2)+rotation(2,0))/S;
				rot[1] = (rotation(1,2)+rotation(2,1))/S;
				rot[2] = 0.25*S;
				rot[3] = (rotation(1,0)-rotation(0,1))/S;
			}
			
			Pt<3> trans;
			trans[0] = translation(0,0);
			trans[1] = translation(1,0);
			trans[2] = translation(2,0);
			pose.translation = trans;
			pose.rotation[0] = rot[0];
			pose.rotation[1] = rot[1];
			pose.rotation[2] = rot[2];
			pose.rotation[3] = rot[3]; 
			
			cout << pose.rotation << " " << pose.translation; getchar();
			/*
			double phi, theta, psi;
			phi 	= 2*( pose.rotation[3]*pose.rotation[0] + pose.rotation[1]*pose.rotation[2] )
					  /( 1-2*(pow(pose.rotation[0], 2)+pow(pose.rotation[1], 2)) );
			theta 	= 2*( pose.rotation[3]*pose.rotation[1]-pose.rotation[2]*pose.rotation[0] );
			psi 	= 2*( pose.rotation[3]*pose.rotation[2] + pose.rotation[0]*pose.rotation[1] )
					  /( 1-2*(pow(pose.rotation[1], 2)+pow(pose.rotation[2], 2)) );		
			phi = atan( phi ) * 180/PI;
			theta = asin( theta ) * 180/PI;
			psi = atan( psi ) * 180/PI;
			cout << "Pose Eular angle: " << phi << ", " << theta << ", " << psi << endl;	
			*/
		}
		
		bool RANSAC( Pose &pose, const vector<Match3DData *> &cluster ) {
			vector<Match3DData *> samples;
			for ( int nIters = 0; nIters < MaxRANSACTests; nIters ++) {
				samples.clear();
				if( !randSample( samples, cluster, NPtsAlign ) ) 
					return false;
				initPose( pose, samples );
//				bool PlanarFlag = PlanarSamples( samples );
				bool PlanarFlag = true;
				PoseDepth( samples, PlanarFlag, pose );
				vector<Match3DData *> consistent;
				testAllPoints( consistent, pose, cluster, ErrorThreshold );
				cout << "consistent size: " << (int)consistent.size() << endl;
				if ( (int)consistent.size() > MinNPtsObject ) {
//					optimizeCamera( pose, consistent, MaxLMTests );
					return true;
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

//			#pragma omp parallel for
			for(int task = 0; task < (int)tasks.size(); task ++) {
				int model = tasks[task].first;
				int cluster = tasks[task].second;
				vector<Match3DData *> cl;
				foreach( point, clusters[model][cluster] )
					cl.push_back( & matchData[model][point] );
//				getchar();
				outputcl(cl);
				Pose pose;
				bool found = RANSAC( pose, cl );
				/* if found is false, no consistent object is found in the cluster
				 * However, cl is used instead of original cluster data,
				 * cluster store the index for match in each frameData and each object
				 * specificly
				 * The detail about how to add the object to the frameData.objects:
				 * 		It seems it didn't follow the found (true or false), the
				 * 		default threads set which is 4, the number of push_back
				 * 		  */
				if ( found > 0 ) {
					cout << "found object ...\n";
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
