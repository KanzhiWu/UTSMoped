/*
 * POSE_RANSAC_LM_DIFF_REPROJECTION_CPU.hpp
 *
 *  Created on: Jun 23, 2013
 *      Author: wu
 */

#pragma once
#include <lm.h>

#ifdef USE_DOUBLE_PRECISION
    #define LEVMAR_DIF dlevmar_dif
#else
    #define LEVMAR_DIF slevmar_dif
#endif

namespace MopedNS {

	class POSE_RANSAC_LM_DIFF_REPROJECTION_CPU :public MopedAlg {

		int MaxRANSACTests; 		// e.g. 500
		int MaxLMTests;     		// e.g. 500
		int MaxObjectsPerCluster; 	// e.g. 4
		int NPtsAlign; 			// e.g. 5
		int MinNPtsObject; 		// e.g. 8
		Float ErrorThreshold; 		// e.g. 5

		struct LmData {

			Image *image;

			Pt<2> coord2D;
			Pt<3> coord3D;
		};

		// This function populates "samples" with nSamples references to object correspondences
		// The samples are randomly choosen, and aren't repeated
		bool randSample( vector<LmData *> &samples, const vector<LmData *> &cluster, unsigned int nSamples) {

			// Do not add a correspondence of the same image at the same coordinate
			map< pair<Image *, Pt<2> >, int > used;

			// Create a vector of samples prefixed with a random int. The int has preference over the pointer when sorting the vector.
			deque< pair< Float, LmData * > > randomSamples;
			foreach( match, cluster )
				randomSamples.push_back( make_pair( (Float)rand(), match ) );
			sort( randomSamples.begin(), randomSamples.end() );

			while( used.size() < nSamples && !randomSamples.empty() ) {

				pair<Image *, Pt<2> > imageAndPoint( randomSamples.front().second->image, randomSamples.front().second->coord2D );

				if( !used[ imageAndPoint ]++ )
					samples.push_back( randomSamples.front().second );

				randomSamples.pop_front();
			}

			return used.size() == nSamples;
		}

		static void lmFuncQuat(Float *lmPose, Float *pts2D, int nPose, int nPts2D, void *data) {

			vector<LmData *> &lmData = *(vector<LmData *> *)data;
			Pose pose;
			pose.rotation.init( lmPose );
			pose.rotation.norm();
			pose.translation.init( lmPose + 4 );

			TransformMatrix PoseTM;
			PoseTM.init( pose );

			for( int i=0; i<nPts2D/2; i++ ) {

				Pt<3> p3D;
				PoseTM.transform( p3D, lmData[i]->coord3D );
				lmData[i]->image->TM.inverseTransform( p3D, p3D );

				Pt<2> p;
				p[0] = p3D[0]/p3D[2] * lmData[i]->image->intrinsicLinearCalibration[0] + lmData[i]->image->intrinsicLinearCalibration[2];
				p[1] = p3D[1]/p3D[2] * lmData[i]->image->intrinsicLinearCalibration[1] + lmData[i]->image->intrinsicLinearCalibration[3];


				if( p3D[2] < 0 ) {

					pts2D[2*i  ] = -p3D[2] + 10;
					pts2D[2*i+1] = -p3D[2] + 10;

				} else {

					pts2D[2*i]   = p[0] - lmData[i]->coord2D[0];
					pts2D[2*i+1] = p[1] - lmData[i]->coord2D[1];

					pts2D[2*i]   *= pts2D[2*i];
					pts2D[2*i+1] *= pts2D[2*i+1];
				}
			}
		}

		Float optimizeCamera( Pose &pose, const vector<LmData *> &samples, const int maxLMTests ) {

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

			if( retValue < 0 ) return retValue;

			pose.rotation.init( camPoseLM );
			pose.translation.init( camPoseLM + 4 );

			pose.rotation.norm();
			// output is in camPoseLM
			return info[1];
		}

		void testAllPoints( vector<LmData *> &consistentCorresp, const Pose &pose, const vector<LmData *> &testPoints, const Float ErrorThreshold ) {

			consistentCorresp.clear();

			foreach( corresp, testPoints ) {

				Pt<2> p = project( pose, corresp->coord3D, *corresp->image );
				p-=corresp->coord2D;

				Float projectionError = p[0]*p[0]+p[1]*p[1];

				if( projectionError < ErrorThreshold )
					consistentCorresp.push_back(corresp);
			}
		}

		void initPose( Pose &pose, const vector<LmData *> &samples ) {

			pose.rotation.init( (rand()&255)/256., (rand()&255)/256., (rand()&255)/256., (rand()&255)/256. );
			pose.translation.init( 0.,0.,0.5 );

		}

		bool RANSAC( Pose &pose, const vector<LmData *> &cluster ) {

			vector<LmData *> samples;
			for ( int nIters = 0; nIters<MaxRANSACTests; nIters++) {

				samples.clear();
				if( !randSample( samples, cluster, NPtsAlign ) ) return false;

				initPose( pose, samples );

				int LMIterations = optimizeCamera( pose, samples, MaxLMTests );
				if( LMIterations == -1 ) continue;

				vector<LmData *> consistent;
				testAllPoints( consistent, pose, cluster, ErrorThreshold );

				if ( (int)consistent.size() > MinNPtsObject ) {
					optimizeCamera( pose, consistent, MaxLMTests );
					return true;
				}
			}
			return false;
		}

		// restore matches data to optData, in LmData structure
		void preprocessAllMatches( 	vector< vector< LmData > > &optData,
										const vector< vector< FrameData::Match > > &matches,
										const vector< SP_Image > &images ) {

			optData.resize( matches.size() );
			for(int model=0; model<(int)matches.size(); model++ )
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
				const SP_Image &image = images[ matches[model][match].imageIdx ];

				optData[model][match].image = image.get();
				optData[model][match].coord2D = matches[model][match].coord2D;
				optData[model][match].coord3D = matches[model][match].coord3D;
			}
		}

		void outputcl(  vector<LmData *> cl) {
			ofstream out;
			out.open( "cl1.txt" );
			for ( int i = 0; i < (int)cl.size(); i ++ ) {
				out << cl[i]->coord2D << "\t" << cl[i]->coord3D << "\n";
			}
			out.close();
		}

	public:

		POSE_RANSAC_LM_DIFF_REPROJECTION_CPU( int MaxRANSACTests, int MaxLMTests, int MaxObjectsPerCluster, int NPtsAlign, int MinNPtsObject, Float ErrorThreshold )
		: MaxRANSACTests(MaxRANSACTests), MaxLMTests(MaxLMTests), MaxObjectsPerCluster(MaxObjectsPerCluster), NPtsAlign(NPtsAlign),
		  MinNPtsObject(MinNPtsObject), ErrorThreshold(ErrorThreshold) {
		}

		void getConfig( map<string,string> &config ) const {

			GET_CONFIG( MaxRANSACTests );
			GET_CONFIG( MaxLMTests );
			GET_CONFIG( NPtsAlign );
			GET_CONFIG( MinNPtsObject );
			GET_CONFIG( ErrorThreshold );
		}

		void setConfig( map<string,string> &config ) {

			SET_CONFIG( MaxRANSACTests );
			SET_CONFIG( MaxLMTests );
			SET_CONFIG( NPtsAlign );
			SET_CONFIG( MinNPtsObject );
			SET_CONFIG( ErrorThreshold );
		}

		void process( FrameData &frameData ) {
//			cout << "POSE_RANSAC_LM_DIFF_REPROJECTION_CPU" << endl;
			int NObjectsCluster = MaxObjectsPerCluster;
			string DescriptorType = "SIFT";
			vector< SP_Image > &images = frameData.images;
			vector< vector< FrameData::Match > > &matches = frameData.matches;
			vector< vector< FrameData::Cluster > > &clusters = frameData.clusters;
			vector< vector< LmData > > lmData;
			preprocessAllMatches( lmData, matches, images );

			vector< pair<int,int> > tasks;
			tasks.reserve(1000);
			int model;
			for( model=0; model<(int)clusters.size(); model++ ){
				for(int cluster=0; cluster<(int)clusters[model].size(); cluster++ )
					for(int obj=0; obj<NObjectsCluster; obj++)
						tasks.push_back( make_pair(model, cluster) );
			}

			/* Tasks.size() = NObjectsCluster * SUM(clusters[i].size())
			 * clusters.size() = 10 which is the number of loaded models
			 * clusters[i].size() = cluster # for each object minimum is 0
			 * each task[i] should be index for objects and the clusters in
			 * 		each models
			 * LmData is just a clone of matches result
			 */
			// Generate task size using NObjectsCluster by cluster
			#pragma omp parallel for
			for(int task=0; task<(int)tasks.size(); task++) {
				int model=tasks[task].first;
				int cluster=tasks[task].second;
				vector<LmData *> cl;
				foreach( point, clusters[model][cluster] ){
					cl.push_back( & lmData[model][point] );
				}
				outputcl( cl );
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
				if ( found > 0 )
					#pragma omp critical(POSE)
					{
						SP_Object obj(new Object);
						frameData.objects->push_back(obj);
						obj->pose = pose;
						obj->model = (*models)[model];
					}

			}
			if( _stepName == "POSE" )
				frameData.oldObjects = *frameData.objects;
		}
	};
};
