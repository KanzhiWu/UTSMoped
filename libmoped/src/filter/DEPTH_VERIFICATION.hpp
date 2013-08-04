/*
 * DEPTH_VERIFICATION.hpp
 *
 *  Created on: Jun 26, 2013
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
	class DEPTH_VERIFICATION : public MopedAlg {
		double DistDiffs;
		int MaxRANSACTests;
		int MaxLMTests;
		int MaxObjectsPerCluster;
		int NPtsAlign;
		int MinNPtsObject;
		Float ErrorThreshold;

		struct LmData {
			Image *image;
			Pt<2> coord2D;
			Pt<3> coord3D;
		};

		void initPose( Pose &pose, const vector<LmData *> &samples ) {

			pose.rotation.init( (rand()&255)/256., (rand()&255)/256., (rand()&255)/256., (rand()&255)/256. );
			pose.translation.init( 0.,0.,0.5 );

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

		bool randSample( vector<LmData *> &samples, const vector<LmData *> &cluster, unsigned int nSamples) {

			// Do not add a correspondence of the same image at the same coordinate

			// Create a vector of samples prefixed with a random int. The int has preference over the pointer when sorting the vector.
			map< pair<Image *, Pt<2> >, int > used;
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

		bool RANSAC( Pose &pose, const vector<LmData *> &cluster ) {
/*			for ( int i = 0; i < cluster.size(); i ++ ) {
				cout << cluster[i]->coord2D << "; " << cluster[i]->coord3D << endl;
			}*/
			vector<LmData *> samples;
			for ( int nIters = 0; nIters < MaxRANSACTests; nIters++) {
				samples.clear();
				randSample( samples, cluster, NPtsAlign );

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

		void outputcl(  vector<LmData *> cl) {
			ofstream out;
			out.open( "cl1.ver.txt" );
			for ( int i = 0; i < (int)cl.size(); i ++ ) {
				out << cl[i]->coord2D << "\t" << cl[i]->coord3D << "\n";
			}
			out.close();
		}

	public:
		DEPTH_VERIFICATION( double DistDiffs, int MaxRANSACTests, int MaxLMTests, int MaxObjectsPerCluster, int NPtsAlign, int MinNPtsObject, Float ErrorThreshold )
		: DistDiffs(DistDiffs), MaxRANSACTests(MaxRANSACTests), MaxLMTests(MaxLMTests), MaxObjectsPerCluster(MaxObjectsPerCluster), NPtsAlign(NPtsAlign),
		  MinNPtsObject(MinNPtsObject), ErrorThreshold(ErrorThreshold){
		}

		void getConfig( map<string, string> &config ) const {
			GET_CONFIG( DistDiffs );
			GET_CONFIG( MaxRANSACTests );
			GET_CONFIG( MaxLMTests );
			GET_CONFIG( NPtsAlign );
			GET_CONFIG( MinNPtsObject );
			GET_CONFIG( ErrorThreshold );
		}

		void setConfig( map<string, string> &config ) {
			SET_CONFIG( DistDiffs );
			SET_CONFIG( MaxRANSACTests );
			SET_CONFIG( MaxLMTests );
			SET_CONFIG( NPtsAlign );
			SET_CONFIG( MinNPtsObject );
			SET_CONFIG( ErrorThreshold );
		}

		bool compareGeo( GeoObject geo1, GeoObject geo2 ) {
			float bdist = geo2.center.euclDist(geo1.center);
			float min1, min2;
			min1 = ( geo1.height < geo1.length )? ((geo1.height<geo1.width)? geo1.height:geo1.width):((geo1.length<geo1.width)? geo1.length:geo1.width);
			min2 = ( geo2.height < geo2.length )? ((geo1.height<geo1.width)? geo1.height:geo1.width):((geo1.length<geo1.width)? geo1.length:geo1.width);
			float dist = (min1+min2)/2;
/*			if ( dist > bdist ) {
				cout << geo1.name << " & " << geo2.name << " are coincide which cover same space!\n";
			}*/

			return (dist < bdist);
		}

		void process( FrameData &frameData ) {
			cout << "DEPTH_VERIFICATION\n";
			list<SP_Object> objects = *frameData.objects;
			vector<SP_Image> &images = frameData.images;
//			const SP_Image &image = images[0];
			for ( int idx = 0; idx < (int)frameData.depthImages.size(); idx ++ ) {
				DepthImage *img = frameData.depthImages[idx].get();
				vector<GeoObject> geos;
				foreach(object, *frameData.objects){
					cout << object->model->name << ":\n";
					string type = object->model->IPs.begin()->first;
					object->flag = true;
					int featCnt = 0, errFeatCnt = 0;
					float alldiffdist = 0., alltdists = 0., alledists = 0.;
					vector< Pt<3> > pt3s;
					vector< Pt<2> > pt2s;
					for ( int ptidx = 0; ptidx < (int)object->model->IPs.begin()->second.size(); ptidx ++ ) {
						if ( object->model->IPs[type][ptidx].observeFlag == true ) {
							Pt<3> pt3 = object->model->IPs.begin()->second[ptidx].coord3D;
							Pt<3> propt3 = project3d( object->pose, pt3, *images[0] );
//							Pt<3> newpt3;
							Pt<2> pt2 = project( object->pose, pt3, *images[0] );
//							cout << pt3 << "; " << pt2 << endl;
							pt3s.push_back( pt3 );
							pt2s.push_back( pt2 );
							int pt2y = (int)pt2.p[1]; int pt2x = (int)pt2.p[0];
							uint16_t depth = img->data[pt2y*img->width+pt2x];
							if ( depth != 0 ) {
								float edist = propt3[2]; float tdist = (float)depth/10;
								float diffdist = edist - tdist;
								alldiffdist += diffdist;
								if ( abs(diffdist)/edist > 0.2 )
									errFeatCnt ++;
								else {
									alltdists += tdist;
									alledists += edist;
								}
							}
							featCnt ++;
						}
					}
					cout << featCnt << "; " << errFeatCnt << " || ";
					cout << "original pose: " << object->pose << " || ";
/*
					if ( (featCnt-errFeatCnt+.1)/(featCnt+.1) > 0.5 ) {
						float scale = alltdists/alledists;
						cout << scale << " || ";
						if ( abs(scale-1) > 0.02 ) {
							vector<LmData *> cl;
							cl.resize( (int)pt3s.size() );
							for ( int idx = 0; idx < (int)pt3s.size(); idx ++ ) {
								LmData *lmdataTmp = new LmData;
								lmdataTmp->coord2D = pt2s[idx];
								lmdataTmp->coord3D[0] = pt3s[idx][0]*scale;
								lmdataTmp->coord3D[1] = pt3s[idx][1]*scale;
								lmdataTmp->coord3D[2] = pt3s[idx][2]*scale;
								lmdataTmp->image = image.get();
								cl[idx] = lmdataTmp;
							}
							outputcl( cl );
							Pose pose;
							bool found = RANSAC( pose, cl );
							object->pose = pose;
							cout << "modified pose: " << pose << endl;
						}
						getchar();
					}
					*/
					if ( (featCnt-errFeatCnt+.0001)/(featCnt+.0001) > 0.5 ) {
						float scale = alltdists/alledists;
						if ( abs(scale-1) > 0.03 ) {
							cout << "modify models ...\n";
							for ( int i = 0; i < (int)object->model->IPs[type].size(); i++ ) {
								object->model->IPs[type][i].coord3D[0] *= scale;
								object->model->IPs[type][i].coord3D[1] *= scale;
								object->model->IPs[type][i].coord3D[2] *= scale;
							}
						}
						else
							cout << "Correct models!\n";
					}

					if ( (errFeatCnt+.0001)/(featCnt+.0001) > 0.8 ) {
						cout << "Error recognition result\n";
						object->flag = false;
						objects.remove_if(ptr_contains(object.get()));
					}
					GeoObject geoinfo = object.get()->getObjectBox( *images[0] );
					geos.push_back( geoinfo );
				}
				vector<string> colObjects;
				for ( int i = 0; i < (int)geos.size(); i ++ ) {
					for ( int j = i+1; j < (int)geos.size(); j ++  ) {
						bool colli = compareGeo( geos[i], geos[j] );
						if ( colli == false ) {
							colObjects.push_back( geos[i].name );
							colObjects.push_back( geos[j].name );
						}
					}
				}
				cout << "colObjects: " << colObjects.size() << endl;
				if ( (int)colObjects.size() != 0 ) {
					vector<string> uniColObjects;
					for ( int i = 0; i < (int)colObjects.size(); i ++ ) {
						// check whether the object is observed before
						if ( uniColObjects.size() == 0 )
							uniColObjects.push_back( colObjects[i] );
						else {
							bool found = false;
							for ( int j = 0; j < (int)uniColObjects.size(); j ++ ) {
								if ( strcmp(uniColObjects[j].c_str(), colObjects[i].c_str()) == 0 )
									found = true | found;
								else
									found = false | found;
							}
							if ( found == false )
								uniColObjects.push_back( colObjects[i] );
						}
					}
					cout << "found similar objects: ";
					for ( int i = 0; i < (int)uniColObjects.size(); i ++ )
						cout << uniColObjects[i] << " ";
					cout << endl;
					vector<Object> simiObjects;
					vector<float> simiObjectsScores;
					for ( int i = 0; i < (int)uniColObjects.size(); i ++ ) {
						foreach(object, objects) {
							string uniname = uniColObjects[i];
							string name = object.get()->model->name;
							if ( strcmp(uniname.c_str(), name.c_str()) == 0 ) {
								simiObjects.push_back( *object.get() );
								simiObjectsScores.push_back( object.get()->score );
							}
						}
					}
					for ( int i = 0; i < (int)simiObjects.size(); i ++ ) {
						// compared the score
						cout << simiObjects[i].model->name << ": " << simiObjects[i].score << " ";
					}
					cout << endl;
					float maxscore = .01;
					int maxidx;
					for ( int i = 0; i < (int)simiObjectsScores.size(); i ++ ) {
						if (simiObjectsScores[i] > maxscore) {
							maxidx = i;
							maxscore = simiObjectsScores[i];
						}
					}
					simiObjects.erase( simiObjects.begin()+maxidx );
					for ( int i = 0; i < (int)simiObjects.size(); i ++ ) {
						// compared the score
						string uniname = simiObjects[i].model->name;
						foreach( object, *frameData.objects ) {
							string name = object->model->name;
							if ( strcmp(name.c_str(), uniname.c_str()) == 0 )
								objects.remove_if(ptr_contains(object.get()));
						}
					}
					cout << endl;
					getchar();
				}
				else {
					cout << "no collision objects found.\n";
				}



			}
		}

	};
};
