/*
 * FILTER_PROJECTION_CPU.hpp
 *
 *  Created on: Jun 23, 2013
 *      Author: wu
 */

#pragma once
namespace MopedNS {

	class FILTER_PROJECTION_CPU :public MopedAlg {
		// 1st round: MinPoints = 5; FeatureDistance = 4096.; MinScore = 2;
		// 2nd round: MinPoints = 7; FeatureDistance = 4096.; MinScore = 3;
		int MinPoints;
		Float FeatureDistance;
		Float MinScore;

	public:
		// MinScore is optional
		FILTER_PROJECTION_CPU( int MinPoints, Float FeatureDistance )
		: MinPoints(MinPoints), FeatureDistance(FeatureDistance), MinScore(0) {
		}

		FILTER_PROJECTION_CPU( int MinPoints, Float FeatureDistance, Float MinScore )
		: MinPoints(MinPoints), FeatureDistance(FeatureDistance), MinScore(MinScore) {
		}

		void getConfig( map<string,string> &config ) const {

			GET_CONFIG( MinPoints );
			GET_CONFIG( FeatureDistance );
			GET_CONFIG( MinScore );
		}

		void setConfig( map<string,string> &config ) {

			SET_CONFIG( MinPoints );
			SET_CONFIG( FeatureDistance );
			SET_CONFIG( MinScore );
		}

		void process( FrameData &frameData ) {
//			cout << "FILTER_PROJECTION_CPU\n";
			vector< SP_Image > &images = frameData.images;
			vector< vector< FrameData::Match > > &matches = frameData.matches;


			// Sanity check (problems when GPU is not behaving)
			if (matches.size() < models->size())
				return;
			map< pair<Pt<2>, Image *>, pair< Float, Object *> > bestPoints;
			map< Object *, FrameData::Cluster > newClusters;

			for(int m=0; m<(int)models->size(); m++) {
//				bool objectFound = false;
				foreach( object, *frameData.objects) {
					string descriptorType = object->model->IPs.begin()->first;
					if( object->model->name == (*models)[m]->name ) {
//						objectFound = ( objectFound || true );
						FrameData::Cluster &newCluster = newClusters[ object.get() ];
						Float score = 0;

						// (int)matches[m].size() counts all the features matched for one given object
						// But multiple parts of one object are recognized, why the NoF shares the same value?
						for(int match=0; match<(int)matches[m].size(); match++ ) {
							Pt<2> p = project( object->pose, matches[m][match].coord3D, *images[matches[m][match].imageIdx] );
							p -= matches[m][match].coord2D;
							int featIdx = matches[m][match].featIdx;
							float projectionError = p[0]*p[0]+p[1]*p[1];
							if( projectionError < FeatureDistance ) {
								newCluster.push_back( match );
								score+=1./(projectionError + 1.);
//								cout << object->model->IPs[descriptorType].size() << ": " << featIdx << " " << 1./(projectionError + 1.) << endl;
								if ( featIdx < (int)object->model->IPs[descriptorType].size() )
									object->model->IPs[descriptorType][featIdx].scores.push_back( 1./(projectionError + 1.) );
								if ( object->model->IPs[descriptorType][featIdx].observeFlag == false ) {
									object->model->IPs[descriptorType][featIdx].observeFlag = true;
								}
							}
						}
						// So object->score counts the E-distance for all features observed
						object->score = score;
						foreach( match, newCluster ) {
							pair< Float, Object *> &point = bestPoints[make_pair(matches[m][match].coord2D, images[matches[m][match].imageIdx].get())];
							if( point.first < score ) {
								point.first = score;
								point.second = object.get();
							}
						}
					}
				}
			}
			// Now put only the best points in each cluster
			newClusters.clear();
			// Go through all the models
			for(int m = 0; m < (int)models->size(); m++) {
				for (int match = 0; match < (int) matches[m].size(); match++) {
					// Find out the object that this point belongs to, and put it in the object's cluster
					pair< Float, Object *> &point = bestPoints[make_pair(matches[m][match].coord2D, images[matches[m][match].imageIdx].get())];
					if ( point.second != NULL && point.second->model->name == (*models)[m]->name )
						newClusters[ point.second ].push_back( match );
				}
			}
			frameData.clusters.clear();
			frameData.clusters.resize( models->size() );

			// Delete the object with points less than MinPoints
			for(int m=0; m<(int)models->size(); m++) {
				eforeach( object, object_it, *frameData.objects) {
					if( object->model->name == (*models)[m]->name ) {
						if( (int)newClusters[object.get()].size() < MinPoints || object->score < MinScore) {
							object_it = frameData.objects->erase(object_it);
						}
						else {
							frameData.clusters[m].push_back( newClusters[object.get()] );
						}
					}
				}
			}
		}
	};
};
