/*
 * MATCH_ANN_CPU.hpp
 *
 *  Created on: Jun 23, 2013
 *      Author: wu
 */

#pragma once

#include <ANN.h>

namespace MopedNS {

	class MATCH_ANN_CPU:public MopedAlg {

		static inline void norm( vector<float> &d ) {
			float norm=0;
			for (int x=0; x<(int)d.size(); x++)
			  norm += d[x]*d[x];
			norm = 1./sqrtf(norm);
			for (int x=0; x<(int)d.size(); x++)
			  d[x] *=norm;
		}

		int DescriptorSize;
		string DescriptorType;
		Float Quality;
		Float Ratio;

		bool skipCalculation;

		vector<int> correspModel;
		vector< Pt<3> * > correspFeat;

		ANNkd_tree *kdtree;

		vector< int > stepIdx;


		void Update() {

			skipCalculation = true;

			unsigned int modelsNFeats = 0;
/* The features and descriptors are all added together for matching
 * seen from the += step in foreach() */
			foreach( model, *models )
				modelsNFeats += model->IPs[DescriptorType].size();

			correspModel.resize( modelsNFeats );
			correspFeat.resize( modelsNFeats );
/* from the input parameters, DescriptorSize is likely to be 128 */
			ANNpointArray refPts = annAllocPts( modelsNFeats , DescriptorSize );

			int x=0;
			stepIdx.push_back(x);
/* Iteration for each models */
			for( int nModel = 0; nModel < (int)models->size(); nModel++ ) {

				vector<Model::IP> &IPs = (*models)[nModel]->IPs[DescriptorType];
/* Iteration for number of features for each model */
				for( int nFeat = 0; nFeat < (int)IPs.size(); nFeat++ ) {

					correspModel[x] = nModel;
					if ( x > 1 && correspModel[x-1] != correspModel[x]  ) {
						stepIdx.push_back(x);
					}
					correspFeat[x]  = &IPs[nFeat].coord3D;
					norm( IPs[nFeat].descriptor );
/* Iteration for push descriptors to each IPs */
					for( int i=0; i<DescriptorSize; i++ )
						refPts[x][i] = IPs[nFeat].descriptor[i];

					x++;
				}
			}

			cout << "modelsNFeats: " << modelsNFeats << endl;
			if( modelsNFeats > 1 ) {
/* Once we have recorded the model feature information
 * skipCalculation is set to be false and
 * we don't have to load this information again in future iteration
 * ABOVE need to be VERIFIED */
				skipCalculation = false;
				if( kdtree )
					delete kdtree;
				kdtree = new ANNkd_tree(refPts, modelsNFeats, DescriptorSize );
			}
			configUpdated = false;
		}

	public:

		MATCH_ANN_CPU( int DescriptorSize, string DescriptorType, Float Quality, Float Ratio )
		: DescriptorSize(DescriptorSize), DescriptorType(DescriptorType), Quality(Quality), Ratio(Ratio)  {

			kdtree = NULL;
			skipCalculation = true;
		}

		void getConfig( map<string,string> &config ) const {

			GET_CONFIG(DescriptorType);
			GET_CONFIG(DescriptorSize);
			GET_CONFIG(Quality);
			GET_CONFIG(Ratio);
		};

		void setConfig( map<string,string> &config ) {

			SET_CONFIG(DescriptorType);
			SET_CONFIG(DescriptorSize);
			SET_CONFIG(Quality);
			SET_CONFIG(Ratio);
		};

		void process( FrameData &frameData ) {
//			cout << "MATCH_ANN_CPU\n";
			/* This step is critical and the configUpdated is defined in util.hpp
			 * And it seems the configUpdated is true */
			if( configUpdated ) 
				Update();
			if( skipCalculation ) 
				return;			

			vector< FrameData::DetectedFeature > &corresp = frameData.detectedFeatures[DescriptorType];
			if( corresp.empty() )
				return;

			vector< vector< FrameData::Match > > &matches = frameData.matches;

			/* 1st vector represents the models' number
			 * 2nd vector represents the number for matches correpounding to each model */
			matches.resize( models->size() );
			ANNpoint pt = annAllocPt(DescriptorSize);
			/* ANNidx is typedef as int
			 * so, we get
			 * int* nx = new int[2];
			 * float* nx = new float[2] */
			ANNidxArray	nx = new ANNidx[2];
			ANNdistArray ds = new ANNdist[2];
			int k;

			/* corresp.size() is assumed to be the number of extracted features
			 * so the out iteraation is the number of whole features */
			for( int i=0; i<(int)corresp.size(); i++)  {
				norm( corresp[i].descriptor);
				for (int j = 0; j < DescriptorSize; j++)
					pt[j] = corresp[i].descriptor[j];

			    #pragma omp critical(ANN)
				(*kdtree).annkSearch(pt, 2, nx, ds, Quality);

				/* Ratio is the defined between the minimum distance
				 * and the second minimum distance
				 * if the ratio is large, it seems more matches will be detected
				 * if the ratio is small, few matches will be detected */
				if(  ds[0]/ds[1] < Ratio ) {
					/* nModel1 may output the category of objects
					 * nModel1 seems range from 0 to model.size() - 1
					 * Matches is resized as the model.size() and
					 * store the matches in frameData */
					int nModel1 = correspModel[nx[0]];
					if( matches[nModel1].capacity() < 1000 )
						matches[nModel1].reserve(1000);
					matches[nModel1].resize( matches[nModel1].size() +1 );
					int objIdx;
					int objFeatIdx;
					for ( k = 0; k < (int)stepIdx.size()-1; k ++ ) {
						if ( nx[0] < stepIdx[k+1] && nx[0] > stepIdx[k]) {
							objIdx = k;
							objFeatIdx = nx[0] - stepIdx[k];
						}
					}
					if ( nx[0] > stepIdx.back() ) {
						objIdx = stepIdx.size() - 1;
						objFeatIdx = nx[0] - stepIdx.back();
					}
					FrameData::Match &match = matches[nModel1].back();
					match.imageIdx = corresp[i].imageIdx;
					match.coord3D = *correspFeat[nx[0]];
					match.coord2D = corresp[i].coord2D;
					match.featIdx = objFeatIdx;
				}
			}
		}
	};
};
