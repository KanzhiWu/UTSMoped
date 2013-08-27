/*
 * MATCH_ANN_CPU.hpp
 *
 *  Created on: Jun 23, 2013
 *      Author: wu
 */

#pragma once

#include <ANN.h>

namespace MopedNS {

	class MATCH_ANN_CPU_WU:public MopedAlg {

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
			foreach( model, *models )
				modelsNFeats += model->IPs[DescriptorType].size();

			correspModel.resize( modelsNFeats );
			correspFeat.resize( modelsNFeats );
			ANNpointArray refPts = annAllocPts( modelsNFeats , DescriptorSize );

			int x=0;
			stepIdx.push_back(x);
			for( int nModel = 0; nModel < (int)models->size(); nModel++ ) {
				vector<Model::IP> &IPs = (*models)[nModel]->IPs[DescriptorType];
				for( int nFeat = 0; nFeat < (int)IPs.size(); nFeat++ ) {
					correspModel[x] = nModel;
					if ( x > 1 && correspModel[x-1] != correspModel[x]  ) {
						stepIdx.push_back(x);
					}
					correspFeat[x]  = &IPs[nFeat].coord3D;
					norm( IPs[nFeat].descriptor );
					for( int i=0; i<DescriptorSize; i++ )
						refPts[x][i] = IPs[nFeat].descriptor[i];
					x++;
				}
			}		
			
			if( modelsNFeats > 1 ) {
				skipCalculation = false;
				if( kdtree )
					delete kdtree;
				kdtree = new ANNkd_tree(refPts, modelsNFeats, DescriptorSize );
			}
			configUpdated = false;
		}

	public:

		MATCH_ANN_CPU_WU( int DescriptorSize, string DescriptorType, Float Quality, Float Ratio )
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
			if( configUpdated ) 
				Update();
			if( skipCalculation ) 
				return;			
			vector< FrameData::DetectedFeature > &corresp = frameData.detectedFeatures[DescriptorType];
			if( corresp.empty() )
				return;
			vector< vector< FrameData::Match > > &matches = frameData.matches;
			matches.resize( models->size() );
			ANNpoint pt = annAllocPt(DescriptorSize);
			ANNidxArray	nx = new ANNidx[2];
			ANNdistArray ds = new ANNdist[2];
			int k;

			Image *img = frameData.images[0].get();
			cv::Mat cvImage( img->height, img->width, CV_8UC1 );
			for (int y = 0; y < img->height; y++) 
				for (int x = 0; x < img->width; x++) 
					cvImage.at<uchar>(y, x) = (float)img->data[img->width*y+x];
			int centerx = cvImage.cols/2, centery = cvImage.rows/2;
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
					cv::Point pt;
					pt.x = corresp[i].coord2D[0];
					pt.y = corresp[i].coord2D[1];
//					if ( pt.x > centerx - 40 && pt.x < centerx + 40 && pt.y > centery - 40 && pt.y < centery + 40 ) {
						int nModel1 = correspModel[nx[0]];
						if( matches[nModel1].capacity() < 1000 )
							matches[nModel1].reserve(1000);
						matches[nModel1].resize( matches[nModel1].size() +1 );
						int objFeatIdx;
		
						objFeatIdx = nx[0] - stepIdx[nModel1];						
						FrameData::Match &match = matches[nModel1].back();
						match.imageIdx = corresp[i].imageIdx;
						match.coord3D = *correspFeat[nx[0]];
						match.coord2D = corresp[i].coord2D;
						match.dist = ds[0];

						Pt<3> tmp;
						tmp[2] = corresp[i].cloud3D[2];
						tmp[0] = tmp[2]*(match.coord2D[0]-317.81)/543.22;
						tmp[1] = tmp[2]*(match.coord2D[1]-265.91)/543.31;
						match.cloud3D = tmp;
						match.featIdx = objFeatIdx;	


//						cv::circle( cvImage, pt, 5, cv::Scalar::all(0), 2 );
//						cout << match.cloud3D << " " << match.coord3D << " " << tmp << endl;														
//					}	
							
				}
			}
//			cv::imshow( "match", cvImage );
//			cv::waitKey( 10 );
		}
	};
};
