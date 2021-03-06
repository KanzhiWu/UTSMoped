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

	class FEAT_SIFT_CPU:public MopedAlg {

		string ScaleOrigin;

	public:

		FEAT_SIFT_CPU( string ScaleOrigin )
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
//			cout << "FEAT_SIFT_CPU\n";
			for( int i=0; i<(int)frameData.images.size(); i++) {
				Image *img = frameData.images[i].get();

				vector<FrameData::DetectedFeature> &detectedFeatures = frameData.detectedFeatures["SIFT"];

				// Convert to a floating point image with pixels in range [0,1].
				SFImage image = CreateImage(img->height, img->width);
				for (int y = 0; y < img->height; y++)
					for (int x = 0; x < img->width; x++)
						image->pixels[y*image->stride+x] = ((float) img->data[img->width*y+x]) * 1./255.;
				// GetKeypoints uses ./libmoped/libs/libsiftfast-1.1-src/libsiftfast.cpp for detail
				Keypoint keypts = GetKeypoints(image);
				Keypoint key = keypts;
				int keycnt = 0;
				while (key) {

					detectedFeatures.resize(detectedFeatures.size()+1);

					detectedFeatures.back().imageIdx = i;

					detectedFeatures.back().descriptor.resize(128);
					for (int x=0; x<128; x++) {
						detectedFeatures.back().descriptor[x] = key->descrip[x];
					}
					detectedFeatures.back().coord2D[0] =  key->col;
					detectedFeatures.back().coord2D[1] =  key->row;

					key = key->next;
					keycnt ++;
				}

				FreeKeypoints(keypts);
				DestroyAllImages();   // we can't destroy just one!
			}
		}
	};
};
