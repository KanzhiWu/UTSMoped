/*
 * WU_TEST.hpp
 *
 *  Created on: Jul 1, 2013
 *      Author: wu
 */

#pragma once

namespace MopedNS {

	class WU_TEST:public MopedAlg {

		int test;
	public:

		WU_TEST( int test )
		: test(test) { }

		void getConfig( map<string,string> &config ) const {

			GET_CONFIG( test );
		}

		void setConfig( map<string,string> &config ) {

			SET_CONFIG( test );
		}

		void process(FrameData &frameData) {
			if( !test ) return;

		}
	};
};
