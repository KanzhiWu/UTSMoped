/*
 * config.hpp
 *
 *  Created on: Jun 23, 2013
 *      Author: wu
 */

#pragma once

#define MAX_THREADS 4


#include <util/UTIL_UNDISTORT.hpp>

#include <feat/FEAT_SIFT_CPU.hpp>
#include <feat/FEAT_DISPLAY.hpp>

#include <match/MATCH_ANN_CPU.hpp>
#include <match/MATCH_DISPLAY.hpp>

#include <cluster/CLUSTER_MEAN_SHIFT_CPU.hpp>
#include <cluster/CLUSTER_DISPLAY.hpp>

#include <pose/POSE_RANSAC_LM_DIFF_REPROJECTION_CPU.hpp>
#include <pose/POSE_DISPLAY.hpp>

#include <filter/FILTER_PROJECTION_CPU.hpp>
#include <filter/DEPTH_VERIFICATION.hpp>
#include <STATUS_DISPLAY.hpp>
#include <GLOBAL_DISPLAY.hpp>
#include <GLOBAL_DISPLAY_WU.hpp>

#include <wu/PLANE_EXTRACTION.hpp>
#include <wu/FEAT_SIFT_CPU_WU.hpp>
#include <wu/MATCH_ANN_CPU_WU.hpp>
#include <wu/POSE_DEPTH_CPU_WU.hpp>
#include <wu/CLUSTER_DISPLAY_WU.hpp>
#include <wu/CLUSTER_MEAN_SHIFT_CPU_WU.hpp>
#include <wu/LONGUET_HIGGINS_WU.hpp>
#include <wu/VERTEX_COVER_WU.hpp>

#define DEFAULT_DISPLAY_LEVEL 1


namespace MopedNS {


	void createPipeline( MopedPipeline &pipeline ) {
		// UTS pipeline

//		pipeline.addAlg( "MODEL_PROCESS", new MODEL_PROCESS );
		pipeline.addAlg( "UNDISTORTED_IMAGE", new UTIL_UNDISTORT );		
		pipeline.addAlg( "PLANE_REMOVAL", new PLANE_EXTRACTION(0., 0., 0., 0., 0.) );
		pipeline.addAlg( "SIFT", new FEAT_SIFT_CPU_WU("-1") );
		pipeline.addAlg( "MATCH_SIFT", new MATCH_ANN_CPU_WU( 128, "SIFT", 5., 0.8) );
//		pipeline.addAlg( "MATCH_DISPLAY", new MATCH_DISPLAY(3) );
		pipeline.addAlg( "CLUSTER", new CLUSTER_MEAN_SHIFT_CPU_WU(10, 3, 20, 100) ); //10, 3, 20, 100
		pipeline.addAlg( "CLUSTER_DISPLAY", new CLUSTER_DISPLAY(3) );
		pipeline.addAlg( "VERTEX_COVER", new VERTEX_COVER_WU(0.0, 0.0) );
//		pipeline.addAlg( "LONGUET_HIGGINS", new LONGUET_HIGGINS_WU(8) );
		pipeline.addAlg( "POSE", new POSE_DEPTH_CPU_WU( 500, 1, 4, 30, 3.0 ) );
		pipeline.addAlg( "POSE_DISPLAY", new POSE_DISPLAY(3) );
//		pipeline.addAlg( "STATUS_DISPLAY", new STATUS_DISPLAY( DEFAULT_DISPLAY_LEVEL ) );
//		pipeline.addAlg( "GLOBAL_DISPLAY", new GLOBAL_DISPLAY( 2 ) );
/*					
		pipeline.addAlg( "UNDISTORTED_IMAGE", new UTIL_UNDISTORT );		
		pipeline.addAlg( "SIFT", new FEAT_SIFT_CPU("-1") );
		pipeline.addAlg( "MATCH_SIFT", new MATCH_ANN_CPU( 128, "SIFT", 5., 0.8) );
		pipeline.addAlg( "CLUSTER", new CLUSTER_MEAN_SHIFT_CPU( 200, 20, 7, 100) );
//		pipeline.addAlg( "CLUSTER_DISPLAY", new CLUSTER_DISPLAY(3) );		
		pipeline.addAlg( "POSE", new POSE_RANSAC_LM_DIFF_REPROJECTION_CPU( 600, 200, 4, 5, 6, 10) );
		pipeline.addAlg( "FILTER", new FILTER_PROJECTION_CPU( 5, 4096., 2) );
		pipeline.addAlg( "POSE2", new POSE_RANSAC_LM_DIFF_REPROJECTION_CPU( 100, 500, 4, 6, 8, 5) );
		pipeline.addAlg( "FILTER2", new FILTER_PROJECTION_CPU( 7, 4096., 5) );
		pipeline.addAlg( "POSE_DISPLAY", new POSE_DISPLAY(3) );
//		pipeline.addAlg( "DEPTH", new DEPTH_VERIFICATION(0., 100, 500, 4, 6, 8, 5) );
//		pipeline.addAlg( "POSE_DISPLAY", new POSE_DISPLAY(3) );
//		pipeline.addAlg( "GLOBAL_DISPLAY", new GLOBAL_DISPLAY( 2 ) );
//		pipeline.addAlg( "STATUS_DISPLAY", new STATUS_DISPLAY( DEFAULT_DISPLAY_LEVEL ) );
*/
					
	}
};
