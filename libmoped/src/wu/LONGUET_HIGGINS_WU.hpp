/*
 * FEAT_SIFT_CPU.hpp
 *
 *  Created on: Jun 23, 2013
 *      Author: wu
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace MopedNS {

	class LONGUET_HIGGINS_WU:public MopedAlg {

		double Sigma;
		struct lhmatch{
			int matchidx;
			int modelidx;
			double gij;			
		};
		
		void SparseOutput( Eigen::MatrixXd mat ) {
			for ( int i = 0; i < mat.rows(); i ++ ) {
				for ( int j = 0; j < mat.cols(); j ++ ) {
					if ( mat(i, j) != 0 )
						cout << mat(i, j) << " ";
				}
				cout << endl;
			}
		}
		
	public:

		LONGUET_HIGGINS_WU( double Sigma )
		: Sigma(Sigma) {
		}

		void getConfig( map<string,string> &config ) const {
			GET_CONFIG( Sigma );
		}

		void setConfig( map<string,string> &config ) {
			SET_CONFIG( Sigma );
		}

		void process( FrameData &frameData ) {
			vector< vector< FrameData::Cluster> > &clusters = frameData.clusters;
			vector< vector< FrameData::Match > > matches = frameData.matches;
			
			vector<int> modelSize;
			foreach( model, *models ) 
				modelSize.push_back(model->IPs["SIFT"].size());
			
			// get the center for each cluster
			vector<vector<Pt<3> > > clusterCenters;
			clusterCenters.resize(clusters.size() );
			for ( int model = 0; model < (int)clusters.size(); model ++ ) {
				for ( int cluster = 0; cluster < (int)clusters[model].size(); cluster ++ ) {
					list<int>::iterator it;
					Pt<3> sum;
					for ( it = clusters[model][cluster].begin(); it != clusters[model][cluster].end(); it ++ ) {
						int idx = *it;
						sum +=  matches[model][idx].cloud3D;
					}
					Pt<3> ave = sum/clusters[model][cluster].size();
					clusterCenters[model].push_back( ave );
				}
			}
			

			for ( int model = 0; model < (int)clusters.size(); model ++ ) {
				int featNum = modelSize[model];				
				for ( int cluster = 0; cluster < (int)clusters[model].size(); cluster ++ ) {
					list<int>::iterator it;
					int count = 0;
					vector<vector<lhmatch> > Gvec;
					map<Pt<3>, int> usedObsPts;
					Pt<3> center = clusterCenters[model][cluster];
					vector<int> obsPts;

					for ( it = clusters[model][cluster].begin(); it != clusters[model][cluster].end(); it ++ ) {
						int idx = *it;
						Pt<3> obserPt = matches[model][idx].cloud3D;
						if ( !usedObsPts[obserPt]++ ) {
							usedObsPts.insert(pair<Pt<3>, int>(obserPt, count));
							Gvec.resize(Gvec.size()+1);
							Gvec.back().resize( featNum );
							int matchIdx = matches[model][idx].featIdx;
							for ( int vec = 0; vec < Gvec.back().size(); vec ++ ) {
								if ( vec == matchIdx ) {
									Pt<3> obserpt = matches[model][idx].cloud3D - center;
									Pt<3> modelpt = matches[model][idx].coord3D;
									double dist = obserpt.euclDist(modelpt);
									lhmatch tmp;
									tmp.matchidx = idx;
									tmp.modelidx = vec;
									tmp.gij = exp(-dist*dist/(2*Sigma*Sigma));
									Gvec.back()[vec] = tmp;
								}
								else {
									lhmatch tmp;
									tmp.matchidx = idx;
									tmp.modelidx = vec;
									tmp.gij = 0.0;
									Gvec.back()[vec] = tmp;
								}
							}		
							count ++;
						}
						else {
							int matchIdx = matches[model][idx].featIdx;						
							Pt<3> obserpt = matches[model][idx].cloud3D - center;
							Pt<3> modelpt = matches[model][idx].coord3D;
							double dist = obserpt.euclDist(modelpt);
							lhmatch tmp;
							tmp.matchidx = idx;
							tmp.modelidx = matchIdx;
							tmp.gij = exp(-dist*dist/(2*Sigma*Sigma));
							Gvec[count-1][matchIdx] = tmp;							
						}
					}
					
					
					// tranform from vector to eigen::mat
					Eigen::MatrixXd Gmat(Gvec.size(), Gvec[0].size());
					Eigen::MatrixXd Geigen(Gvec.size(), Gvec[0].size());
					for ( int matrow = 0; matrow < Gmat.rows(); matrow ++ ) {
						for ( int matcol = 0; matcol < Gmat.cols(); matcol ++ ) {
							Gmat(matrow, matcol) = Gvec[matrow][matcol].gij;
							if ( matcol == matrow )
								Geigen(matrow, matcol) = 1.0;
							else
								Geigen(matrow, matcol) = 0.0;
						}
					}
					Eigen::MatrixXd svdu, svdv;
					Eigen::JacobiSVD<Eigen::MatrixXd> svd(Gmat, Eigen::ComputeThinU | Eigen::ComputeThinV);
					cout << "Its singular values are:" << endl << svd.singularValues().size() << endl << svd.singularValues(); getchar();
					svdu = svd.matrixU();
					svdv = svd.matrixV();
//					SparseOutput( svdu ); getchar();
//					SparseOutput( svdv ); getchar();
					Eigen::MatrixXd Pmat_, Pmat;
					Pmat_ = svdu*Geigen;
					Pmat = Pmat_*svdv;
					SparseOutput( Pmat ); getchar();
					// find the largest value in cols and rows
					for ( int matrow = 0; matrow < Pmat.rows(); matrow ++ ) {
						// find the largest value in each row
						double max = -INT_MAX; int maxidx; 
						for ( int matcol = 0; matcol < Pmat.cols(); matcol ++ ) {
							if ( Pmat(matrow, matcol) > max ) {
								max = Pmat( matrow, matcol );
								maxidx = matcol;
							}
						}
						// check the value in corresponding column
						bool large = true;
						for ( int rowidx = 0; rowidx < Pmat.rows(); rowidx ++ ) {
							if ( Pmat(rowidx, maxidx) > max )
								large = false;
						}
						// maximum value in both direction
						if ( large == true ) {
							cout << Gvec[matrow][maxidx].matchidx << " " << maxidx << " " << Gvec[matrow][maxidx].modelidx << endl;
						}
					}
				}	
			}
		}
	};
};
