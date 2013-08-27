#pragma once
#include <Eigen/Dense>

namespace MopedNS {
	class VERTEX_COVER_GEO_WU :public MopedAlg {
		double param1;
		double param2;
		
		struct Node{
			Pt<3> obsPos;
			Pt<3> mdlPos;
			Pt<2> imgPos;
			int obsIdx;
			int mdlIdx;			
			friend ostream& operator<< (ostream &out, const Node &node) { 
				out << "observation: " << node.obsPos << " " << node.obsIdx << "\n ";
				out << "Model: " << node.mdlPos << " " << node.mdlIdx << "\n"; 
			}		
		};
		
		struct Vertex{
			int seedNode;
			vector<int> connectNodes;
			friend ostream& operator<< (ostream &out, const Vertex &vertex) { 
				out << "seed: " << vertex.seedNode << "\n";
				if ( vertex.connectNodes.size() == 0 )
					cout << "No connect nodes.\n";
				else {
					out << "Connect: ";
					for ( int i = 0; i < vertex.connectNodes.size(); i ++ ) { cout << vertex.connectNodes[i] << " "; } cout << "\n";
				}
			}				
		};

		bool NonZero( Eigen::MatrixXd mat ) {
			for ( int i = 0; i < mat.rows(); i ++ ) {
				for ( int j = 0; j < mat.cols(); j ++ ) {
					if ( mat(i,j) != 0.0 )
						return true;
				}
			}
			return false;
		}
	

	public:
		VERTEX_COVER_GEO_WU(double param1, double param2)
		:param1(param1), param2(param2) {
		}
		
		void getConfig( map<string, string> &config ) const {
			GET_CONFIG( param1 );
			GET_CONFIG( param2 );
		}
		
		void setConfig( map<string, string> &config ) {
			SET_CONFIG( param1 );
			SET_CONFIG( param2 );
		}
		
		void process( FrameData &frameData ) {
			vector< vector< FrameData::Match > > matches = frameData.matches;		
			vector< vector<FrameData::Cluster> >	&clusters = frameData.clusters;
			clusters.resize( models->size() );
			Image *img = frameData.images[0].get();
			cv::Mat cvImage( img->height, img->width, CV_8UC1 );
			for (int y = 0; y < img->height; y++) {
				for (int x = 0; x < img->width; x++) {
					cvImage.at<uchar>(y, x) =  (float) img->data[img->width*y+x];
				}
			}	
			
			cout << matches.size() << "*";
			#pragma omp parallel for
			for ( int model = 0; model < (int)matches.size(); model ++ ) {
				
				// erase same points i observation
				vector<Node> nodes;
				map<Pt<3>, int> usedObsPts;
				for ( int i = 0; i < (int)matches[model].size(); i ++ ) {
					Pt<3> obsPt = matches[model][i].coord3D;
					if ( !usedObsPts[obsPt]++ ) {
						usedObsPts.insert( pair<Pt<3>, int>(obsPt, i) );
						//usedObsPts[obsPt] = i;
						Node node;
						node.obsPos = matches[model][i].cloud3D;
						node.mdlPos = matches[model][i].coord3D;
						node.imgPos = matches[model][i].coord2D;
						node.obsIdx = i;
						node.mdlIdx = matches[model][i].featIdx;
						nodes.push_back( node );
					}
					//else {
					//	matches[model].erase( matches[model].begin() + i );
					//}
				}
				// build graph 
				Eigen::MatrixXd graph( nodes.size(), nodes.size() );
				for ( int i = 0; i < graph.rows(); i ++ ) {
					for ( int j = i; j < graph.cols(); j ++ ) {
						if ( i == j )
							graph(i, j) = 100.0;
						else {
							Pt<3> obsDiff = nodes[i].obsPos-nodes[j].obsPos;
							double obsDiffNorm = sqrt(	obsDiff[0]*obsDiff[0] +
							                            obsDiff[1]*obsDiff[1] +
							                            obsDiff[2]*obsDiff[2] );
							Pt<3> mdlDiff = nodes[i].mdlPos-nodes[j].mdlPos;
							double mdlDiffNorm = sqrt(	mdlDiff[0]*mdlDiff[0] +
							                            mdlDiff[1]*mdlDiff[1] +
							                            mdlDiff[2]*mdlDiff[2] );
							graph(i, j) = abs(obsDiffNorm-mdlDiffNorm);
							graph(j, i) = graph(i, j);
						}
					}
				}
				// threshold the graph
				double T = 0.15;
				Eigen::MatrixXd tGraph( nodes.size(), nodes.size() );
				for ( int i = 0; i < tGraph.rows(); i ++ ) {
					for ( int j = i; j < tGraph.cols(); j ++ ) {
						if ( graph(i, j) > T ) {
							tGraph(i, j) = 0.0;
							tGraph(j, i) = 0.0;
						}
						else {
							tGraph(i, j) = 1.0;
							tGraph(j, i) = 1.0;
						}
					}
				}
				
				vector<Vertex> vertexs;
				map<int, int> usedNodes;
				while ( NonZero( tGraph ) ) {
					vector<int> edgeNums;
					edgeNums.resize(tGraph.rows() );
					for ( int i = 0; i < tGraph.rows(); i ++ ) {
						int edgeNum = 0;
						for ( int j = 0; j < tGraph.cols(); j ++ ) {
							if ( tGraph(i, j) > 0.1 )
								edgeNum ++;
						}
						edgeNums[i] = edgeNum;					
					}
					// Calculate the max connected node
					int maxEdge = INT_MIN;
					int maxEdgeIdx;
					for ( int eIdx = 0; eIdx < edgeNums.size(); eIdx ++ ) {
						if ( edgeNums[eIdx] > maxEdge ) {
							maxEdge = edgeNums[eIdx];
							maxEdgeIdx = eIdx;
						}
					}
					Vertex vt;
					vt.seedNode = maxEdgeIdx;
					for ( int i = 0; i < tGraph.cols(); i ++ ) {
						if ( tGraph(maxEdgeIdx, i) > 0.1 ) {
							tGraph.row(i).setZero();
							tGraph.col(i).setZero();
							vt.connectNodes.push_back(i);
						}
					}
					vertexs.push_back(vt);
					tGraph.row(maxEdgeIdx).setZero();
					tGraph.col(maxEdgeIdx).setZero();
				}

				if ( vertexs.size() > 0 ) {
					if ( vertexs[0].connectNodes.size() > 8 ){
						
						clusters[model].resize(clusters[model].size()+1);
						for ( int i = 0; i < vertexs[0].connectNodes.size(); i ++ ) {
							cv::Point2f pt;
							int matchIdx = nodes[vertexs[0].connectNodes[i]].obsIdx;
							clusters[model][0].push_back( matchIdx );
							pt.x = matches[model][matchIdx].coord2D[0];
							pt.y = matches[model][matchIdx].coord2D[1];
							cv::circle( cvImage, pt, 6, cv::Scalar::all(255), 2 );					
						}
						cv::Point2f pt;					
						int matchIdx = nodes[vertexs[0].seedNode].obsIdx;
						pt.x = matches[model][matchIdx].coord2D[0];
						pt.y = matches[model][matchIdx].coord2D[1];
						cv::circle( cvImage, pt, 6, cv::Scalar::all(0), 2 );
					}
				}	
				usedObsPts.clear();
				usedNodes.clear();
			}
			cv::imshow( "Cluster", cvImage );
			cv::waitKey(10);			
					
		}		
	};
};
