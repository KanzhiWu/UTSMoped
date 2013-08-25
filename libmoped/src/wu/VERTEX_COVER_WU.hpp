#pragma once

namespace MopedNS {
	class VERTEX_COVER_WU :public MopedAlg {
		double param1;
		double param2;
		
		struct Node{
			Pt<3> obsPos;
			vector<Pt<3> > mdlPos;
			vector<Pt<2> > imgPos;
			int obsIdx;
			vector<int> mdlIdxs;
			
			friend ostream& operator<< (ostream &out, const Node &node) { 
				out << "observation: \n" << node.obsPos << " " << node.obsIdx << "\n ";
				out << "Model: \n"; 
				for ( int i = 0; i < node.mdlPos.size(); i ++ ) {
					out << node.mdlIdxs[i] << "  " << node.mdlPos[i] << "  " << node.imgPos[i] << "\n";
				}
				return out;
			}		
		};

	public:
		VERTEX_COVER_WU(double param1, double param2)
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
			vector< vector< FrameData::Cluster> > &clusters = frameData.clusters;
			vector< vector< FrameData::Match > > matches = frameData.matches;
			
			Image *img = frameData.images[0].get();
			cv::Mat cvImage( img->height, img->width, CV_8UC1 );
			for (int y = 0; y < img->height; y++) {
				for (int x = 0; x < img->width; x++) {
					cvImage.at<uchar>(y, x) =  (float) img->data[img->width*y+x];
				}
			}	
			
			vector<int> modelSize;
			foreach( model, *models ) 
				modelSize.push_back(model->IPs["SIFT"].size());			
					
			for ( int model = 0; model < (int)clusters.size(); model ++ ) {
				int featNum = modelSize[model];				
				for ( int cluster = 0; cluster < (int)clusters[model].size(); cluster ++ ) {
					// Unify the observation features
					list<int>::iterator it;
					int count = 0;
					vector<Node> nodes;
					map<Pt<3>, int> usedObsPts;
					vector<int> obsPts;	
					for ( it = clusters[model][cluster].begin(); it != clusters[model][cluster].end(); it ++ ) {
						
						int idx = *it;
						Pt<3> obsPt = matches[model][idx].cloud3D;
						Pt<3> mdlPt = matches[model][idx].coord3D;
						Pt<2> imgPt = matches[model][idx].coord2D;
						int mdlIdx = matches[model][idx].featIdx;
						if ( usedObsPts[obsPt] == 0 ) {
							usedObsPts[obsPt] = count;
							Node node;
							node.obsPos = obsPt;
							node.mdlPos.push_back(mdlPt);
							node.imgPos.push_back(imgPt);
							node.obsIdx = idx;
							node.mdlIdxs.push_back( mdlIdx );
							nodes.push_back( node );
							count ++;
						}
						else {
							int tmp = usedObsPts[obsPt];
							nodes[tmp].mdlPos.push_back(mdlPt);
							nodes[tmp].imgPos.push_back(imgPt);
							nodes[tmp].mdlIdxs.push_back( mdlIdx );							
						}
					}
					
					map<int, int> usedMdlPts;
					for ( int it = 0; it < nodes.size(); it ++ ) {
						for ( int i = 0; i < nodes[it].mdlIdxs.size(); i ++ ) {
							int mdlIdx = nodes[it].mdlIdxs[i];
							if ( usedMdlPts[mdlIdx] == 0 ) {
								usedMdlPts[mdlIdx] = it;
							}
							else {
								nodes.erase( nodes.begin() + it );
							}
						}
					}
					
					// resetting cluster data
					clusters[model][cluster].clear();
					for ( int it = 0; it < nodes.size(); it ++ )
						clusters[model][cluster].push_back( nodes[it].obsIdx );
//					for ( map<Pt<3>, int>::iterator it = usedObsPts.begin(); it != usedObsPts.end(); ++ it )
//						cout << it->first << " => " << it->second << '\n';		

				}
			}	
		}
	};
};
