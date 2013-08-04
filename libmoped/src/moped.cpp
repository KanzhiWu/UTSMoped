/*
 * moped.cpp
 *
 *  Created on: Jun 23, 2013
 *      Author: wu
 */


#include <moped.hpp>
#include <util.hpp>

#include <config.hpp>

#include <cstdio>
#include <iostream>



using namespace MopedNS;
using namespace std;

struct Moped::MopedPimpl {
	// Command pipeline and models
	MopedPipeline pipeline;
	vector<SP_Model> models;

	// Load the config.hpp data and get the pipeline structure
	MopedPimpl() {
		createPipeline( pipeline );
		map<string,string> config = getConfig();
		setConfig( config );
		modelsUpdated();
	}

	// Get the loaded pipeline to algorithms
	map<string,string> getConfig() {
		map<string,string> config;
		list<MopedAlg *> algs = pipeline.getAlgs();
		foreach( alg, algs )
			alg->getConfig( config );
		return config;
	}

	// Set the listed algorithms
	void setConfig( map<string,string> &config) {
		list<MopedAlg *> algs=pipeline.getAlgs();
		foreach( alg, algs )
			alg->setConfig( config );
	}

	//
	void modelsUpdated() {

		list<MopedAlg *> algs=pipeline.getAlgs( true );
		foreach( alg, algs )
			alg->modelsUpdated( models );
	}

	// Load models from .xml file
	string addModel( sXML &sxml ) {
		
		SP_Model m(new Model);

		m->name = sxml["name"];
		

		
		m->boundingBox[0].init( 10E10, 10E10, 10E10 );
		m->boundingBox[1].init(-10E10,-10E10,-10E10 );

		sXML *points = NULL;
		foreach( pts, sxml.children)
			if( pts.name == "Points" )
				points = &pts;

		if( points == NULL ) return "";

		foreach( pt, points->children ) {

			Model::IP ip;
			ip.observeFlag = false;
			std::istringstream iss( pt["p3d"] );
			iss >> ip.coord3D;
			m->boundingBox[0].min( ip.coord3D );
			m->boundingBox[1].max( ip.coord3D );

			std::istringstream jss( pt["desc"] );
			Float f;
			while( jss >> f ) ip.descriptor.push_back(f);

			m->IPs[ pt["desc_type"] ].push_back(ip);
		}

		addModel( m );

		return m->name;
	}

	void modelToPCL ( sXML &sxml ){
		SP_Model m(new Model);
		m->name = sxml["name"];
		m->boundingBox[0].init( 10E10, 10E10, 10E10 );
		m->boundingBox[1].init(-10E10,-10E10,-10E10 );

		sXML *points = NULL;
		foreach( pts, sxml.children)
			if( pts.name == "Points" )
				points = &pts;

		foreach( pt, points->children ) {
			Model::IP ip;
			ip.observeFlag = false;
			std::istringstream iss( pt["p3d"] );
			iss >> ip.coord3D;
			m->boundingBox[0].min( ip.coord3D );
			m->boundingBox[1].max( ip.coord3D );

			std::istringstream jss( pt["desc"] );
			Float f;
			while( jss >> f )
				ip.descriptor.push_back(f);
			m->IPs[ pt["desc_type"] ].push_back(ip);
		}
		addModel( m );
	}

	// Add recognized models
	void addModel( SP_Model &model ) {

		int found = false;
		foreach( m, models )
			if( (found = (m->name == model->name ) ) )
				m = model;

		if( !found )
			models.push_back(model);

		modelsUpdated();
	}


	void removeModel( const string &name ) {

		eforeach( model, model_it, models )
			if( model->name == name )
				model_it = models.erase(model_it);

		modelsUpdated();
	}

	const vector<SP_Model> &getModels() const  {

		return models;
	}


	int processImages( vector<SP_Image> &images, vector<SP_DepthImage> &depthImages, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, list<SP_Object> &objects  ) {

		foreach( image, images )
			image->TM.init( image->cameraPose );

		objects.clear();

		// output image

		FrameData frameData;

		frameData.objects = &objects;
		frameData.images = images;
		frameData.depthImages = depthImages;
		frameData.cloudPclPtr = cloud;

		list<MopedAlg *> algs=pipeline.getAlgs( true );

		// Initiallize all clusters observe flag to be false

		struct timespec tStep, tEnd;
		foreach( alg, algs ) {
			clock_gettime(CLOCK_REALTIME, &tStep);
			alg->process( frameData );
			clock_gettime(CLOCK_REALTIME, &tEnd);
			Float tstep = ( (tEnd.tv_sec -  tStep.tv_sec)*1000000000LL + tEnd.tv_nsec -  tStep.tv_nsec )/1000000000.;
			frameData.times[alg->_stepName] = tstep;
		}

		int observeCnt;
		int scoreCnt;
		float scores, score;

		string DescriptorType = "SIFT";
		foreach ( model, models ) {
			observeCnt = 0;
			scoreCnt = 0;
			scores = 0.0;
			for ( int featIdx = 0; featIdx < (int)model->IPs[DescriptorType].size(); featIdx ++ ) {
				if ( model->IPs[DescriptorType][featIdx].observeFlag == true )
					observeCnt ++;
				if ( (int)model->IPs[DescriptorType][featIdx].scores.size() != 0 ) {
					score = 0.0;
					for ( 	int scoreIdx = 0;
							scoreIdx < (int)model->IPs[DescriptorType][featIdx].scores.size();
							scoreIdx ++ ) {
						if ( model->IPs[DescriptorType][featIdx].scores[scoreIdx] > score )
							score = model->IPs[DescriptorType][featIdx].scores[scoreIdx];
					}
					scoreCnt ++;
					scores += score;
				}
			}
//			cout << model->name << ": "
			//	 << "observed feature#: " << observeCnt << ", "
			//	 << "scores: " << scores << ", "
			//	 << "total features#: " << model->IPs[DescriptorType].size() << ", "
			//	 << "average score: " << scores/observeCnt << ", "
			//	 << "feature converage: " << (double)observeCnt/(double)model->IPs[DescriptorType].size()
			//	 << endl;
		}

		return objects.size();
	}

	vector<std::tr1::shared_ptr<sXML> > createPlanarModelsFromImages( vector<MopedNS::SP_Image> &images, Float scale ) {

		foreach( image, images )
			image->TM.init( image->cameraPose );

		list<SP_Object> objects;

		FrameData frameData;
		frameData.objects = &objects;
		frameData.images = images;

		list<MopedAlg *> algs=pipeline.getAlgs( true );
		foreach( alg, algs )
			alg->process( frameData );

		vector<std::tr1::shared_ptr<sXML> > xmls;
		foreach( image, images ) {
			xmls.push_back( std::tr1::shared_ptr<sXML>(new sXML) );
			xmls.back()->name="Model";
			(*xmls.back())["name"]=image->name;
			xmls.back()->children.resize(1);
			xmls.back()->children[0].name="Points";
		}

		foreach( vd, frameData.detectedFeatures ) {
			foreach( d, vd.second) {

				sXML point;
				point.name = "Point";
				point["p3d"] = toString(d.coord2D[0]*scale)+" "+toString(d.coord2D[1]*scale)+" 0";
				point["desc_type"] = vd.first;

				for(int x=0; x<(int)d.descriptor.size(); x++)
					point["desc"] += string(x?" ":"") + toString(d.descriptor[x]);

				xmls[d.imageIdx]->children[0].children.push_back(point);
			}
		}

		return xmls;
	}

};

Moped::Moped() { mopedPimpl = new MopedPimpl(); }
Moped::~Moped() { delete mopedPimpl; }

map<string,string> Moped::getConfig() {
	return mopedPimpl->getConfig(); }

void Moped::setConfig( map<string,string> &config) {
	       mopedPimpl->setConfig( config ); }

string Moped::addModel( sXML &sxml ) {
	return mopedPimpl->addModel( sxml ); }

void Moped::modelToPCL( sXML &sxml ) {
	mopedPimpl->modelToPCL( sxml );}

void Moped::addModel( SP_Model &model ) {
	       mopedPimpl->addModel( model ); }

void Moped::removeModel( const string &name ) {
		   mopedPimpl->removeModel( name ); }

const vector<SP_Model> &Moped::getModels() const {
	return mopedPimpl->getModels(); }

int Moped::processImages( vector<SP_Image> &images, vector<SP_DepthImage> &depthImages, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, list<SP_Object> &objects ) {
	return mopedPimpl->processImages( images, depthImages, cloud, objects ); }

vector<std::tr1::shared_ptr<sXML> > Moped::createPlanarModelsFromImages( vector<MopedNS::SP_Image> &images, float scale ) {
	return mopedPimpl->createPlanarModelsFromImages( images, scale ); }



