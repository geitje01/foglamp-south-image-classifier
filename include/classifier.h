#ifndef _CLASSIFIER_H
#define _CLASSIFIER_H
/*
 * FogLAMP south service plugin
 *
 * Copyright (c) 2019 Dianomic Systems
 *
 * Released under the Apache 2.0 Licence
 *
 * Author: Amandeep Singh Arora
 */
#include <reading.h>
#include "opencv2/imgproc.hpp"

class ImageClassifier {
	public:
		ImageClassifier();
		~ImageClassifier();
		Reading		takeReading();
	void	setAssetName(const std::string& assetName)
		{
			m_asset_name = assetName;
		}

	private:
		std::string	m_asset_name;
		int identifyDigit(cv::Mat &mat);
		int processImage(cv::Mat &image);
		bool takeImage(cv::Mat& mat);
};
#endif
