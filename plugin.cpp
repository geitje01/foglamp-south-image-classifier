/*
 * FogLAMP south plugin.
 *
 * Copyright (c) 2019 Dianomic Systems
 *
 * Released under the Apache 2.0 Licence
 *
 * Author: Amandeep Singh Arora
 */
#include <classifier.h>
#include <plugin_api.h>
#include <string>
#include <logger.h>
#include <plugin_exception.h>
#include <config_category.h>
#include <version.h>
#include <unistd.h>

using namespace std;

#define TO_STRING(...) DEFER(TO_STRING_)(__VA_ARGS__)
#define DEFER(x) x
#define TO_STRING_(...) #__VA_ARGS__
#define QUOTE(...) TO_STRING(__VA_ARGS__)

#define PLUGIN_NAME "ImageClassifier"

const char *def_cfg = QUOTE(
	{
		"plugin" : 
		{
			"description" : "ImageClassifier data generation plugin", 
			"type" : "string",
			"default" : "ImageClassifier", 
			"readonly" : "true"
		},
		"asset" : 
		{
			"description" : "Asset name",
			"type" : "string",
			"default" : "ImageClassifier",
			"displayName": "Asset Name",
			"order" : "1"
		},
		"model" :
		{
			"description" : "Tensorflow lite model file path",
			"type" : "string",
			"default" : "/home/pi/dev/FogLAMP/plugins/south/ImageClassifier/digit_recognition.tflite",
			"displayName": "Tensorflow lite model file path",
			"order" : "2"
		},
		"minAccuracy" :
		{
			"description" : "Minimum accuracy percentage threshold to be met by TF model output",
			"type" : "float",
			"default" : "80.0",
			"displayName": "Minimum accuracy threshold",
			"order" : "3"
		}
	}
);


/**
 * The Image classifier plugin interface
 */
extern "C" {

/**
 * The plugin information structure
 */
static PLUGIN_INFORMATION info = {
	PLUGIN_NAME,              // Name
	VERSION,                  // Version
	0,    			  // Flags
	PLUGIN_TYPE_SOUTH,        // Type
	"1.0.0",                  // Interface version
	def_cfg                   // Default configuration
};

/**
 * Return the information about this plugin
 */
PLUGIN_INFORMATION *plugin_info()
{
	return &info;
}

/**
 * Initialise the plugin, called to get the plugin handle
 */
PLUGIN_HANDLE plugin_init(ConfigCategory *config)
{
ImageClassifier *classifier = new ImageClassifier();

	if (config->itemExists("asset"))
	{
		classifier->setAssetName(config->getValue("asset"));
	}
	else
	{
		classifier->setAssetName(PLUGIN_NAME);
	}

	if (config->itemExists("minAccuracy"))
	{
		classifier->setMinAccuracy(stof(config->getValue("minAccuracy")));
	}
	else
	{
		classifier->setMinAccuracy(80.5);
	}
	
	if (config->itemExists("model"))
	{
		if (access(config->getValue("model").c_str(), F_OK|R_OK) != 0)
		{
			Logger::getLogger()->error("Valid TFlite model path is mandatory");
			delete classifier;
			classifier = NULL;
		}
		classifier->setModel(config->getValue("model"));
	}
	else
	{
		Logger::getLogger()->error("Valid Tensorflow lite model path is mandatory");
		delete classifier;
		classifier = NULL;
	}
	
	return (PLUGIN_HANDLE)classifier;
}

/**
 * Start the Async handling for the plugin
 */
void plugin_start(PLUGIN_HANDLE *handle)
{
}

/**
 * Poll for a plugin reading
 */
Reading plugin_poll(PLUGIN_HANDLE *handle)
{
ImageClassifier *classifier = (ImageClassifier *)handle;

	return classifier->takeReading();
}

/**
 * Reconfigure the plugin
 */
void plugin_reconfigure(PLUGIN_HANDLE *handle, string& newConfig)
{
ConfigCategory	config("newCfg", newConfig);
ImageClassifier		*classifier = (ImageClassifier *)*handle;

	if (config.itemExists("asset"))
	{
		classifier->setAssetName(config.getValue("asset"));
	}
	if (config.itemExists("model"))
	{
		classifier->setModel(config.getValue("model"));
	}
}

/**
 * Shutdown the plugin
 */
void plugin_shutdown(PLUGIN_HANDLE *handle)
{
ImageClassifier *classifier = (ImageClassifier *)handle;

	delete classifier;
}
};
