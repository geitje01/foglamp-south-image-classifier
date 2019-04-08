/*
 * FogLAMP south plugin.
 *
 * Copyright (c) 2018 Dianomic Systems
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

#define PLUGIN_NAME "ImageClassifier"
#define CONFIG	"{\"plugin\" : { \"description\" : \"" PLUGIN_NAME " data generation plugin\", " \
			"\"type\" : \"string\", \"default\" : \"" PLUGIN_NAME "\", \"readonly\" : \"true\"}, " \
		"\"asset\" : { \"description\" : \"Asset name\", " \
			"\"type\" : \"string\", \"default\" : \"" PLUGIN_NAME "\", \"displayName\": \"Asset Name\"  }, " \
		"\"model\" : { \"description\" : \"Tensorflow lite model file path\", " \
			"\"type\" : \"string\", \"default\" : \"\", \"displayName\": \"Tensorflow lite model file path\"  } } "
		  
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
	CONFIG                    // Default configuration
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
