/*
 * FogLAMP south service plugin
 *
 * Copyright (c) 2019 Dianomic Systems
 *
 * Released under the Apache 2.0 Licence
 *
 * Author: Amandeep Singh Arora
 */
 
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <math.h>
#include <unistd.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <classifier.h>
#include <reading.h>
#include <logger.h>

#define IMAGE_SIZE 28

using namespace cv;
using namespace std;

const vector<string> labels { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

/**
 * Constructor for the random "sensor"
 */
ImageClassifier::ImageClassifier()
{
	srand(time(0));
}

/**
 * Destructor for the random "sensor"
 */
ImageClassifier::~ImageClassifier()
{
}

template<typename T>
std::vector<int> GetTopNIndices(const T* data, int size, int n) {
    std::vector<int> topn;
    auto comp = [&](int i, int j) -> bool { return data[i] > data[j]; };
    for (int i = 0; i < size; i++) {
		//cout << "i=" << i << ", data[i]=" << data[i] << endl;
        topn.push_back(i);
        std::push_heap(topn.begin(), topn.end(), comp);
        if (topn.size() > n) {
            std::pop_heap(topn.begin(), topn.end(), comp);
            topn.pop_back();
        }
    }
    std::sort_heap(topn.begin(), topn.end(), comp);
    return topn;
}

std::vector<std::string> GetTopN(TfLiteTensor* output, int n) {
    std::vector<int> topn;
    switch (output->type) {
        case kTfLiteFloat32:
            topn = GetTopNIndices<float>(output->data.f, output->dims->data[1], n);
            break;
        case kTfLiteUInt8:
            topn = GetTopNIndices<uint8_t>(output->data.uint8, output->dims->data[1], n);
            break;
        default:
            Logger::getLogger()->error("Unable to process output tensor flow data");
    }
    std::vector<std::string> topn_labels(topn.size());
    for (int i = 0; i < topn.size(); i++) {
        if (topn[i] < labels.size()) topn_labels[i] = labels[topn[i]];
    }
    return topn_labels;
}

int ImageClassifier::identifyDigit(Mat &mat)
{
    Logger::getLogger()->info("get_current_dir_name()=%s", get_current_dir_name());
	string model_file("/home/pi/dev/FogLAMP/plugins/south/ImageClassifier/digit_recognition.tflite");
	// Load model.
    auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
    if (!model) {
        Logger::getLogger()->error("Failed to load tensorflow lite model: %s", model_file.c_str());
        return -1;
    }
    // Create interpreter.
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        Logger::getLogger()->error("Failed to create TFlite interpreter");
        return -1;
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Logger::getLogger()->error("Failed to allocate tensors");
        return -1;
    }
    interpreter->SetNumThreads(1);
    // Get input / output.
    const int input = interpreter->inputs()[0];
    TfLiteTensor* input_tensor = interpreter->tensor(input);
    TfLiteIntArray* input_dims = input_tensor->dims;
    const uint32_t width = input_dims->data[1], height = input_dims->data[2];
	//cout << "input_dims->data[0]=" << input_dims->data[0] << ", input_dims->data[1]=" << input_dims->data[1] << endl;
	//cout << "input_tensor->type=" << input_tensor->type << ", kTfLiteFloat32=" << kTfLiteFloat32 << ", kTfLiteUInt8=" << kTfLiteUInt8 << endl;
	//cout << "mat.rows=" << mat.rows << ", mat.cols=" << mat.cols << endl;
	for (int i = 0; i < mat.cols; i++)
	{
		input_tensor->data.f[i] = mat.at<float>(0,i);
	}
    const int output = interpreter->outputs()[0];
    TfLiteTensor* output_tensor = interpreter->tensor(output);
	TfLiteIntArray* output_dims = output_tensor->dims;
	//cout << "output_dims->data[0]=" << output_dims->data[0] << ", output_dims->data[1]=" << output_dims->data[1] << endl;
	//cout << "output_tensor->type=" << output_tensor->type << ", kTfLiteFloat32=" << kTfLiteFloat32 << ", kTfLiteUInt8=" << kTfLiteUInt8 << endl;

	const auto start = std::chrono::high_resolution_clock::now();
    const TfLiteStatus rc = interpreter->Invoke();
    const std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;

	const auto topn = GetTopN(output_tensor, 3);

	//cout << "I think the digit is " << topn[0] << " or " << topn[1] << " or " << topn[2] << endl;
	Logger::getLogger()->info("Recognized digit is %d or %d or %d", stoi(topn[0]), stoi(topn[1]), stoi(topn[2]));
	
	//interpreter->close();
	
	return stoi(topn[0]);
}

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

void saveImage(Mat &image, string filename)
{
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);

	imwrite(filename, image, compression_params);
}

int ImageClassifier::processImage(Mat &image)
{
	Mat image2;
	Size size(IMAGE_SIZE,IMAGE_SIZE);

	//image = imread(argv[1] , CV_LOAD_IMAGE_GRAYSCALE);

	if(! image.data ) {
	  Logger::getLogger()->error("Could not find image data") ;
	  return -1;
	}
	saveImage(image, "image-1.jpg");

	Logger::getLogger()->info("Input Matrix type is %s %d x %d", type2str(image.type()).c_str(), image.rows, image.cols);

	//cout << "Input image : " << endl << image << endl;

	//namedWindow( "Display window", CV_WINDOW_NORMAL); // CV_WINDOW_AUTOSIZE

	//imshow( "original image", image );
	//waitKey(0);
	
	resize(image, image2, size, 0, 0, CV_INTER_LINEAR);
	saveImage(image2, "image-2.jpg");

	bitwise_not(image2, image);
	image2 = image;
	saveImage(image, "image-3.jpg");

	double thresh = threshold(image2, image, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	saveImage(image, "image-4.jpg");

	Mat mThreshold;
	threshold(image, mThreshold, 254, 255, CV_THRESH_BINARY_INV);
	
	Mat Points;
	findNonZero(mThreshold, Points);
	Rect Min_Rect = boundingRect(Points);

	image2 = image(Min_Rect);
	//cout << "bounding rect: image2.rows=" << image2.rows << ", image2.cols=" << image2.cols << endl;
	image = image2;
	rectangle(image, Min_Rect.tl(), Min_Rect.br(), Scalar(0,255,0), 0);
	//imshow( "bounded image", image );
	//waitKey(0);
	saveImage(image, "image-5.jpg");

	int rows = image.rows;
	int cols = image.cols;

	//cout << "bounding rect: image.rows=" << image.rows << ", image.cols=" << image.cols << endl;
	
	double factor;
	if (rows > cols)
	{
		factor = 20.0/rows;
		rows = 20;
		cols = int(round(cols*factor));
	}
	else
	{
		factor = 20.0/cols;
		cols = 20;
		rows = int(round(rows*factor));
	}
	resize(image, image2, Size(rows,cols), 0, 0, CV_INTER_LINEAR);
	//imshow( "20 x 20 image", image2 );
	//waitKey(0);
	//cout << "20 x 20 image: image2.rows=" << image2.rows << ", image2.cols=" << image2.cols << endl;

	// Initialize arguments for the filter
	int top = (int) (ceil((28-cols)/2.0));
	int bottom = (int) (floor((28-cols)/2.0));
	int left = (int) (ceil((28-rows)/2.0));
	int right = (int) (floor((28-rows)/2.0));

	//cout << "20 x 20 image: top=" << top << ", bottom=" << bottom << ", left=" << left << ", right=" << right << endl;

	copyMakeBorder( image2, image, top, bottom, left, right, BORDER_CONSTANT, Scalar());
	//cout << "padded image 28 x 28: image.rows=" << image.rows << ", image.cols=" << image.cols << endl;
	saveImage(image, "image-6.jpg");

	threshold(image, image2, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
	//cout << "padded image 28 x 28: image2.rows=" << image2.rows << ", image2.cols=" << image2.cols << endl;
	image = image2;

	saveImage(image2, "image-7.jpg");
	
	//imshow( "padded image 28 x 28", image2);
	//waitKey(0);

	//cout << "saved Image : " << endl << image2 << endl;

	Mat image3;
	image2.convertTo(image3, CV_32F, 1.0 / 255, 0); // normalize pixel values

	//cout << "normalized matrix type is " << type2str(image3.type()) << "  " << image3.rows << " x " << image3.cols << endl;
	
	//cout << "normalized Image : " << endl << image3 << endl;

	Mat flat_image = image3.reshape(1,1); // flat 1d, (channels, rows)

	//cout << "flat_image matrix type is " << type2str(flat_image.type()) << "  " << flat_image.rows << " x " << flat_image.cols << endl;

	//cout << "flat_image : " << endl << flat_image << endl;
	
	//cout << "flat_image.rows=" << flat_image.rows << ", flat_image.cols=" << flat_image.cols << endl;

	return (identifyDigit(flat_image));
}

bool ImageClassifier::takeImage(Mat& mat)
{
	//mat = imread("digit-8.jpg" , CV_LOAD_IMAGE_GRAYSCALE);
	VideoCapture cap(0);
	PRINT_FUNC;
	if(!cap.isOpened())  // check if we succeeded
	{
		Logger::getLogger()->error("Camera 0 is not open");
        return false;
	}
	PRINT_FUNC;

	cap >> mat; // get a new image from camera
	PRINT_FUNC;

	if(!mat.data)
	  return false;
	PRINT_FUNC;

	saveImage(mat, "image-orig.jpg");

	Logger::getLogger()->info("Captured image, type=%s   %d x %d", type2str(mat.type()).c_str(), mat.rows, mat.cols);
	
	return true;
}

/**
 * Take a reading from the random "sensor"
 */
Reading	ImageClassifier::takeReading()
{
	Mat image;
	bool rv = takeImage(image);

	PRINT_FUNC;
	if(!rv) {
	  Logger::getLogger()->error("Could not take image using camera");
	  DatapointValue value((long) -1);
	  return Reading(m_asset_name, new Datapoint("digit", value));
	}

	Mat image2;
	cv::cvtColor(image, image2, CV_BGR2GRAY);

	PRINT_FUNC;
	long rv2 = processImage(image2);
	Logger::getLogger()->info("I think digit is %d", rv2);
	
	PRINT_FUNC;
	DatapointValue value(rv2);
	return Reading(m_asset_name, new Datapoint("digit", value));
}
