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
#include "opencv2/tracking.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <math.h>
#include <unistd.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <classifier.h>
#include <reading.h>
#include <logger.h>


using namespace cv;
using namespace std;

const vector<string> labels { "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" };

int digit;
float topn_prob=0.0;

/**
 * Constructor for ImageClassifer class
 */
ImageClassifier::ImageClassifier()
{
	srand(time(0));
}

/**
 * Destructor for ImageClassifer class
 */
ImageClassifier::~ImageClassifier()
{
}

/**
 * Get top 'n' indices in output tensor that input tensor could possibly be representing
 */
template<typename T>
std::vector<int> GetTopNIndices(const T* data, int size, int n) {
    std::vector<int> topn;
    auto comp = [&](int i, int j) -> bool { return data[i] > data[j]; };
    topn_prob=0.0;
    for (int i = 0; i < size; i++) {
		//Logger::getLogger()->info("i=%d, data[i]=%f", i, data[i]);
		topn_prob=std::max(topn_prob, (float) data[i]);
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

/**
 * Get top 'n' digits the input tensor could possibly be representing
 */
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

/**
 * Pass image information into tensorflow model and find the digit it maps to
 */
int ImageClassifier::identifyDigit(Mat &mat)
{
    //Logger::getLogger()->info("get_current_dir_name()=%s", get_current_dir_name());
	//string model_file("/home/pi/dev/FogLAMP/plugins/south/ImageClassifier/digit_recognition.tflite");
	string model_file(m_tflite_model);
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
	//Logger::getLogger()->info("Recognized digit is %d or %d or %d", stoi(topn[0]), stoi(topn[1]), stoi(topn[2]));
	
	//interpreter->close();
	
	return stoi(topn[0]);
}

/**
 * Get string representation of type of cv::Mat - for debug purpose only
 */
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

/**
 * Save image as a jpeg file - for debug purpose only
 */
void saveImage(Mat &image, string filename)
{
	vector<int> compression_params;
	compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	compression_params.push_back(95);

	imwrite(filename, image, compression_params);
}

/**
 * Process image so that it can be fed into tensorflow model
 */
int ImageClassifier::processImage(Mat &image)
{
	Mat image2;
	Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);

	//image = imread(argv[1] , CV_LOAD_IMAGE_GRAYSCALE);

	if(! image.data ) {
	  Logger::getLogger()->error("Could not find image data") ;
	  return -1;
	}
	ofstream logfile;
  	logfile.open ("logs.txt");
	Mat src = image; // keep a copy of original image
	saveImage(image, "image-1.jpg");

	//Logger::getLogger()->info("Input Matrix type is %s %d x %d", type2str(image.type()).c_str(), image.rows, image.cols);

	//cout << "Input image : " << endl << image << endl;

	//namedWindow( "Display window", CV_WINDOW_NORMAL); // CV_WINDOW_AUTOSIZE

	//imshow( "original image", image );
	//waitKey(0);

	Mat gray; // = image;
	GaussianBlur(image, gray, Size(5,5), 0);

	/// Find contours   
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat canny_output;
	
	// detect edges using canny
	Canny( gray, canny_output, 100, 150, 3 );
    dilate(canny_output, canny_output, kernel);
    findContours( canny_output, contours, cv::noArray(), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	
	saveImage(canny_output, "image-1.5.jpg");

	//cout << "gray: gray.rows=" << gray.rows << ", gray.cols=" << gray.cols << ", area=" << gray.rows*gray.cols << endl;
	//Logger::getLogger()->info("gray: %d x %d, area=%d", gray.rows, gray.cols, gray.rows*gray.cols);

	Rect bounding_rect;
	int largest_area=0;
	int largest_contour_index=0;
	Scalar color( 255,255,255);  // color of the contour in the
	int num=1;
	for( int i = 0; i< contours.size(); i++ )
    {
        //  Find the area of contour
        double a=contourArea( contours[i],false);
		if (a<300 || a>gray.rows*gray.cols/2) continue;
		//cout << "Contour area=" << a << ", contours[i].size()=" << contours[i].size() << endl;
		
		bounding_rect = boundingRect(contours[i]);
		float ar = (float) bounding_rect.width / bounding_rect.height;
		if(ar < 1.0/4.0 || ar > 4.0)
		{
			//cout << "Skipping contour because AR=" << ar << ", bounding_rect.width=" << bounding_rect.width << ", bounding_rect.height=" << bounding_rect.height << endl;
			continue; // skip if aspect ratio is more than 4 or less than 1/4
		}
		
		string name("image-" + to_string(num) + "-");
		num++;
		
		RotatedRect rect = minAreaRect( Mat(contours[i]) );
		
        Mat M, rotated, cropped;
        // get angle and size from the bounding box
        float angle = rect.angle;
        Size rect_size = rect.size;
		//cout << "angle=" << angle << ", rect_size=" << rect_size << endl;
        
        if (rect.angle < -45.0) {
            angle += 90.0;
            swap(rect_size.width, rect_size.height);
        }
        // get the rotation matrix
        M = getRotationMatrix2D(rect.center, angle, 1.0);
        // perform the affine transformation
        warpAffine(gray, rotated, M, src.size(), INTER_CUBIC);
        // crop the resulting image
        getRectSubPix(rotated, rect_size, rect.center, cropped);
		saveImage(cropped, name+"2.jpg");

		bitwise_not(cropped, image);

		threshold(image, image2, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		saveImage(image2, name+"3.jpg");

		image = image2;

		int rows = image.rows;
		int cols = image.cols;

		logfile << "bounding rect: image.rows=" << image.rows << ", image.cols=" << image.cols << endl;
		
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
		logfile << "resize to rows=" << rows << ", cols=" << cols << ", factor=" << factor << endl;
		if (rows==0 || cols==0) continue;
		resize(image, image2, Size(cols,rows), 0, 0, INTER_CUBIC);
		//imshow( "20 x 20 image", image2 );
		//waitKey(0);
		//Logger::getLogger()->info("20 x 20 image: image2 is %d x %d", image2.rows, image2.cols);
		saveImage(image2, name+"5.jpg");

		// Initialize arguments for the filter
		int top = (int) (ceil((28-rows)/2.0));
		int bottom = (int) (floor((28-rows)/2.0));
		int left = (int) (ceil((28-cols)/2.0));
		int right = (int) (floor((28-cols)/2.0));

		//cout << "20 x 20 image: top=" << top << ", bottom=" << bottom << ", left=" << left << ", right=" << right << endl;
		//Logger::getLogger()->info("20 x 20 image: top=%d, bottom=%d, left=%d, right=%d", top, bottom, left, right);

		copyMakeBorder( image2, image, top, bottom, left, right, BORDER_CONSTANT, Scalar());
		logfile << "padded image 28 x 28: image.rows=" << image.rows << ", image.cols=" << image.cols << endl;
		saveImage(image, name+"6.jpg");

		//threshold(image, image2, 128, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
		//cout << "padded image 28 x 28: image2.rows=" << image2.rows << ", image2.cols=" << image2.cols << endl;
		//image = image2;
		image2 = image;

		saveImage(image2, name+"7.jpg");
		
		//imshow( "padded image 28 x 28", image2);
		//waitKey(0);

		logfile << "saved Image : " << endl << image2 << endl << endl;

		Mat image3;
		image2.convertTo(image3, CV_32F, 1.0 / 255, 0); // normalize pixel values

		//cout << "normalized matrix type is " << type2str(image3.type()) << "  " << image3.rows << " x " << image3.cols << endl;
		
		//logfile << "normalized Image : " << endl << image3 << endl << endl;

		Mat flat_image = image3.reshape(1,1); // flat 1d, (channels, rows)

		//cout << "flat_image matrix type is " << type2str(flat_image.type()) << "  " << flat_image.rows << " x " << flat_image.cols << endl;

		//logfile << "flat_image : " << endl << flat_image << endl;
		
		//cout << "flat_image.rows=" << flat_image.rows << ", flat_image.cols=" << flat_image.cols << endl;

		digit = identifyDigit(flat_image);
		Logger::getLogger()->info("digit=%d, topn_prob=%f", digit, topn_prob);
		if(topn_prob > 0.70)
		{
			logfile.close();
			return digit;
		}
		else
			digit = -1;
	}
	logfile.close();
	return digit;
}

/**
 * Take image from camera 0
 */
bool ImageClassifier::takeImage(Mat& mat)
{
	//mat = imread("digit-8.jpg" , CV_LOAD_IMAGE_GRAYSCALE);
	VideoCapture cap(0);
	if(!cap.isOpened())  // check if we succeeded
	{
		Logger::getLogger()->error("Camera 0 is not open");
        return false;
	}

	cap >> mat; // get a new image from camera

	if(!mat.data)
	  return false;

	saveImage(mat, "image-orig.jpg");

	Logger::getLogger()->info("Captured image, type=%s   %d x %d", type2str(mat.type()).c_str(), mat.rows, mat.cols);
	
	return true;
}

/**
 * Generate a reading
 */
Reading	ImageClassifier::takeReading()
{
	Mat image;
	vector<Datapoint *> vec;
	
	bool rv = takeImage(image);

	if(!rv) {
	  Logger::getLogger()->error("Could not take image using camera");
	  return Reading(m_asset_name, vec);
	}

	Mat image2;
	cv::cvtColor(image, image2, CV_BGR2GRAY);

	long digit = processImage(image2);
	//Logger::getLogger()->info("I think digit is %d, prob=%f", rv2, topn_prob);

	if (digit != -1)
	{
		DatapointValue dpv1(digit);
		vec.push_back(new Datapoint("digit", dpv1));

		float value = (int)(topn_prob * 100 + 0.5); // round topn_prob
		DatapointValue dpv2((double)value / 100);
		vec.push_back(new Datapoint("probability", dpv2));
	}

	return Reading(m_asset_name, vec);
}

