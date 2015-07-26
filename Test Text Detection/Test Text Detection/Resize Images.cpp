///**
//  Code Written by JMS:
//  First Written: June 10, 2015
//
//   This code resizes all the image that is saved from folder D:/image_file/training_image/
//
//**/
//
//
//#include "opencv2/shape.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include <opencv2/core/utility.hpp>
//#include <iostream>
//#include <string>
//#include <vector>
//
//using namespace cv;
//using namespace std;
//int thresh = 100;
//int max_thresh = 255;
//RNG rng(12345);
//
///// Function header
//void thresh_callback(int, void*, Mat& src, Mat &cropped);
//
//
//
//// std:: indicates native C++ objects
//// cv:: indicates OpenCV objects
//
//
//// ImageInfo struct contains the ascii equivalent of
//// the input image
//
//struct ImageInfo{
//
//	int asciiCode;
//	cv::Mat sourceImgMat;
//
//};
//
//
//int main(int argc, char** argv)
//{
//	
//	std::string pathSourceStr = "D:/image_file/training_image/";
//	std::string pathDestinationStr = "D:/image_file/resized_training_image/";
//
//	if (argc < 2)
//	{
//		std::cout << "Acquiring Images from the source." << std::endl;
//	}
//	else
//	{
//		//	sscanf(argv[1], "%i", &indexQuery);
//	}
//
//	std::vector<ImageInfo> imageInfoList;
//	cv::Size imageSize(131, 159);
//
//	std::cout << "Acquisition and resizing of images begins... " << std::endl;
//
//	// acquiring images of numeric character 0-9 /w ascii code starting 
//	// with 48-57
//	for (int i = 48; i <= 57; i++){
//		
//		std::stringstream inputFileNameSstr;
//		inputFileNameSstr << pathSourceStr << i << ".jpg";
//		cv::Mat inputMat = imread(inputFileNameSstr.str(), CV_LOAD_IMAGE_ANYCOLOR);
//		cv::Mat croppedInput;
//		thresh_callback(0, 0, inputMat, croppedInput);
//		cv::Mat toBeResizedMat;
//	    resize(croppedInput, toBeResizedMat, imageSize);
//		ImageInfo tempInfo;
//		tempInfo.asciiCode = i;
//		toBeResizedMat.copyTo(tempInfo.sourceImgMat);
//		imageInfoList.push_back(tempInfo);
//	}
//
//	// acquiring images of upper-case letters A-Z /w ascii code starting 
//	// with 65-90
//	for (int i = 65; i <= 90; i++){
//
//		cout << i << endl;
//		std::stringstream inputFileNameSstr;
//		inputFileNameSstr << pathSourceStr << i << ".jpg";
//		cv::Mat inputMat = imread(inputFileNameSstr.str(), CV_LOAD_IMAGE_ANYCOLOR);
//		cv::Mat croppedInput;
//		thresh_callback(0, 0, inputMat, croppedInput);
//		cv::Mat toBeResizedMat;
//		resize(croppedInput, toBeResizedMat, imageSize);
//		ImageInfo tempInfo;
//		tempInfo.asciiCode = i;
//		toBeResizedMat.copyTo(tempInfo.sourceImgMat);
//		imageInfoList.push_back(tempInfo);
//	}
//
//	// acquiring images of lower-case letters a-z /w ascii code starting 
//	// with 97-122
//	for (int i = 97; i <= 122; i++){
//
//		cout << i << endl;
//
//		std::stringstream inputFileNameSstr;
//		inputFileNameSstr << pathSourceStr << i << ".jpg";
//		cv::Mat inputMat = imread(inputFileNameSstr.str(), CV_LOAD_IMAGE_ANYCOLOR);
//		cv::Mat croppedInput;
//		thresh_callback(0, 0, inputMat, croppedInput);
//		cv::Mat toBeResizedMat;
//		resize(croppedInput, toBeResizedMat, imageSize);
//		ImageInfo tempInfo;
//		tempInfo.asciiCode = i;
//		toBeResizedMat.copyTo(tempInfo.sourceImgMat);
//		imageInfoList.push_back(tempInfo);
//
//	}
//
//	std::cout << "Acquisition and resizing of images ends... " << std::endl;
//
//	std::cout << "Resizing begins..." << std::endl;
//
//	// writing new resized images
//	// writing images of numeric character 0-9 /w ascii code starting 
//	for (int i = 0; i < imageInfoList.size(); i++){
//	    
//		std::stringstream outputFileNameSstr;
//		outputFileNameSstr << pathDestinationStr << imageInfoList.at(i).asciiCode << ".jpg";
//		imwrite(outputFileNameSstr.str(), imageInfoList.at(i).sourceImgMat);
//
//		
//	}
//
//	//	stringstream queryName;
//	//	queryName << path << indexQuery << ".jpg";
//
//	std::cout << "Resizing done..." << std::endl;
//	
//
//	cvWaitKey();
//	return 0;
//}
//
//void thresh_callback(int, void*, Mat &src, Mat& cropped)
//{
//	Mat threshold_output;
//	Mat src_gray;
//	Mat sub_img;
//    cvtColor(src, src_gray, CV_BGR2GRAY);
//	blur(src_gray, src_gray, Size(3, 3));
//	vector<vector<Point> > contours;
//	vector<Vec4i> hierarchy;
//
//	/// Detect edges using Threshold
//	threshold(src_gray, threshold_output, thresh, 255, THRESH_BINARY);
//	/// Find contours
//	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
//
//	/// Approximate contours to polygons + get bounding rects and circles
//	vector<vector<Point> > contours_poly(contours.size());
//	vector<Rect> boundRect(contours.size());
//	vector<Point2f>center(contours.size());
//	vector<float>radius(contours.size());
//
//	for (int i = 0; i < contours.size(); i++)
//	{
//		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
//		boundRect[i] = boundingRect(Mat(contours_poly[i]));
//		minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
//	}
//
//
//	/// Draw polygonal contour + bonding rects + circles
//	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
//	for (int i = 0; i< contours.size(); i++)
//	{   
//		
//	//	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//		Scalar color = Scalar(125, 125, 200, 154);
//		//	drawContours(drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point());
//		rectangle(src, boundRect[i].tl(), boundRect[i].br(), color, 1, 8, 0);
//
//		//circle(drawing, center[i], (int)radius[i], color, 2, 8, 0);
//	}
//	int x = boundRect.at(1).x;
//	int y = boundRect.at(1).y;
//	int width = boundRect.at(1).width;
//	int height = boundRect.at(1).height;
//
//	cv::Mat imageTemp(src);
//	cv::Rect myROI(x, y, width, height);
//	cropped = imageTemp(myROI);
//	/// Show in a window
//	//namedWindow("Contours", CV_WINDOW_AUTOSIZE);
//	//imshow("Contours", src);
//	//namedWindow("Cropped Image", CV_WINDOW_AUTOSIZE);
//	//imshow("Cropped Image", cropped);
//}