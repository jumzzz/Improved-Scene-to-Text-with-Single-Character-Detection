/////**
////  Code Written by JMS:
////  First Written: June 10, 2015
////
////   This code uses ER Filter all the image that is saved from folder D:/image_file/resized_training_image/
////
////**/
////
//
//#include "text.hpp"
//#include "opencv2/core/utility.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//
//#include <iostream>
//
//using namespace std;
//using namespace cv;
//using namespace cv::text;
//
//
//
//
//struct ImageInfo{
//
//	int asciiCode;
//	cv::Mat sourceImgMat;
//
//};
//
//
//
////Calculate edit distance netween two words
//size_t edit_distance(const string& A, const string& B);
//size_t min(size_t x, size_t y, size_t z);
//bool   isRepetitive(const string& s);
//bool   sort_by_lenght(const string &a, const string &b);
////Draw ER's in an image via floodFill
//void   er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);
//cv::Mat performERFiltering(cv::Mat &image);
//
//
////Perform text detection and recognition and evaluate results using edit distance
//int main(int argc, char* argv[])
//{
//
//	std::vector<ImageInfo> imageInputList;
//    
//	std::string pathSourceStr = "D:/image_file/resized_training_image/";
//	std::string pathDestinationStr = "D:/image_file/er_filtered_image/";
//	
//	
//	std::cout << "Acquisition of Images Begins... " << std::endl;
//
//	// acquiring images of numeric character 0-9 /w ascii code starting 
//	// with 48-57
//	for (int i = 48; i <= 57; i++){
//
//		std::stringstream inputFileNameSstr;
//		inputFileNameSstr << pathSourceStr << i << ".jpg";
//		cv::Mat inputMat = imread(inputFileNameSstr.str(), cv::IMREAD_GRAYSCALE);
//		ImageInfo tempInfo;
//		tempInfo.asciiCode = i;
//
//
//		tempInfo.asciiCode = i;
//		inputMat.copyTo(tempInfo.sourceImgMat);
//	
//		imageInputList.push_back(tempInfo);
//	}
//
//	// acquiring images of upper-case letters A-Z /w ascii code starting 
//	// with 65-90
//	for (int i = 65; i <= 90; i++){
//
//		std::stringstream inputFileNameSstr;
//		inputFileNameSstr << pathSourceStr << i << ".jpg";
//		cv::Mat inputMat = imread(inputFileNameSstr.str(), cv::IMREAD_GRAYSCALE);
//		ImageInfo tempInfo;
//		tempInfo.asciiCode = i;
//
//		tempInfo.asciiCode = i;
//		inputMat.copyTo(tempInfo.sourceImgMat);
//
//		imageInputList.push_back(tempInfo);
//
//
//	}
//
//	// acquiring images of lower-case letters a-z /w ascii code starting 
//	// with 97-122
//	for (int i = 97; i <= 122; i++){
//
//		std::stringstream inputFileNameSstr;
//		inputFileNameSstr << pathSourceStr << i << ".jpg";
//		cv::Mat inputMat = imread(inputFileNameSstr.str(), cv::IMREAD_GRAYSCALE);
//		ImageInfo tempInfo;
//		tempInfo.asciiCode = i;
//
//
//		tempInfo.asciiCode = i;
//		inputMat.copyTo(tempInfo.sourceImgMat);
//
//		imageInputList.push_back(tempInfo);
//
//
//	}
//
//	std::cout << "Acquisition images ends... " << std::endl;
//
//	std::cout << "ER Filtering Begins..." << std::endl;
//
//	// writing new resized images
//	// writing images of numeric character 0-9 /w ascii code starting 
//	for (int ii = 0; ii < imageInputList.size(); ii++){
//		std::cout << "Num of iterations =  " << ii << std::endl;
//		std::stringstream outputFileNameSstr;
//		outputFileNameSstr << pathDestinationStr << imageInputList.at(ii).asciiCode << ".jpg";
//		
//		cv::Mat tempImgFiltered = performERFiltering(imageInputList.at(ii).sourceImgMat);
//		imwrite(outputFileNameSstr.str(), tempImgFiltered);
//
//		tempImgFiltered.release();
//	}
//
//	//	stringstream queryName;
//	//	queryName << path << indexQuery << ".jpg";
//
//	std::cout << "ER Filtering Done.." << std::endl;
//
//
//
//	cv::waitKey();
//	return 0;
//}
//
//
//size_t min(size_t x, size_t y, size_t z)
//{
//	return x < y ? min(x, z) : min(y, z);
//}
//
//size_t edit_distance(const string& A, const string& B)
//{
//	size_t NA = A.size();
//	size_t NB = B.size();
//
//	vector< vector<size_t> > M(NA + 1, vector<size_t>(NB + 1));
//
//	for (size_t a = 0; a <= NA; ++a)
//		M[a][0] = a;
//
//	for (size_t b = 0; b <= NB; ++b)
//		M[0][b] = b;
//
//	for (size_t a = 1; a <= NA; ++a)
//		for (size_t b = 1; b <= NB; ++b)
//		{
//		size_t x = M[a - 1][b] + 1;
//		size_t y = M[a][b - 1] + 1;
//		size_t z = M[a - 1][b - 1] + (A[a - 1] == B[b - 1] ? 0 : 1);
//		M[a][b] = min(x, y, z);
//		}
//
//	return M[A.size()][B.size()];
//}
//
//bool isRepetitive(const string& s)
//{
//	int count = 0;
//	for (int i = 0; i<(int)s.size(); i++)
//	{
//		if ((s[i] == 'i') ||
//			(s[i] == 'l') ||
//			(s[i] == 'I'))
//			count++;
//	}
//	if (count >((int)s.size() + 1) / 2)
//	{
//		return true;
//	}
//	return false;
//}
//
//
//void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
//{
//	for (int r = 0; r<(int)group.size(); r++)
//	{
//		ERStat er = regions[group[r][0]][group[r][1]];
//		if (er.parent != NULL) // deprecate the root region
//		{
//			int newMaskVal = 255;
//			int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
//			floodFill(channels[group[r][0]], segmentation, Point(er.pixel%channels[group[r][0]].cols, er.pixel / channels[group[r][0]].cols),
//				Scalar(255), 0, Scalar(er.level), Scalar(0), flags);
//		}
//	}
//}
//
//bool   sort_by_lenght(const string &a, const string &b){ return (a.size()>b.size()); }
//
//cv::Mat performERFiltering(cv::Mat &image){
//
//	std::vector<Mat> channels;
//
//	cv::Mat grey;
//	image.copyTo(grey);
//
//	// Notice here we are only using grey channel, see textdetection.cpp for example with more channels
//	channels.push_back(grey);
//	channels.push_back(255 - grey);
//
//	double t_d = (double)getTickCount();
//	// Create ERFilter objects with the 1st and 2nd stage default classifiers
//	cv::Ptr<cv::text::ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"), 100, 0.005f, 0.80f, 0.1f, true, 0.1f);
//	cv::Ptr<cv::text::ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"), 0.5);
//
//	std::vector<std::vector<cv::text::ERStat> > regions(channels.size());
//	// Apply the default cascade classifier to each independent channel (could be done in parallel)
//	for (int c = 0; c<(int)channels.size(); c++)
//	{
//		er_filter1->run(channels[c], regions[c]);
//		er_filter2->run(channels[c], regions[c]);
//	}
//
//	cv::Mat out_img_decomposition = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
//	std::vector<cv::Vec2i> tmp_group;
//
//	for (int i = 0; i<(int)regions.size(); i++)
//	{
//		for (int j = 0; j<(int)regions[i].size(); j++)
//		{
//			tmp_group.push_back(Vec2i(i, j));
//		}
//		cv::Mat tmp = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
//
//		er_draw(channels, regions, tmp_group, tmp);
//		if (i > 0)
//			tmp = tmp / 2;
//		out_img_decomposition = out_img_decomposition | tmp;
//		tmp_group.clear();
//	}
//
//	er_filter1.release();
//	er_filter2.release();
//
//	return out_img_decomposition;
//
//}
//
//
//
/////// pasting the next code here