//#include "opencv2/shape.hpp"
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//#include <opencv2/core/utility.hpp>
//#include <iostream>
//#include <string>
//#include "text.hpp"
//
//using namespace cv;
//using namespace std;
//int thresh = 100;
//int max_thresh = 255;
//RNG rng(12345);
//
//
//
//void thresh_callback(int, void*, Mat &src, Mat& cropped);
//
//
//static void help()
//{
//	printf("\n"
//		"This program demonstrates a method for shape comparisson based on Shape Context\n"
//		"You should run the program providing a number between 1 and 20 for selecting an image in the folder ../data/shape_sample.\n"
//		"Call\n"
//		"./shape_example [number between 1 and 20]\n\n");
//}
//
//struct ImageInfo{
//
//	int asciiCode;
//	cv::Mat sourceImgMat;
//
//};
//
////Calculate edit distance netween two words
//size_t edit_distance(const std::string& A, const std::string& B);
//size_t min(size_t x, size_t y, size_t z);
//bool   isRepetitive(const std::string& s);
//bool   sort_by_lenght(const std::string &a, const std::string &b);
////Draw ER's in an image via floodFill
//void   er_draw(std::vector<cv::Mat> &channels, 
//	std::vector<std::vector<cv::text::ERStat> > &regions, 
//	std::vector<cv::Vec2i> group, cv::Mat& segmentation);
//cv::Mat performERFiltering(cv::Mat &image);
//
//
//
//static std::vector<cv::Point> simpleContour(const cv::Mat& currentQuery, int n = 300)
//{
//	std::vector<std::vector<cv::Point> > _contoursQuery;
//	std::vector <cv::Point> contoursQuery;
//
//
//	
//	findContours(currentQuery, _contoursQuery, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
//	for (size_t border = 0; border<_contoursQuery.size(); border++)
//	{
//		for (size_t p = 0; p<_contoursQuery[border].size(); p++)
//		{
//			contoursQuery.push_back(_contoursQuery[border][p]);
//		}
//	}
//	// In case actual number of points is less than n
//	int dummy = 0;
//	for (int add = (int)contoursQuery.size() - 1; add<n; add++)
//	{
//		contoursQuery.push_back(contoursQuery[dummy++]); //adding dummy values
//	}
//	// Uniformly sampling
//	std::random_shuffle(contoursQuery.begin(), contoursQuery.end());
//	std::vector<cv::Point> cont;
//	
//	for (int i = 0; i<n; i++)
//	{
//		cont.push_back(contoursQuery[i]);
//	}
//	return cont;
//}
//
//
//int main(int argc, char** argv){
//
//	std::string inputFileStr = "D:/image_file/9.jpg";
//	std::string pathSourceStr = "D:/image_file/resized_training_image/";
//
//	int bestMatch = 0;
//	float bestDis = FLT_MAX;
//	int index = 0;
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
////	cv::Size imageSize(30, 40);
//
//	std::cout << "Acquisition and resizing of images begins... " << std::endl;
//
//
//	std::cout << "Acquisition and resizing of images ends... " << std::endl;
//
//	std::cout << "Resizing begins..." << std::endl;
//
//
//	cv::Mat inputMat = cv::imread(inputFileStr, CV_LOAD_IMAGE_UNCHANGED);
//	cv::Mat croppedInput;
//	thresh_callback(0, 0, inputMat, croppedInput);
//
//
//
//	//	cv::Mat inputFilteredMat = performERFiltering(inputMat);
//	cv::Mat inputScaledMat;
//	cv::Size scalingSize(131, 159);
//
//	resize(croppedInput, inputScaledMat, scalingSize);
//	cv::imwrite("D:/image_file/temporary.jpg", inputScaledMat);
//	cv::Mat imagenew = imread("D:/image_file/temporary.jpg", cv::IMREAD_GRAYSCALE);
//	cv::Mat imageFiltered = performERFiltering(imagenew);
//	cv::imwrite("D:/image_file/temporaryfiltered.jpg", imageFiltered);
//	cv::Mat imagenew2 = imread("D:/image_file/temporaryfiltered.jpg", 0);
//
//	std::vector<cv::Point> inputCountour;
////	inputCountour = simpleContour(imagenew2);
//	inputCountour = simpleContour(imagenew);
//
//	cv::Ptr <cv::ShapeContextDistanceExtractor> mysc = cv::createShapeContextDistanceExtractor();
//	cv::moveWindow("TEST", 0, 0);
//
//	// acquiring images of numeric character 0-9 /w ascii code starting 
//	// with 48-57
//	for (int i = 48; i <= 57; i++){
//
//		std::stringstream iiname;
//		iiname << pathSourceStr << i << ".jpg";
//		std::cout << "name: " << iiname.str() << std::endl;
//		std::vector<cv::Point> imageCountour;
//		cv::Mat tempimg = cv::imread(iiname.str(), 0);
//		imageCountour = simpleContour(tempimg);
//		float dis = mysc->computeDistance(inputCountour, imageCountour);
//		std::cout << "Distance at : " << i << ".jpg " << dis << std::endl;
//
//		imshow("TEST", tempimg);
//		if (dis<bestDis)
//		{
//			index = i;
//			bestMatch = i;
//			bestDis = dis;
//		}
//	}
//
//	// acquiring images of upper-case letters A-Z /w ascii code starting 
//	// with 65-90
//	for (int i = 65; i <= 90; i++){
//
//		std::stringstream iiname;
//		iiname << pathSourceStr << i << ".jpg";
//		std::cout << "name: " << iiname.str() << std::endl;
//		std::vector<cv::Point> imageCountour;
//		cv::Mat tempimg = cv::imread(iiname.str(), 0);
//		imageCountour = simpleContour(tempimg);
//		float dis = mysc->computeDistance(inputCountour, imageCountour);
//		std::cout << "Distance at : " << i << ".jpg " << dis << std::endl;
//
//		imshow("TEST", tempimg);
//		if (dis<bestDis)
//		{
//			index = i;
//			bestMatch = i;
//			bestDis = dis;
//		}
//	}
//
//
//
//	// acquiring images of lower-case letters a-z /w ascii code starting 
//	// with 97-122
//	for (int i = 97; i <= 122; i++){
//
//		std::stringstream iiname;
//		iiname << pathSourceStr << i << ".jpg";
//		std::cout << "name: " << iiname.str() << std::endl;
//		std::vector<cv::Point> imageCountour;
//		cv::Mat tempimg = cv::imread(iiname.str(), 0);
//		imageCountour = simpleContour(tempimg);
//		float dis = mysc->computeDistance(inputCountour, imageCountour);
//		std::cout << "Distance at : " << i << ".jpg " << dis << std::endl;
//
//		imshow("TEST", tempimg);
//		if (dis<bestDis)
//		{
//			index = i;
//			bestMatch = i;
//			bestDis = dis;
//		}
//	}
//
//	// writing new resized images
//	// writing images of numeric character 0-9 /w ascii code starting 
//	
//
//	cv::destroyWindow("TEST");
//
//	std::stringstream bestname;
//	bestname << pathSourceStr << index << ".jpg";
//	cv::imshow("Input", inputMat);
//	cvNamedWindow("Best Match", CV_WINDOW_AUTOSIZE);
//		cv::Mat best = cv::imread(bestname.str(), 0);
//	cv::imshow("Best Match", best );
//
//
//
//	//	stringstream queryName;
//	//	queryName << path << indexQuery << ".jpg";
//
//	std::cout << "Resizing done..." << std::endl;
//
//
//	cvWaitKey();
//	return 0;
//	cvWaitKey();
//	return 0;
//}
//
//
//size_t min(size_t x, size_t y, size_t z)
//{
//	return x < y ? std::min(x, z) : std::min(y, z);
//}
//
//size_t edit_distance(const std::string& A, const std::string& B)
//{
//	size_t NA = A.size();
//	size_t NB = B.size();
//
//	std::vector< std::vector<size_t> > M(NA + 1, std::vector<size_t>(NB + 1));
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
//bool isRepetitive(const std::string& s)
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
//void er_draw(std::vector<cv::Mat> &channels, 
//	std::vector<std::vector<cv::text::ERStat> > &regions, std::vector<cv::Vec2i> group, 
//	cv::Mat& segmentation)
//{
//	for (int r = 0; r<(int)group.size(); r++)
//	{
//		cv::text::ERStat er = regions[group[r][0]][group[r][1]];
//		if (er.parent != NULL) // deprecate the root region
//		{
//			int newMaskVal = 255;
//			int flags = 4 + (newMaskVal << 8) + cv::FLOODFILL_FIXED_RANGE +
//				cv::FLOODFILL_MASK_ONLY;
//			floodFill(channels[group[r][0]], segmentation, cv::Point(er.pixel%channels[group[r][0]].cols, er.pixel / channels[group[r][0]].cols),
//				cv::Scalar(255), 0, cv::Scalar(er.level), cv::Scalar(0), flags);
//		}
//	}
//}
//
//bool   sort_by_lenght(const std::string &a, const std::string &b){ return (a.size()>b.size()); }
//
//cv::Mat performERFiltering(cv::Mat &image){
//
//	std::vector<cv::Mat> channels;
//
//	cv::Mat grey;
//	image.copyTo(grey);
//
//	// Notice here we are only using grey channel, see textdetection.cpp for example with more channels
//	channels.push_back(grey);
//	channels.push_back(255 - grey);
//
////	double t_d = (double)getTickCount();
//	// Create ERFilter objects with the 1st and 2nd stage default classifiers
//	cv::Ptr<cv::text::ERFilter> er_filter1 = cv::text::createERFilterNM1(
//		cv::text::loadClassifierNM1("trained_classifierNM1.xml"), 100, 0.005f, 0.80f, 0.1f, true, 0.1f);
//	cv::Ptr<cv::text::ERFilter> er_filter2 = cv::text::createERFilterNM2(
//		cv::text::loadClassifierNM2("trained_classifierNM2.xml"), 0.5);
//
//	std::vector<std::vector<cv::text::ERStat> > regions(channels.size());
//	// Apply the default cascade classifier to each independent channel (could be done in parallel)
//	for (int c = 0; c<(int)channels.size(); c++)
//	{
//		er_filter1->run(channels[c], regions[c]);
//		er_filter2->run(channels[c], regions[c]);
//	}
//
//	cv::Mat out_img_decomposition = cv::Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
//	std::vector<cv::Vec2i> tmp_group;
//
//	for (int i = 0; i<(int)regions.size(); i++)
//	{
//		for (int j = 0; j<(int)regions[i].size(); j++)
//		{
//			tmp_group.push_back(cv::Vec2i(i, j));
//		}
//		cv::Mat tmp = cv::Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);
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
///** @function thresh_callback */
//void thresh_callback(int, void*, Mat &src, Mat& cropped)
//{
//	Mat threshold_output;
//	Mat src_gray;
//	Mat sub_img;
//	cvtColor(src, src_gray, CV_BGR2GRAY);
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
//		//	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
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
//
//
/////// pasting the next code here