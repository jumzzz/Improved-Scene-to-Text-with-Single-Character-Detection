

#include "text.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
//
#include <iostream>
#include <algorithm>
//
//
using namespace std;
using namespace cv;
using namespace cv::text;
//
Mat frame, grey, orig_grey, out_img;
Mat dst, detected_edges, src_gray;
int edgeThresh = 1;
//
bool onBrightness = true;
bool onContrast = true;
bool onCanny = true;
//
int const max_lowThreshold = 255;
int ratio = 3;
int kernel_size = 5;
//
// variables for single character

int threshval3 = 255;
RNG rng(12345);

bool sortbyXaxis(const Point & a, const Point &b);

bool sortbyYaxis(const Point & a, const Point &b);




struct OCROutInfo{

	Rect boundBox;
	String outputText;
	float confidence;

};


struct OCROutInfoGroup{
   
	Rect boundBox;
	String outputText;

};


double windowWidth;
double windowHeight;


bool isBetween(int a, int x, int b);
bool isSuperSet(Rect inside, Rect outside);
bool isSuperSetList(Rect inside, vector<Rect> list);
void performSingleCharDetection(Mat &frame, std::vector<Rect> &scBoundBoxList, int &treshval
	, vector<Rect> &boxes);
bool isThereEqualROI(Rect refROI, vector<Rect> roiList);
void removeRedundancy(vector<OCROutInfo> &ocrInfoList);
void removeRedundancyFinal(vector<OCROutInfo> &ocrInfoList);

// detection function for single character


// detection function for single character ends


//
//vector< vector<Rect> > boxesSC((int)detectionsSC.size());
//vector< vector<string> > wordsSC((int)detectionsSC.size());
//vector< vector<float> > confidencesSC((int)detectionsSC.size());



//
//
////ERStat extraction is done in parallel for different channels
class Parallel_extractCSER : public cv::ParallelLoopBody
{
private:
	vector<Mat> &channels;
	vector< vector<ERStat> > &regions;
	vector< Ptr<ERFilter> > er_filter1;
	vector< Ptr<ERFilter> > er_filter2;

public:
	Parallel_extractCSER(vector<Mat> &_channels, vector< vector<ERStat> > &_regions,
		vector<Ptr<ERFilter> >_er_filter1, vector<Ptr<ERFilter> >_er_filter2)
		: channels(_channels), regions(_regions), er_filter1(_er_filter1), er_filter2(_er_filter2){}

	virtual void operator()(const cv::Range &r) const
	{
		for (int c = r.start; c < r.end; c++)
		{
			er_filter1[c]->run(channels[c], regions[c]);
			er_filter2[c]->run(channels[c], regions[c]);
		}
	}
	Parallel_extractCSER & operator=(const Parallel_extractCSER &a);
};
//
////OCR recognition is done in parallel for different detections
template <class T>
class Parallel_OCR : public cv::ParallelLoopBody
{
private:
	vector<Mat> &detections;
	vector<string> &outputs;
	vector< vector<Rect> > &boxes;
	vector< vector<string> > &words;
	vector< vector<float> > &confidences;
	vector< Ptr<T> > &ocrs;

public:
	Parallel_OCR(vector<Mat> &_detections, vector<string> &_outputs, vector< vector<Rect> > &_boxes,
		vector< vector<string> > &_words, vector< vector<float> > &_confidences,
		vector< Ptr<T> > &_ocrs)
		: detections(_detections), outputs(_outputs), boxes(_boxes), words(_words),
		confidences(_confidences), ocrs(_ocrs)
	{}

	virtual void operator()(const cv::Range &r) const
	{
		for (int c = r.start; c < r.end; c++)
		{
			ocrs[c%ocrs.size()]->run(detections[c], outputs[c], &boxes[c], &words[c], &confidences[c], OCR_LEVEL_WORD);
		}
	}
	Parallel_OCR & operator=(const Parallel_OCR &a);
};
//
//
////Discard wrongly recognised strings
bool   isRepetitive(const string& s);
//Draw ER's in an image via floodFill
void   er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation);
//
//
//
////Perform text detection and recognition from webcam
int main(int argc, char* argv[])   /// main starts
{
	cout << endl << argv[0] << endl << endl;
	cout << "Implementation of End-to-end Scene Text Detection and Recognition using webcam." << endl << endl;
	cout << "  Usage:  " << argv[0] << " [camera_index]" << endl << endl;
	cout << "  Press 'r' to switch between MSER/CSER regions." << endl;
	cout << "  Press 'g' to switch between Horizontal and Arbitrary oriented grouping." << endl;
	cout << "  Press 'x' to disable/enable brightness/contrast trackbar." << endl;
	cout << "  Press 'c' to disable/enable canny trackbar." << endl;

	cout << "  Press 'ESC' to exit." << endl << endl;

	namedWindow("recognition", WINDOW_AUTOSIZE);
	bool downsize = false;
	int  REGION_TYPE = 1;
	int  GROUPING_ALGORITHM = 0;
	int  RECOGNITION = 0;
	char *region_types_str[2] = { const_cast<char *>("ERStats"), const_cast<char *>("MSER") };
	char *grouping_algorithms_str[2] = { const_cast<char *>("exhaustive_search"), const_cast<char *>("multioriented") };
	char *recognitions_str[2] = { const_cast<char *>("Tesseract"), const_cast<char *>("NM_chain_features + KNN") };


	vector<Mat> channels;
	vector<vector<ERStat> > regions(2); //two channels


	
	/************************************************************************/
	/*
	 1. Load the OCR Tesseracts Objects ()
	*/
	/************************************************************************/


	// Create ERFilter objects with the 1st and 2nd stage default classifiers
	// since er algorithm is not reentrant we need one filter for channel
	vector< Ptr<ERFilter> > er_filters1;
	vector< Ptr<ERFilter> > er_filters2;

	// 1a. load ocr objects for group of texts recogntions
	for (int i = 0; i < 2; i++)
	{
		Ptr<ERFilter> er_filter1 = createERFilterNM1(loadClassifierNM1("trained_classifierNM1.xml"), 8, 0.00015f, 0.13f, 0.2f, true, 0.1f);
		Ptr<ERFilter> er_filter2 = createERFilterNM2(loadClassifierNM2("trained_classifierNM2.xml"), 0.5);
		er_filters1.push_back(er_filter1);
		er_filters2.push_back(er_filter2);
	}

	//double t_r = getTickCount();

	//Initialize OCR engine for group of texts (we initialize 10 instances in order to work several recognitions in parallel)
	cout << "Initializing OCR engines for Group of Texts ..." << endl;
	int num_ocrs = 10;
	vector< Ptr<OCRTesseract> > ocrs;
	for (int o = 0; o < num_ocrs; o++)
	{
		ocrs.push_back(OCRTesseract::create((const char *)0, (const char*)0, (const char*)0, 3, 8));
	}


	//Initialize OCR engine for single character (we initialize 10 instances in order to work several recognitions in parallel)



	cout << "Initializing OCR engines for Single Character ..." << endl;
	int num_ocrs_single = 10;
	vector< Ptr<OCRTesseract> > ocrs_single;
	for (int o = 0; o < num_ocrs_single; o++)
	{
		ocrs_single.push_back(OCRTesseract::create((const char *)0, (const char*)0, (const char*)0, 3, 10));
	}




	cout << "Initializing OCR engines for OCRHMM ..." << endl;

	Mat transition_p;
	string filename = "OCRHMM_transitions_table.xml";
	FileStorage fs(filename, FileStorage::READ);
	fs["transition_probabilities"] >> transition_p;
	fs.release();
	Mat emission_p = Mat::eye(62, 62, CV_64FC1);
	string voc = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";

	vector< Ptr<OCRHMMDecoder> > decoders;
	for (int o = 0; o < num_ocrs; o++)
	{
		decoders.push_back(OCRHMMDecoder::create(loadOCRHMMClassifierNM("OCRHMM_knn_model_data.xml.gz"),
			voc, transition_p, emission_p));
	}
	cout << " Done!" << endl;



	//cout << "TIME_OCR_INITIALIZATION_ALT = "<< ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;


	// added revision for single character
	std::string trackbarName3 = "Treshold";


	// added revision for single character

	//int cam_idx = 0;
	int cam_idx = 0;
	if (argc > 1)
		cam_idx = atoi(argv[1]);

	VideoCapture cap(cam_idx);


	cap.read(frame);
	imshow("recognition", frame);

	double dWidth = cap.get(CV_CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
	double dHeight = cap.get(CV_CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

	windowWidth = dWidth;
	windowHeight = dHeight;


	cout << "Frame size : " << dWidth << " x " << dHeight << endl;
	//Create trackbar to change brightness
	int iSliderValue1 = 50;
	createTrackbar("Brightness", "recognition", &iSliderValue1, 100);

	//Create trackbar to change contrast
	int iSliderValue2 = 50;
	createTrackbar("Contrast", "recognition", &iSliderValue2, 100);

	createTrackbar(trackbarName3, "recognition", &threshval3, 150);


	if (!cap.isOpened())
	{
		cout << "ERROR: Cannot open default camera (0)." << endl;
		return -1;
	}


	while (cap.read(frame))
	{
		double t_all = (double)getTickCount();

		if (downsize)
			resize(frame, frame, Size(320, 240));


		bool bSuccess = cap.read(frame); // read a new frame from video

		if (!bSuccess) //if not success, break loop
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		int    iBrightness = iSliderValue1 - 50;        // beta
		double dContrast = iSliderValue2*2.0 / 50.0;        // alpha


		if (onBrightness && onContrast) frame.convertTo(dst, -1, dContrast, iBrightness);
		else frame.copyTo(dst);

		//if (onCanny){
		//	Mat tempGray;
		//	cvtColor(dst, tempGray, COLOR_RGB2GRAY);
		//	blur(tempGray, detected_edges, Size(3, 3));

		//	/// Canny detector
		//	Canny(detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size);

		//	/// Using Canny's output as a mask, we display our result
		//	Mat dstTemp;
		//	dstTemp = Scalar::all(0);
		//	Mat dstGray;
		//	//	cvtColor(dstTemp, dstTempColor, COLOR_GRAY2RGB);
		//	frame.copyTo(dst, detected_edges);
		//}


		//sharpen(dst, dst, iSharpening);


		//Mat tmp;
		/*	cv::GaussianBlur(dst, tmp, cv::Size(3, 3), 3);
		cv::addWeighted(dst, dContrast, tmp, iBrightness, 0, dst);*/

		/*Text Detection*/

		cvtColor(dst, grey, COLOR_RGB2GRAY);
		grey.copyTo(orig_grey);
		// Extract channels to be processed individually
		channels.clear();
		channels.push_back(grey);
		channels.push_back(255 - grey);


		regions[0].clear();
		regions[1].clear();
		//double t_d = (double)getTickCount();

		switch (REGION_TYPE)
		{
		case 0:
		{
			parallel_for_(cv::Range(0, (int)channels.size()), Parallel_extractCSER(channels, regions, er_filters1, er_filters2));
			break;
		}
		case 1:
		{
			//Extract MSER
			vector<vector<Point> > contours;
			vector<Rect> bboxes;
			Ptr<MSER> mser = MSER::create(21, (int)(0.00002*grey.cols*grey.rows), (int)(0.05*grey.cols*grey.rows), 1, 0.7);
			mser->detectRegions(grey, contours, bboxes);

			//Convert the output of MSER to suitable input for the grouping/recognition algorithms
			if (contours.size() > 0)
				MSERsToERStats(grey, contours, regions);

			break;
		}
		case 2:
		{
			break;
		}
		}
		//cout << "TIME_REGION_DETECTION_ALT = " << ((double)getTickCount() - t_d)*1000/getTickFrequency() << endl;

		// Detect character groups
		//double t_g = getTickCount();
		vector< vector<Vec2i> > nm_region_groups;
		vector<Rect> nm_boxes;
		switch (GROUPING_ALGORITHM)
		{
		case 0:
		{
			erGrouping(dst, channels, regions, nm_region_groups, nm_boxes, ERGROUPING_ORIENTATION_HORIZ);
			break;
		}
		case 1:
		{
			erGrouping(dst, channels, regions, nm_region_groups, nm_boxes, ERGROUPING_ORIENTATION_ANY, "./trained_classifier_erGrouping.xml", 0.5);
			break;
		}
		}
		//cout << "TIME_GROUPING_ALT = " << ((double)getTickCount() - t_g)*1000/getTickFrequency() << endl;




		/*Text Recognition (OCR)*/


		dst.copyTo(out_img);
		float scale_img = (float)(600.f / dst.rows);
		float scale_font = (float)(2 - scale_img) / 1.4f;
		vector<string> words_detection;
		float min_confidence1 = 0.f, min_confidence2 = 0.f;

		if (RECOGNITION == 0)
		{
			min_confidence1 = 51.f; min_confidence2 = 60.f;
		}

		vector<Mat> detections;

		//t_r = getTickCount();


		vector<OCROutInfoGroup> ocrInfoGroup;
		for (int i = 0; i < (int)nm_boxes.size(); i++)
		{
			rectangle(out_img, nm_boxes[i].tl(), nm_boxes[i].br(), Scalar(255, 255, 0), 3);


			Mat group_img = Mat::zeros(dst.rows + 2, dst.cols + 2, CV_8UC1);
			er_draw(channels, regions, nm_region_groups[i], group_img);
			group_img(nm_boxes[i]).copyTo(group_img);
			copyMakeBorder(group_img, group_img, 15, 15, 15, 15, BORDER_CONSTANT, Scalar(0));
			detections.push_back(group_img);
		}
		vector<string> outputs((int)detections.size());
		vector< vector<Rect> > boxes((int)detections.size());
		vector< vector<string> > words((int)detections.size());
		vector< vector<float> > confidences((int)detections.size());

		// parallel process detections in batches of ocrs.size() (== num_ocrs)
		for (int i = 0; i < (int)detections.size(); i = i + (int)num_ocrs)
		{
			Range r;
			if (i + (int)num_ocrs <= (int)detections.size())
				r = Range(i, i + (int)num_ocrs);
			else
				r = Range(i, (int)detections.size());

			switch (RECOGNITION)
			{
			case 0:
				parallel_for_(r, Parallel_OCR<OCRTesseract>(detections, outputs, boxes, words, confidences, ocrs));
				break;
			case 1:
				parallel_for_(r, Parallel_OCR<OCRHMMDecoder>(detections, outputs, boxes, words, confidences, decoders));
				break;
			}
		}


		for (int i = 0; i < (int)detections.size(); i++)
		{



			OCROutInfoGroup ocrInfoGroupTemp;
			ocrInfoGroupTemp.boundBox = nm_boxes[i];
			ocrInfoGroupTemp.outputText = outputs[i];

			ocrInfoGroup.push_back(ocrInfoGroupTemp);


			//outputs[i].erase(remove(outputs[i].begin(), outputs[i].end(), '\n'), outputs[i].end());
			//cout << "OCR output = \"" << outputs[i] << "\" length = " << outputs[i].size() << endl;
			if (outputs[i].size() < 3)
				continue;

			for (int j = 0; j < (int)boxes[i].size(); j++)
			{
				boxes[i][j].x += nm_boxes[i].x - 15;
				boxes[i][j].y += nm_boxes[i].y - 15;

				//cout << "  word = " << words[i][j] << "\t confidence = " << confidences[i][j] << endl;
				if ((words[i][j].size() < 2) || (confidences[i][j] < min_confidence1) ||
					((words[i][j].size() == 2) && (words[i][j][0] == words[i][j][1])) ||
					((words[i][j].size() < 4) && (confidences[i][j] < min_confidence2)) ||
					isRepetitive(words[i][j]))
					continue;
				words_detection.push_back(words[i][j]);
				rectangle(out_img, boxes[i][j].tl(), boxes[i][j].br(), Scalar(255, 0, 255), 3);
				Size word_size = getTextSize(words[i][j], FONT_HERSHEY_SIMPLEX, (double)scale_font, (int)(3 * scale_font), NULL);
				rectangle(out_img, boxes[i][j].tl() - Point(3, word_size.height + 3), boxes[i][j].tl() + Point(word_size.width, 0), Scalar(255, 0, 255), -1);
				putText(out_img, words[i][j], boxes[i][j].tl() - Point(1, 1), FONT_HERSHEY_SIMPLEX, scale_font, Scalar(255, 255, 255), (int)(3 * scale_font));




			}

		}

		//cout << "TIME_OCR_ALT = " << ((double)getTickCount() - t_r)*1000/getTickFrequency() << endl;


		t_all = ((double)getTickCount() - t_all) * 1000 / getTickFrequency();
		char buff[100];
		sprintf(buff, "%2.1f Fps. @ 640x480", (float)(1000 / t_all));
		string fps_info = buff;
		rectangle(out_img, Point(out_img.rows - 160, out_img.rows - 70), Point(out_img.cols, out_img.rows), Scalar(255, 255, 255), -1);
		putText(out_img, fps_info, Point(10, out_img.rows - 10), FONT_HERSHEY_DUPLEX, scale_font, Scalar(255, 0, 0));
		putText(out_img, region_types_str[REGION_TYPE], Point(out_img.rows - 150, out_img.rows - 50), FONT_HERSHEY_DUPLEX, scale_font, Scalar(255, 0, 0));
		putText(out_img, grouping_algorithms_str[GROUPING_ALGORITHM], Point(out_img.rows - 150, out_img.rows - 30), FONT_HERSHEY_DUPLEX, scale_font, Scalar(255, 0, 0));
		putText(out_img, recognitions_str[RECOGNITION], Point(out_img.rows - 150, out_img.rows - 10), FONT_HERSHEY_DUPLEX, scale_font, Scalar(255, 0, 0));


		//		if (boxes.size() == 0){ /// Assumes that no group of texts are detected

		//			cout << "No Group of Texts detected.." << endl;
		//cout << "Testing for single text..." << endl;
		std::vector<Rect> scBoundBoxList;


		performSingleCharDetection(frame, scBoundBoxList, threshval3, nm_boxes);

		// OCR for single character starts here

		//		/*Text Recognition (OCR)*/
		//      

		//	cout << "scBoundBoxList Size = " << scBoundBoxList.size() << endl;
		Mat out_imgSC;
		//
		frame.copyTo(out_imgSC);
		float scale_imgSC = (float)(600.f / frame.rows);
		float scale_fontSC = (float)(2 - scale_imgSC) / 1.4f;
		vector<string> words_detectionSC;
		float min_confidence1SC = 0.f, min_confidence2SC = 0.f;

		if (RECOGNITION == 0)
		{
			min_confidence1SC = 51.f; min_confidence2SC = 60.f;
		}

		vector<Mat> detectionsSC;
		//cout << "scBoundBoxList.size() = " << scBoundBoxList.size();
		//t_r = getTickCount();
		for (int i = 0; i < scBoundBoxList.size(); i++){

			Mat tempDetSC = frame(scBoundBoxList.at(i));
			Mat tempDetGraySC;
			cvtColor(tempDetSC, tempDetGraySC, COLOR_RGB2GRAY);
			detectionsSC.push_back(tempDetGraySC);
		}

		vector <OCROutInfo> ocrInfoSC;

		for (int i = 0; i < (int)scBoundBoxList.size(); i++){

			Mat group_imgSC = Mat::zeros(frame.rows + 2, frame.cols + 2, CV_8UC1);
			//rectangle(out_img, scBoundBoxList[i].tl(), scBoundBoxList[i].br(), Scalar(255, 255, 0), 3);

			vector<string> outputsSC((int)detectionsSC.size());
			vector< vector<Rect> > boxesSC((int)detectionsSC.size());
			vector< vector<string> > wordsSC((int)detectionsSC.size());
			vector< vector<float> > confidencesSC((int)detectionsSC.size());

			/*struct OCROutInfo{

				Rect boundBox;
				String outputText;
				float confidence;

				};*/




			// parallel process detections in batches of ocrs.size() (== num_ocrs)
			for (int i = 0; i < (int)detectionsSC.size(); i = i + (int)num_ocrs_single)
			{
				Range rSC;
				if (i + (int)num_ocrs_single <= (int)detectionsSC.size())
					rSC = Range(i, i + (int)num_ocrs_single);
				else
					rSC = Range(i, (int)detectionsSC.size());

				parallel_for_(rSC, Parallel_OCR<OCRTesseract>(detectionsSC, outputsSC, boxesSC, wordsSC, confidencesSC, ocrs_single));

			}


			for (int i = 0; i < (int)detectionsSC.size(); i++)
			{

				outputsSC[i].erase(remove(outputsSC[i].begin(), outputsSC[i].end(), '\n'), outputsSC[i].end());

				for (int j = 0; j < outputsSC[i].size(); j++){

					if (confidencesSC[i][j] > 90.0){
						rectangle(out_img, scBoundBoxList[i].tl(), scBoundBoxList[i].br(), Scalar(255, 255, 0), 3);

						//cout << "Char detected = " << wordsSC[i][j] << " Confidence level = " << confidencesSC[i][j] << endl;
						cvWaitKey(100); /// pagbabago ng delay

						OCROutInfo tempOCRInfo;
						tempOCRInfo.boundBox = boxesSC[i][j];
						tempOCRInfo.outputText = outputsSC[i][j];
						tempOCRInfo.confidence = confidencesSC[i][j];
						ocrInfoSC.push_back(tempOCRInfo);
					}

				}

			}

		}

		removeRedundancy(ocrInfoSC);
		//	removeRedundancy(ocrInfoGroup);
		removeRedundancyFinal(ocrInfoSC);
		//	removeRedundancyFinal(ocrInfoGroup);

		//for (int i = 0; i < ocrInfoSC.size(); i++){

		//	cout << "Output String = " << ocrInfoSC[i].outputText << endl;
		//}

		//for (int i = 0; i < ocrInfoGroup.size(); i++){

		//	cout << "Output String Group = " << ocrInfoGroup[i].outputText << endl;

		//}

		vector<Point> bbPoints;

		for (int i = 0; i < ocrInfoSC.size(); i++)
			bbPoints.push_back(Point(ocrInfoSC[i].boundBox.x,
			ocrInfoSC[i].boundBox.y));


		for (int i = 0; i < ocrInfoGroup.size(); i++)
			bbPoints.push_back(Point(ocrInfoGroup[i].boundBox.x,
			ocrInfoGroup[i].boundBox.y));


		//cout << "Unsorted: " << endl;

		//for (int i = 0; i < bbPoints.size(); i++){

		//	cout << "[" << i << "] = " << bbPoints[i] << endl;
		//}

		//cout << "Sorted: " << endl;

		sort(bbPoints.begin(), bbPoints.end(), sortbyYaxis);
		sort(bbPoints.begin(), bbPoints.end(), sortbyXaxis);

		//for (int i = 0; i < bbPoints.size(); i++){

		//	cout << "[" << i << "] = " << bbPoints[i] << endl;
		//}

		vector<String> organizedText(bbPoints.size());
		vector<OCROutInfoGroup> combinedInfo;

		combinedInfo = ocrInfoGroup;

		for (int i = 0; i < ocrInfoSC.size(); i++){
		
			OCROutInfoGroup temp;
			temp.boundBox = ocrInfoSC[i].boundBox;
			temp.outputText = ocrInfoSC[i].outputText;

			combinedInfo.push_back(temp);
		}


		for (int i = 0; i < combinedInfo.size(); i++){
		
			Point tempPoint(combinedInfo[i].boundBox.x, combinedInfo[i].boundBox.y);
			
			for (int j = 0; j < combinedInfo.size(); j++){
			     
				if (tempPoint == bbPoints[j]){
					
					organizedText[j] = combinedInfo[i].outputText;
				
				}
			
			}
		}

		//for (int i = organizedText.size() - 1; i >= 0; i--){
		//
		//	cout << organizedText[i];
		//}

		for (int i = 0; i < organizedText.size(); i++) cout << organizedText[i] << endl;


			


			//out_imgSC.copyTo(out_img);



		//}




		imshow("recognition", out_img);
		dst.release();
		//imwrite("recognition_alt.jpg", out_img);
		int key = waitKey(30);
		if (key == 27) //wait for key
		{
			cout << "esc key pressed" << endl;
			break;
		}
		else
		{
			switch (key)
			{
			case 103: //g
				GROUPING_ALGORITHM = (GROUPING_ALGORITHM + 1) % 2;
				cout << "Grouping switched to " << grouping_algorithms_str[GROUPING_ALGORITHM] << endl;
				break;

			case 114: //r
				REGION_TYPE = (REGION_TYPE + 1) % 2;
				cout << "Regions switched to " << region_types_str[REGION_TYPE] << endl;
				break;

			case (int)'x': //x

				if (onBrightness && onContrast) cout << "Brightness/Contrast trackbar is disabled." << endl;
				else cout << "Brightness/Contrast trackbar is enabled" << endl;

				onBrightness = !onBrightness;
				onContrast = !onContrast;
				break;

			case (int)'c':

				if (onCanny) cout << "Canny trackbar is disabled." << endl;
				else cout << "Canny trackbar is enabled." << endl;

				onCanny = !onCanny;
				break;
			default:
				break;

			}
		}

	}
	//
	return 0;
} /// main ends
//
bool isRepetitive(const string& s)
{
	int count = 0;
	int count2 = 0;
	int count3 = 0;
	int first = (int)s[0];
	int last = (int)s[(int)s.size() - 1];
	for (int i = 0; i<(int)s.size(); i++)
	{
		if ((s[i] == 'i') ||
			(s[i] == 'l') ||
			(s[i] == 'I'))
			count++;
		if ((int)s[i] == first)
			count2++;
		if ((int)s[i] == last)
			count3++;
	}
	if ((count > ((int)s.size() + 1) / 2) || (count2 == (int)s.size()) || (count3 > ((int)s.size() * 2) / 3))
	{
		return true;
	}


	return false;
}
//
//
void er_draw(vector<Mat> &channels, vector<vector<ERStat> > &regions, vector<Vec2i> group, Mat& segmentation)
{
	for (int r = 0; r < (int)group.size(); r++)
	{
		ERStat er = regions[group[r][0]][group[r][1]];
		if (er.parent != NULL) // deprecate the root region
		{
			int newMaskVal = 255;
			int flags = 4 + (newMaskVal << 8) + FLOODFILL_FIXED_RANGE + FLOODFILL_MASK_ONLY;
			floodFill(channels[group[r][0]], segmentation, Point(er.pixel%channels[group[r][0]].cols, er.pixel / channels[group[r][0]].cols),
				Scalar(255), 0, Scalar(er.level), Scalar(0), flags);
		}
	}
}



bool isBetween(int a, int x, int b){

	bool s1 = a <= x && x <= b;
//	bool s2 = (a + (int)0.05*a) <= x && (x <= b + (int)0.05*b);
	return s1;

}

bool intersects(Rect roi1, Rect roi2){

	int px0 = (int)roi1.x;
	int px1 = (int)roi1.x + (int)(roi1.x + roi1.width);
	int py0 = (int)roi1.y;
	int py1 = (int)roi1.y + (int)(roi1.y + roi1.height);

	int qx0 = (int)roi2.x;
	int qx1 = (int)roi2.x + (int)(roi2.x + roi2.width);
	int qy0 = (int)roi2.y;
	int qy1 = (int)roi2.y + (int)(roi2.y + roi2.height);

	bool s1 = isBetween(px0, qx0, px1);
	bool s2 = isBetween(py0, qy0, py1);

	bool a1 = s1 && s2;

	bool s3 = isBetween(px0, qx1, px1);

	bool a2 = s3 && s2;

	bool s4 = isBetween(py0, qy1, py1);

	bool a3 = s4 && s1;

	bool a4 = s3 && s4;


	bool s5 = isBetween(qx0, px0, qx1);
	bool s6 = isBetween(qy0, py0, qy1);

	bool a5 = s5 && s6;

	bool s7 = isBetween(qx0, px1, qx1);

	bool a6 = s7 && s6;

	bool s8 = isBetween(qy0, py1, qy1);

	bool a7 = s8 && s5;

	bool a8 = s8 && s7;


	return a1 || a2 || a3 || a4 || a5 || a6 || a7 || a8;
}

bool isSuperSet(Rect inside, Rect outside){

	float sf = 0.3; // scaling factor
	int px0 = (int)outside.x - (int)sf*outside.x;
	int px1 = (int)outside.x + (int)(outside.x + outside.width + sf*outside.width);
	int py0 = (int)outside.y - (int)sf*outside.y;
	int py1 = (int)outside.y + (int)(outside.y + outside.height + sf*outside.height);

	int qx0 = (int)inside.x;
	int qx1 = (int)inside.x + (int)(inside.x + inside.width);
	int qy0 = (int)inside.y;
	int qy1 = (int)inside.y + (int)(inside.y + inside.height);

	bool s1 = isBetween(px0, qx0, px1);
	bool s2 = isBetween(px0, qx1, px1);
	bool s3 = isBetween(py0, qy0, py1);
	bool s4 = isBetween(py0, qy1, py1);

	return s1 && s2 && s3 && s4;

}

bool isSuperSetList(Rect inside, vector<Rect> list){

	bool result = false;


	
		for (int j = 0; j < list.size(); j++){

			//if (isSuperSet(inside, list[j])){
			//	result = true;
			//	break;
			//}

			//float sf = 0.02; // scaling factor
			//int px0 = (int)list[j].x - (int)sf*list[j].x;
			//if (px0 < 0) px0 = 1;
			//int px1 = (int)list[j].x + (int)(list[j].x + list[j].width + sf*list[j].width);
			//if (px1 > 640) px1 = 639;
			//int py0 = (int)list[j].y - (int)sf*list[j].y;
			//if (py0 < 0) py0 = 1;
			//int py1 = (int)list[j].y + (int)(list[j].y + list[j].height + sf*list[j].height);
			//if (py1 > 480) py1 = 479;

			//Point p0(px0, py0);
			//Point p1(px1, py1);
			//Rect temp(p0, p1);
			if ((inside & list[j]) == inside){
				result = true;
				break;
			}

			if (result) break;
	
		
	
	}

	return result;


}

void removeRedundancy(vector<OCROutInfo> &ocrInfoList){

	
	for (int i = 0; i < ocrInfoList.size(); i++){
	   
		OCROutInfo ocrInfoTemp = ocrInfoList[i];

		for (int j = 0; j < ocrInfoList.size(); j++){
		
			if (j != i){
			
				if (ocrInfoList[j].confidence == ocrInfoTemp.confidence
					&& ocrInfoList[j].outputText == ocrInfoTemp.outputText){

					ocrInfoList.erase(ocrInfoList.begin() + j);
				}
			
			}
		}
	
	
	}

	

}

void removeRedundancyFinal(vector<OCROutInfo> &ocrInfoList){


	for (int i = 0; i < ocrInfoList.size(); i++){

		OCROutInfo ocrInfoTemp = ocrInfoList[i];

		for (int j = 0; j < ocrInfoList.size(); j++){

			if (j != i){

				if (ocrInfoList[j].outputText == ocrInfoTemp.outputText){

					ocrInfoList.erase(ocrInfoList.begin() + j);
				}

			}
		}


	}



}

bool sortbyXaxis(const Point & a, const Point &b)
{
	return a.x > b.x;
}
bool sortbyYaxis(const Point & a, const Point &b)
{
	return a.y > b.y;
}






void performSingleCharDetection(Mat &frame, std::vector<Rect> &scBoundBoxList, int &treshval
	,vector<Rect> &boxes){
	Mat img;


	cvtColor(frame, img, CV_RGB2GRAY);

	Mat bw = threshval3 < 128 ? (img < threshval3) : (img >
		threshval3);
	Mat labelImage(frame.size(), CV_32S);
	Mat stats, centroids;
	int nLabels = connectedComponentsWithStats(bw, labelImage,
		stats, centroids);
	std::vector<Vec3b> colors(nLabels);
	colors[0] = Vec3b(0, 0, 0);//background
	for (int label = 1; label < nLabels; ++label){
		colors[label] = Vec3b((rand() & 200), (rand() & 200),
			(rand() & 200));
	}
	Mat dst2(img.size(), CV_8UC3);
	for (int r = 0; r < dst2.rows; ++r){
		for (int c = 0; c < dst2.cols; ++c){
			int label = labelImage.at<int>(r, c);
			Vec3b &pixel = dst.at<Vec3b>(r, c);
			pixel = colors[label];
		}
	}


	scBoundBoxList.clear();
	for (int i = 1; i< nLabels; i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		float a = stats.at<int>(i, CC_STAT_AREA);

		if (a > 0.001*windowWidth*windowHeight && a < 0.3*windowWidth*windowHeight){

			Point pt1(stats.at<int>(i, CC_STAT_LEFT), stats.at<int>(i, CC_STAT_TOP));
			Point pt2(stats.at<int>(i, CC_STAT_WIDTH) + stats.at<int>(i, CC_STAT_LEFT), stats.at<int>(i, CC_STAT_TOP) + stats.at<int>(i, CC_STAT_HEIGHT));

			Rect tempBoundSC = Rect(pt1, pt2);
		

			if (!isSuperSetList(tempBoundSC, boxes)){
			     
				//cout << "tempBoundSC.x = " << tempBoundSC.x << endl;
				//cout << "tempBoundSC.y = " << tempBoundSC.y << endl;
				//cout << "tempBoundSC.x1 = " << tempBoundSC.x + tempBoundSC.width << endl;
				//cout << "tempBoundSC.y1 = " << tempBoundSC.y + tempBoundSC.height << endl;
				scBoundBoxList.push_back(tempBoundSC);
		
			}

		}



	}


}

