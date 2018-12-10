#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <iostream>

// set parameters
const int fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
const double fps = 30;

const double beta = 0.8;

//-------------------
// init camera
//-------------------
void initCamera(cv::Mat &scene, cv::VideoCapture &cap)
{
	int cap_width = 640, cap_height = 480;

	cap = cv::VideoCapture(0);
	if (!cap.isOpened())
	{
		std::cout << "Connect the camera." << std::endl;
		exit(0);
	}
	cap.set(cv::CAP_PROP_FRAME_WIDTH, cap_width);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, cap_height);
	std::cout << "Width & Height: (" << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "," << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << ")" << std::endl;
	//cap >> scene;
}


// Detect Features & Output Images
// target : target image (color), scene : scene image (color)
// t_gray : target image (grayscale), s_gray : scene image (grayscale)
// dst : output image
// num : select method (0:SURF / 1:BRISK / 2:ORB / 3:A-KAZE)
void features(cv::Mat &target, cv::Mat &scene, cv::Mat &t_gray, cv::Mat &s_gray, cv::Mat &dst, int num)
{
	// init (For time Calculation)
	double f = 1000.0 / cv::getTickFrequency();

	int64 time_s; // start of time
	double time_detect; // end of detection time
	double time_match; // end of matching time

	// Detect Features & Calculate Descriptor
	cv::Ptr<cv::Feature2D> feature;
	cv::Ptr<cv::DescriptorMatcher> matcher;
	std::stringstream ss;


	// 0:SURF / 1:BRISK / 2:ORB / 3:A-KAZE
	switch (num)
	{
	case 0:
		feature = cv::xfeatures2d::SURF::create(400);
		//feature = cv::AKAZE::create();
		matcher = cv::DescriptorMatcher::create("BruteForce");
		ss << "SURF";
		break;
	case 1:
		feature = cv::BRISK::create();
		//feature = cv::AKAZE::create();
		matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
		ss << "BRISK";
		break;
	case 2:
		feature = cv::ORB::create();
		//feature = cv::AKAZE::create();
		matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
		ss << "ORB";
		break;
	case 3:
		feature = cv::AKAZE::create();
		matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
		ss << "A-KAZE";
		break;
	default:
		break;
	}


	//******************************************
	// Key points detection & Feature description
	//******************************************
	std::vector<cv::KeyPoint> kpts1, kpts2;
	cv::Mat desc1, desc2;

	// Start time measurement <Detect Time>
	time_s = cv::getTickCount();

	feature->detectAndCompute(t_gray, cv::noArray(), kpts1, desc1);
	feature->detectAndCompute(s_gray, cv::noArray(), kpts2, desc2);

	// End time measurement <Detect Time>
	time_detect = (cv::getTickCount() - time_s)*f;

	// If no key points are detected
	if (desc1.rows == 0)
		return;

	//*********************
	// Matching feartures 
	//*********************
	//int matchtype = feature->defaultNorm(); // NORM_HAMMINGなど
	
	//cv::BFMatcher matcher(matchtype);
	std::vector< std::vector<cv::DMatch> > knn_matches;

	//std::string mtype = feature->
	
	// Start time measurement <Matching Time>
	time_s = cv::getTickCount();

	matcher->knnMatch(desc1, desc2, knn_matches, 3);

	// End time measurement <Matching Time>
	time_match = (cv::getTickCount() - time_s)*f;


	//****************************************
	// Narrow down the corresponding points
	//****************************************
	const double match_par = .6; //threshold Value (for corresponding points)
	std::vector<cv::DMatch> good_matches;

	std::vector<cv::Point2f> match_point1;
	std::vector<cv::Point2f> match_point2;

	for (size_t i = 0; i < knn_matches.size(); ++i) {
		double dist1 = knn_matches[i][0].distance;
		double dist2 = knn_matches[i][1].distance;

		// Extract only points away from the second candidate
		// (Leave only good points)
		if (dist1 <= dist2 * match_par) {
			good_matches.push_back(knn_matches[i][0]);
			match_point1.push_back(kpts1[knn_matches[i][0].queryIdx].pt);
			match_point2.push_back(kpts2[knn_matches[i][0].trainIdx].pt);
		}
	}

	// Estimate Homograpoh matrix
	cv::Mat masks;
	cv::Mat H = cv::findHomography(match_point1, match_point2, masks, cv::RANSAC, 3);

	// Extract only corresponding points used in RANSAC
	std::vector<cv::DMatch> inlinerMatches;
	for (size_t i = 0; i < masks.rows; ++i) {
		uchar *inliner = masks.ptr<uchar>(i);
		if (inliner[0] == 1) {
			inlinerMatches.push_back(good_matches[i]);
		}
	}

	if (!H.empty()) {
		// Draw correspondig points only
		cv::drawMatches(target, kpts1, scene, kpts2, good_matches, dst);

		// Draw correspondig points of inliner only
		cv::drawMatches(target, kpts1, scene, kpts2, inlinerMatches, dst);

		// Get a corner from the target object image
		// (The target object is "detected")
		std::vector<cv::Point2f> obj_corners(4);
		obj_corners[0] = cv::Point2f(0, 0); obj_corners[1] = cv::Point2f(target.cols, 0);
		obj_corners[2] = cv::Point2f(target.cols, target.rows); obj_corners[3] = cv::Point2f(0, target.rows);

		// Estimate the projection to the scene
		std::vector<cv::Point2f> scene_corners(4);
		perspectiveTransform(obj_corners, scene_corners, H);

		// Connect corners by a line (The target object drawn in the scene)
		line(dst, scene_corners[0] + cv::Point2f(target.cols, 0), scene_corners[1] + cv::Point2f(target.cols, 0), cv::Scalar(0, 255, 0), 4);
		line(dst, scene_corners[1] + cv::Point2f(target.cols, 0), scene_corners[2] + cv::Point2f(target.cols, 0), cv::Scalar(0, 255, 0), 4);
		line(dst, scene_corners[2] + cv::Point2f(target.cols, 0), scene_corners[3] + cv::Point2f(target.cols, 0), cv::Scalar(0, 255, 0), 4);
		line(dst, scene_corners[3] + cv::Point2f(target.cols, 0), scene_corners[0] + cv::Point2f(target.cols, 0), cv::Scalar(0, 255, 0), 4);
	}



	putText(dst, ss.str(), cv::Point(10, target.rows + 40), cv::FONT_HERSHEY_SIMPLEX, beta, cv::Scalar(255, 255, 255), 1, CV_AA);
	ss.str("");
	ss << "Detection & Description";
	putText(dst, ss.str(), cv::Point(10, target.rows + 70), cv::FONT_HERSHEY_SIMPLEX, beta, cv::Scalar(0, 255, 255), 1, CV_AA);
	ss.str("");
	ss << "Time: " << time_detect << " [ms]";
	putText(dst, ss.str(), cv::Point(10, target.rows + 90), cv::FONT_HERSHEY_SIMPLEX, beta, cv::Scalar(0, 255, 255), 1, CV_AA);

	ss.str("");
	ss << "--Matches--";
	putText(dst, ss.str(), cv::Point(10, target.rows + 120), cv::FONT_HERSHEY_SIMPLEX, beta, cv::Scalar(255, 255, 0), 1, CV_AA);
	ss.str("");
	ss << "Good Matches: " << good_matches.size();
	putText(dst, ss.str(), cv::Point(10, target.rows + 140), cv::FONT_HERSHEY_SIMPLEX, beta, cv::Scalar(255, 255, 0), 1, CV_AA);

	ss.str("");
	ss << "Inlier: " << inlinerMatches.size();
	putText(dst, ss.str(), cv::Point(10, target.rows + 160), cv::FONT_HERSHEY_SIMPLEX, beta, cv::Scalar(255, 255, 0), 1, CV_AA);

	ss.str("");
	ss << "Inlier ratio: " << inlinerMatches.size()*1.0 / good_matches.size();
	putText(dst, ss.str(), cv::Point(10, target.rows + 200), cv::FONT_HERSHEY_SIMPLEX, beta, cv::Scalar(255, 255, 0), 1, CV_AA);

}


//-------
// run
// flgCamera : true = use camera(or video) / false = one image(not use camera)
// target : target image, scene : scene image
// cap : input camera object, writer : output video object, filename : output filename 
//-------
void run(bool &flgCamera, cv::Mat &target, cv::Mat &scene, cv::VideoCapture &cap, cv::VideoWriter &writer, std::string &filename)
{
	cv::Mat t_gray;	//target image (grayscale)
	cv::Mat s_gray;	//scene image (grayscalse)

	bool flgVideoWrite = false;
	int max_width = target.cols, max_height = target.rows;

	while (1)
	{
		if (flgCamera)
		{
			cap >> scene;
		}
		if (scene.empty())
		{
			std::cout << "There is no image." << std::endl;
			break;
		}

		// Convert "color image" to "grayscale image"
		if (target.channels() != 1)
			cv::cvtColor(target, t_gray, cv::COLOR_BGR2GRAY);
		else
			t_gray = target;

		if (scene.channels() != 1)
			cv::cvtColor(scene, s_gray, cv::COLOR_BGR2GRAY);
		else
			s_gray = scene;

		// Histogram normalization
		cv::normalize(t_gray, t_gray, 0, 255, cv::NORM_MINMAX);
		cv::normalize(s_gray, s_gray, 0, 255, cv::NORM_MINMAX);

		// Local feature extraction and matching
		std::vector<cv::Mat> results;
		cv::Mat dst;
		for (int i = 0; i < 4; i++) {
			features(target, scene, t_gray, s_gray, dst, i);
			results.push_back(dst.clone());
		}

		// Combine the four results into one image
		cv::Mat tmp0, tmp1, tmp;
		cv::hconcat(results[0], results[1], tmp0);
		cv::hconcat(results[2], results[3], tmp1);
		cv::vconcat(tmp0, tmp1, tmp);

		// When the image size is too large, reduce the size.
		cv::Mat small;
		if (tmp.cols >= 640 || tmp.rows >= 480)
		{
			resize(tmp, small, cv::Size(), beta, beta);
		}
		else
			tmp.copyTo(small);

		if (flgCamera)
		{
			// Video setting (first time only)
			if (!flgVideoWrite)
			{
				max_width = small.cols;
				if (max_height < small.rows) {
					max_height = small.rows;
				}
				cv::Size video_size(max_width, max_height);
				writer.open(filename, fourcc, fps, video_size);
				if (!writer.isOpened())
				{
					std::cout << "Output video file can not be opened." << std::endl;
					break;
				}

				std::cout << "-- video setting --" << std::endl;
				std::cout << "total_width: " << max_width << " max_height: " << max_height << std::endl;
				std::cout << "video_size  width: " << video_size.width << " height: " << video_size.height << std::endl;
				std::cout << "small: (" << small.cols << "," << small.rows << ")" << std::endl;
				flgVideoWrite = true;
			}
			else
				writer << small;
		}

		imshow("Match", small);

		// finish
		int key = cv::waitKey(33);
		if (key == 0x1b) // ESCキー
		{
			if (!flgCamera)
				cv::imwrite(filename, small);
			std::cout << "finish..." << std::endl;
			break;
		}
	}
}


//-------------------
// Main
//-------------------
int main(int argc, char* argv[])
{
	cv::Mat target;	//target image
	cv::Mat scene;	//scene image
	cv::VideoCapture cap;
	bool flgCamera = false;	//use camera / don't use camera

	std::string filename;	//output video file name
	cv::VideoWriter writer;

	if (argc <= 2)
	{
		std::cout << "Arguments for using the camera\n  <target image> <output filename(.avi)>" << std::endl;
		std::cout << "Arguments for using the image (don't use camera)\n  <target image> <scene image> <output filename(.jpeg)>" << std::endl;
		return -1;
	}
	else if (argc == 3)
	{
		flgCamera = true;
		target = cv::imread(argv[1]);
		filename = argv[2];
		//filename += ".avi";
		initCamera(scene, cap);
		writer = cv::VideoWriter(filename, fourcc, fps, cv::Size(640, 480));

	}
	else if (argc == 4)
	{
		target = cv::imread(argv[1]);
		scene = cv::imread(argv[2]);

		if (target.empty() || scene.empty())
		{
			std::cout << "Don't open images." << std::endl;
			return -1;
		}

		// output video filename
		filename = argv[3];
		//filename += ".jpeg";
	}

	//cv::imshow("target", target);
	//cv::imshow("scene", scene);
	//cv::waitKey();

	run(flgCamera, target, scene, cap, writer, filename);


	return 0;
}
