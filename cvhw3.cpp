#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>

#include <opencv2/opencv.hpp>

using std::vector;
using std::string;
using std::map;
using std::multimap;
using std::pair;
using std::tuple;

using cv::Mat;
using cv::Mat1i;
using cv::Mat1f;
using cv::Mat1d;
using cv::Mat1s;
using cv::Matx22f;
using cv::Matx44f;
using cv::Vec4f;

using cv::Scalar;
using cv::InputArray;
using cv::OutputArray;
using cv::Point2f;
using cv::Point2i;
using cv::Rect;
using cv::Size;
using cv::String;
using cv::VideoCapture;
using cv::KalmanFilter;

#define QQQ do {std::cerr << "QQQ " << __FUNCTION__ << " " << __LINE__ << std::endl;} while(0)

void printTimeSinceLastCall(const char* message)
{
	static int64 freq = static_cast<int>(cv::getTickFrequency());
	static int64 last = cv::getTickCount();

	int64 curr = cv::getTickCount();
	int64 delta = curr - last;
	double deltaMs = (double)delta / freq * 1000;
	printf("%s: %.4f\n", message, deltaMs);
	fflush(stdout);

	last = curr;
}

Mat videoMedian(const vector<Mat> &images)
{
	if (images.size() == 0)
		return Mat{};

	// TODO: Work with color too
	CV_Assert(images[0].type() == CV_8U);

	vector<int> pixelValues(images.size());

	int imageCount = images.size();
	int rows = images[0].rows;
	int cols = images[0].cols;

	Mat median{rows, cols, images[0].type()};

	for (int r = 0; r < rows; r++)
	{
		std::cout << "Background of row " << r << std::endl;
		for (int c = 0; c < cols; c++)
		{
			for (int t = 0; t < imageCount; t++)
			{
				pixelValues[t] = images[t].at<unsigned char>(r, c);
			}

			std::nth_element(pixelValues.begin(), pixelValues.begin() + pixelValues.size() / 2, pixelValues.end());
			median.at<unsigned char>(r, c) = pixelValues[pixelValues.size() / 2];
		}
	}

	return median;
}

struct KalmanState
{
	Vec4f mean;
	Matx44f covar;
};

struct Tracker
{
	int id;
	float prob;
	Point2f dir;
	KalmanState kalman;

	Tracker(int id = 0, float prob = 1, Point2f dir = Point2f{0, 0})
		: id{id}, prob{prob}, dir{dir}
	{ }

	bool operator<(const Tracker& other) const
	{
		return id < other.id;
	}
};

struct FrameState
{
	vector<Point2f> markers;
	vector<Matx22f> covar;

	// How the markers from the last frame moved
	multimap<int, int> mov;
	multimap<int, int> imov;

	//multimap<int, Tracker> trackers;
	vector<vector<Tracker>> trackers;
	map<int, int> ids;

	int firstNewId;
};

void drawStateTransitions(Mat &on, const FrameState& first, const FrameState& second, float scalingFactor = 1)
{
	for (pair<int, int> transition : second.mov)
	{
		cv::arrowedLine(on, first.markers[transition.first] * scalingFactor, second.markers[transition.second] * scalingFactor, Scalar{0, 255, 0}, std::floor(scalingFactor));
	}
}

// Taken from: http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/

// Chi-Square table is: https://people.richland.edu/james/lecture/m170/tbl-chi.html
// The square root of the probability is passed

// Modified: Now expects float matrix instead of double

cv::RotatedRect getErrorEllipse(double chisquare_val, cv::Point2f mean, cv::Mat covmat) {

	//Get the eigenvalues and eigenvectors
	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(covmat, eigenvalues, eigenvectors);

	//Calculate the angle between the largest eigenvector and the x-axis
	double angle = atan2(eigenvectors.at<float>(0, 1), eigenvectors.at<float>(0, 0));

	//Shift the angle to the [0, 2pi] interval instead of [-pi, pi]
	if (angle < 0)
		angle += 6.28318530718;

	//Conver to degrees instead of radians
	angle = 180 * angle / 3.14159265359;

	//Calculate the size of the minor and major axes
	double halfmajoraxissize = chisquare_val*sqrt(eigenvalues.at<float>(0));
	double halfminoraxissize = chisquare_val*sqrt(eigenvalues.at<float>(1));

	//Return the oriented ellipse
	//The -angle is used because OpenCV defines the angle clockwise instead of anti-clockwise
	// NOTE(Andrey): Modified because for us Y axis is down)
	return cv::RotatedRect(mean, cv::Size2f(halfmajoraxissize, halfminoraxissize), angle);

}

int main(int argc, char *argv[])
{
	VideoCapture vid;

	//
	// Load the video
	//

	{
		string fileName;

		if (argc == 2)
			fileName = string{argv[1]};

		bool openedImages = false;

		while (!openedImages)
		{
			if (fileName == "")
			{
				std::cout << std::endl;
				std::cout << "\tEnter the filename: (bug00, bugs11, bugs12, bugs14, bugs25 ...) If no extension is specified .mp4 is assumed:" << std::endl;
				std::cout << "?> ";
				std::cin >> fileName;
			}

			if (fileName.find(".") == std::string::npos)
				fileName += ".mp4";

			fileName = "../video/" + fileName;

			vid = VideoCapture(fileName);

			if (vid.isOpened())
			{
				openedImages = true;
			}
			else
			{
				std::cerr << "\tERROR: Couldn't open file " + fileName << std::endl;
				fileName = "";
			}
		}
	}

	auto getForeground = [](const Mat &frame, const Mat &background) -> Mat
	{
		Mat foreground;
		cv::absdiff(frame, background, foreground);
		return foreground;
	};

	auto getMarkerMask = [](const Mat &foreground) -> Mat
	{
		Mat markerMask = foreground >= 30;

		cv::erode(markerMask,
			markerMask,
			cv::getStructuringElement(cv::MORPH_ELLIPSE, Size{3, 3}),
			cv::Point{-1, -1},
			1);

		//cv::morphologyEx(markerMask,
		//markerMask,
		//cv::MORPH_CLOSE,
		//cv::getStructuringElement(cv::MORPH_ELLIPSE, Size{15, 15}));

		//cv::morphologyEx(markerMask,
		//markerMask,
		//cv::MORPH_OPEN,
		//cv::getStructuringElement(cv::MORPH_ELLIPSE, Size{7, 7}));

		return markerMask;
	};

	auto getFrameMarkers = [](const Mat &markerMask, vector<Point2f> &outMean, vector<Matx22f> &outCovar)
	{
		Mat labels;
		Mat stats;
		Mat1d centroids;

		cv::connectedComponentsWithStats(
			markerMask,
			labels,
			stats,
			centroids,
			4, CV_32S);

		// NOTE: label 0 is the background. We ignore it here.

		vector<Point2f> ccMean(std::max(centroids.rows - 1, 0));
		vector<Matx22f> ccCovar(ccMean.size(), Matx22f{0, 0, 0, 0});

		for (int r = 1; r < centroids.rows; r++)
			ccMean[r - 1] = Point2f{(float)centroids(r, 0), (float)centroids(r, 1)};

		for (int r = 0; r < labels.rows; r++)
		{
			int *ptr = labels.ptr<int>(r);

			for (int c = 0; c < labels.cols; c++, ptr++)
			{
				if (*ptr == 0)
					continue;

				int id = *ptr - 1;

				Point2f mean = ccMean[id];
				Matx22f covar = ccCovar[id];

				covar(0, 0) += (c - mean.x) * (c - mean.x);
				covar(0, 1) += (c - mean.x) * (r - mean.y);
				covar(1, 0) += (r - mean.y) * (c - mean.x);
				covar(1, 1) += (r - mean.y) * (r - mean.y);

				ccCovar[id] = covar;
			}
		}

		for (int r = 1; r < stats.rows; r++)
		{
			int count = stats.at<int>(r, cv::CC_STAT_AREA);

			if (count > 0)
			{
				ccCovar[r - 1](0, 0) /= count;
				ccCovar[r - 1](0, 1) /= count;
				ccCovar[r - 1](1, 0) /= count;
				ccCovar[r - 1](1, 1) /= count;
			}
		}

		outMean.resize(0);
		outCovar.resize(0);

		for (int r = 1; r < centroids.rows; r++)
		{
			if (stats.at<int>(r, cv::CC_STAT_AREA) > 20)
			{
				outMean.push_back(ccMean[r - 1]);
				outCovar.push_back(ccCovar[r - 1]);
			}
		}
	};

	vector<Mat> frames;

	{
		int i = 0;
		while (true)
		{
			Mat frame;
			if (!vid.read(frame))
				break;

			// TODO: Work with color video
			cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

			if (frames.size() > 0)
			{
				Mat diff;
				cv::absdiff(frame, frames.back(), diff);
				if (cv::countNonZero(diff > 10) <= 50)
					continue;
			}

			frames.push_back(frame);
			std::cout << "Loaded frame " << i++ << std::endl;
		}
	}

	Mat background = videoMedian(vector<Mat>{frames.begin(), frames.begin() + std::min(frames.size(), (size_t)100)});

	int frameCount = frames.size();
	int frameIndex = 0;

	vector<FrameState> states(frames.size());
	int idCount = 0;

	//
	// Create Kalman Filter
	//

	KalmanFilter kalmanFilter;

	{
		// State: [px, py, vx, vy]
		// Measurement: [px, py]
		kalmanFilter.init(4, 2);

		// Transition matrix:
		// [1	0	dt	0]
		// [0	1	0	dt]
		// [0	0	1	0]
		// [0	0	0	1]
		//
		// The matrix is initialized by Mat::eye, we only need to set the time delta.
		// We set the dt to 1 time unit;
		kalmanFilter.transitionMatrix.at<float>(0, 2) = 1;
		kalmanFilter.transitionMatrix.at<float>(1, 3) = 1;

		// Measurement matrix:
		// [1	0	0	0]
		// [0	1	0	0]
		kalmanFilter.measurementMatrix.at<float>(0, 0) = 1;
		kalmanFilter.measurementMatrix.at<float>(1, 1) = 1;

		// Process noise covariance:
		// [epx	0	0	0]
		// [0	epy	0	0]
		// [0	0	evx	0]
		// [0	0	0	evy]
		kalmanFilter.processNoiseCov.at<float>(0, 0) = 0.01f;
		kalmanFilter.processNoiseCov.at<float>(1, 1) = 0.01f;
		kalmanFilter.processNoiseCov.at<float>(2, 2) = 1.0f * 10;
		kalmanFilter.processNoiseCov.at<float>(3, 3) = 1.0f * 10;
	}

	auto updateKalman = [&kalmanFilter](KalmanState currState, Point2f measurement, Matx22f measurementCov, float *outDist) -> KalmanState
	{
		// Copy state into the Kalman Filter
		for (int i = 0; i < 4; i++)
			kalmanFilter.statePost.at<float>(i) = currState.mean[i];
		for (int i = 0; i < 4; i++)
			for (int k = 0; k < 4; k++)
				kalmanFilter.errorCovPost.at<float>(i, k) = currState.covar(i, k);

		// Predict
		kalmanFilter.predict();

		if (outDist)
		{
			Point2f velocity{currState.mean[2], currState.mean[3]};
			Point2f direction{measurement.x - currState.mean[0], measurement.y - currState.mean[1]};
			float magV = std::sqrt(velocity.x * velocity.x + velocity.y * velocity.y);
			float magD = std::sqrt(direction.x * direction.x + direction.y * direction.y);

			float cosTheta = velocity.dot(direction) / std::max(magV, 0.01f) / std::max(magD, 0.01f);

			// Mahalanobis distance https://en.wikipedia.org/wiki/Mahalanobis_distance
			// between predicted position and measured position.
			float predictedX = kalmanFilter.statePre.at<float>(0);
			float predictedY = kalmanFilter.statePre.at<float>(1);

			predictedX -= measurement.x;
			predictedY -= measurement.y;

			*outDist = std::sqrt(
				predictedX * (measurementCov(0, 0) * predictedX + measurementCov(0, 1) * predictedY) +
				predictedY * (measurementCov(1, 0) * predictedX + measurementCov(1, 1) * predictedY));

			*outDist *= 5 - cosTheta;
		}

		// Correct
		for (int i = 0; i < 2; i++)
			for (int k = 0; k < 2; k++)
				kalmanFilter.measurementNoiseCov.at<float>(i, k) = measurementCov(i, k);

		kalmanFilter.correct(Mat{measurement, false});

		// Copy state out of the Kalman Filter
		for (int i = 0; i < 4; i++)
			currState.mean[i] = kalmanFilter.statePost.at<float>(i);
		for (int i = 0; i < 4; i++)
			for (int k = 0; k < 4; k++)
				currState.covar(i, k) = kalmanFilter.errorCovPost.at<float>(i, k);

		return currState;
	};

	for (int t = 0; t < frameCount; t++)
	{
		//
		// Extract foreground and markers
		//

		Mat foreground = getForeground(frames[t], background);
		Mat markerMask = getMarkerMask(foreground);


		vector<Point2f> currMarkers;
		getFrameMarkers(markerMask, currMarkers, states[t].covar);

		vector<Point2f> lastMarkers = t > 0 ? states[t - 1].markers : vector<Point2f>{};

		vector<vector<Tracker>> emptyTrackers;

		auto &currTrackers = states[t].trackers;
		auto &lastTrackers = t > 0 ? states[t - 1].trackers : emptyTrackers;

		//
		// Create the transition graph
		//

		for (int i = 0; i < lastMarkers.size(); i++)
		{
			for (int k = 0; k < currMarkers.size(); k++)
			{
				float dx = lastMarkers[i].x - currMarkers[k].x;
				float dy = lastMarkers[i].y - currMarkers[k].y;

				float squareDist = dx * dx + dy * dy;

				const float lowThresh = 50 * 50;

				if (squareDist <= lowThresh)
				{
					states[t].mov.insert({i, k});
					states[t].imov.insert({k, i});
				}
			}
		}

		//
		// Simulate movement from previous frame
		//

		currTrackers = vector<vector<Tracker>>(currMarkers.size());

		for (int i = 0; i < (int)lastTrackers.size(); i++)
		{
			auto edgeRange = states[t].mov.equal_range(i);

			vector<int> edges;

			for (auto e = edgeRange.first; e != edgeRange.second; e++)
				edges.push_back(e->second);

			vector<KalmanState> kalmanStates(edges.size());
			vector<float> weights(edges.size());

			for (Tracker tracker : lastTrackers[i])
			{
				float totalWeight = 0;

				for (int i = 0; i < (int)edges.size(); i++)
				{
					float distance;
					kalmanStates[i] = updateKalman(tracker.kalman, currMarkers[edges[i]], states[t].covar[edges[i]], &distance);
					weights[i] = 1 / std::max(distance, 0.1f);
					totalWeight += weights[i];
				}

				for (int i = 0; i < edges.size(); i++)
				{
					int targetMarker = edges[i];
					float newProb = tracker.prob * weights[i] / totalWeight;
					newProb *= 0.97f;

					auto it = std::find_if(currTrackers[targetMarker].begin(), currTrackers[targetMarker].end(), [tracker](Tracker other) { return other.id == tracker.id; });

					if (newProb >= 0.005)
					{
						if (it != currTrackers[targetMarker].end())
						{
							it->kalman.mean = (it->kalman.mean * it->prob + kalmanStates[i].mean * newProb) / (it->prob + newProb);
							it->kalman.covar = (it->kalman.covar * it->prob + kalmanStates[i].covar * newProb) * (1 / (it->prob + newProb));
							it->prob += newProb;
						}
						else
						{
							Tracker newTracker{tracker.id, newProb, tracker.dir};
							newTracker.kalman = kalmanStates[i];
							currTrackers[targetMarker].push_back(newTracker);
						}
					}
				}
			}
		}

		vector<bool> markerUsed(currMarkers.size(), false);
		vector<int> usedIds;

		vector<tuple<float, int, Tracker>> allTrackers;

		for (int i = 0; i < (int)currTrackers.size(); i++)
		{
			for (Tracker t : currTrackers[i])
			{
				allTrackers.push_back({t.prob, i, t});
			}
		}

		std::sort(allTrackers.rbegin(), allTrackers.rend());

		vector<pair<int, Tracker>> newTrackers;

		for (auto item : allTrackers)
		{
			float prob;
			int marker;
			Tracker tracker;

			std::tie(prob, marker, tracker) = item;

			if (!markerUsed[marker] && std::find(usedIds.begin(), usedIds.end(), tracker.id) == usedIds.end())
			{
				tracker.prob = 1;
				newTrackers.push_back({marker, tracker});
				markerUsed[marker] = true;
				usedIds.push_back(tracker.id);
				states[t].ids[tracker.id] = marker;

				// Backpatch - put the id in all previous frames where
				// it was just a probability.

				int lastMarker = marker;

				for (int i = t - 1; i >= 0; i--)
				{
					if (states[i].ids.find(tracker.id) != states[i].ids.end())
						break;

					auto positionRange = states[i + 1].imov.equal_range(lastMarker);

					int maxMarker = -1;
					float maxProb = 0;

					// Find all edges that went to the last marker,
					// and find where was the tracker with the biggest probability.

					for (auto p = positionRange.first; p != positionRange.second; p++)
					{
						int targetMarker = p->second;

						for (Tracker t : states[i].trackers[targetMarker])
						{
							if (t.id == tracker.id &&
								t.prob > maxProb)
							{
								maxMarker = targetMarker;
								maxProb = t.prob;
							}
						}
					}

					if (maxMarker < 0)
						break;

					states[i].ids[tracker.id] = maxMarker;
					lastMarker = maxMarker;
				}
			}
		}

		for (auto item : allTrackers)
		{
			float prob;
			int marker;
			Tracker tracker;

			std::tie(prob, marker, tracker) = item;

			if (std::find(usedIds.begin(), usedIds.end(), tracker.id) == usedIds.end())
				newTrackers.push_back({marker, tracker});
		}

		for (auto &v : currTrackers)
			v.resize(0);
		for (pair<int, Tracker> t : newTrackers)
			currTrackers[t.first].push_back(t.second);

		//
		// Assign ids to empty markers
		//

		states[t].firstNewId = idCount;

		for (int i = 0; i < (int)currTrackers.size(); i++)
		{
			if (currTrackers[i].empty())
			{
				int newId = idCount++;
				Tracker tracker{newId};
				tracker.kalman.mean = Vec4f{currMarkers[i].x, currMarkers[i].y, 0, 0};
				tracker.kalman.covar = Matx44f::zeros();
				currTrackers[i].push_back(tracker);
				states[t].ids[newId] = i;
			}
		}

		states[t].markers = currMarkers;

		std::cout << "Processed frame " << t << std::endl;
	}

#ifdef _WIN32
	enum Keys
	{
		KEY_ARROW_LEFT = 0x250000,
		KEY_ARROW_UP = 0x260000,
		KEY_ARROW_RIGHT = 0x270000,
		KEY_ARROW_DOWN = 0x280000,
		KEY_HOME = 0x240000,
		KEY_END = 0x230000,
	};
#else
	enum Keys
	{
		KEY_ARROW_LEFT = 65361,
		KEY_ARROW_UP = 65362,
		KEY_ARROW_RIGHT = 65363,
		KEY_ARROW_DOWN = 65364,
		KEY_HOME = 65360,
		KEY_END = 65367,
	};
#endif

	bool playing = true;
	int mode = 4;

	bool qShowMarkers = true;
	bool qShowTransitions = false;
	bool qShowIds = true;
	bool qShowTrails = false;
	bool qShowMarkerIds = false;
	bool qShowVelocities = false;
	bool qShowMarkerCovar = false;
	bool qShowTrackerCovar = false;

	float scalingFactor = 2;

	while (true)
	{
		Mat frame = frames[frameIndex];
		Mat foreground = getForeground(frame, background);
		Mat markerMask = getMarkerMask(foreground);

		auto &currMarkers = states[frameIndex].markers;
		vector<Point2f> lastMarkers;

		if (frameIndex > 0)
		{
			lastMarkers = states[frameIndex - 1].markers;
		}

		Mat display;

		switch (mode)
		{
		case 1:
			display = frame.clone();
			break;
		case 2:
			display = foreground.clone();
			break;
		case 3:
			cv::cvtColor(markerMask, display, cv::COLOR_GRAY2BGR);
			break;
		case 4:
			cv::cvtColor(frame, display, cv::COLOR_GRAY2BGR);
			break;
		case 5:
		case 6:
		{
			display = Mat{frame.size(), CV_8UC3};
			display.setTo(0);

			const int historyLength = 16;

			for (int t = 0; t < historyLength; t++)
			{
				if (frameIndex - historyLength + t < 0)
					continue;

				Mat valueMat{1, 1, CV_8U};
				valueMat.setTo(t * 256 / historyLength);
				Mat colorMat{1, 1, CV_8UC3};
				// TODO(Andrey): Apply colormap to grayscale image
				cv::applyColorMap(valueMat, colorMat, cv::COLORMAP_JET);
				Scalar color{colorMat.at<cv::Vec3b>()};

				Mat frame = frames[frameIndex - historyLength + t];
				Mat foreground = getForeground(frame, background);
				Mat markerMask = getMarkerMask(foreground);

				if (mode == 5)
				{
					display.setTo(color, markerMask);
				}

				auto &lastMarkers = states[frameIndex - historyLength + t].markers;
				for (int i = 0; i < lastMarkers.size(); i++)
				{
					if (mode == 6)
					{
						cv::circle(display, lastMarkers[i], 2, color, -1);
					}
				}
			}
			for (int i = 0; i < currMarkers.size(); i++)
			{
				cv::circle(display, currMarkers[i], 4, Scalar{0, 0, 255}, -1);
			}
			break;
		}
		case 7:
			if (frameIndex > 0)
				cv::absdiff(frames[frameIndex], frames[frameIndex - 1], display);
			else
				display = frame.clone();
			break;
		default:
			display = frame.clone();
		}

		if (display.type() == CV_8U)
			cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);

		cv::resize(display, display, Size{(int)(display.cols * scalingFactor), (int)(display.rows * scalingFactor)});

		if (qShowTrails)
		{
			const vector<Scalar> colorPalette = {
				{180, 119, 31},
				{14, 127, 255},
				{44, 160, 44},
				{40, 39, 214},
				{189, 103, 148},
				{75, 86, 140},
				{194, 119, 227},
				{127, 127, 127},
				{34, 189, 188},
				{207, 190, 23},
			};

			vector<int> activeIds;

			for (pair<int, int> p : states[frameIndex].ids)
				activeIds.push_back(p.first);

			for (int k = frameIndex - 1; k >= 0 && frameIndex - k < 30; k--)
			{
				if (activeIds.size() == 0)
					break;

				vector<int> newIds;

				for (int id : activeIds)
				{
					if (states[k].ids.find(id) != states[k].ids.end())
					{
						// TODO: Refactor
						int fromMarker = states[k + 1].ids[id];
						Vec4f from4 = std::find_if(
							states[k + 1].trackers[fromMarker].begin(),
							states[k + 1].trackers[fromMarker].end(),
							[id](const Tracker &other) { return other.id == id; })->kalman.mean;

						fromMarker = states[k].ids[id];
						Vec4f to4 = std::find_if(
							states[k].trackers[fromMarker].begin(),
							states[k].trackers[fromMarker].end(),
							[id](const Tracker &other) { return other.id == id; })->kalman.mean;

						Point2f from{from4[0], from4[1]};
						Point2f to{to4[0], to4[1]};

						Scalar color = colorPalette[id % colorPalette.size()];
						cv::line(display, from * scalingFactor, to * scalingFactor, color, std::floor(scalingFactor));
						newIds.push_back(id);
					}
				}

				activeIds = newIds;
			}
		}
		if (qShowMarkerCovar)
		{
			for (int i = 0; i < (int)states[frameIndex].markers.size(); i++)
			{
				float chi_square_95_percent = 2.4477f;
				cv::RotatedRect rect = getErrorEllipse(chi_square_95_percent + 1, states[frameIndex].markers[i], Mat{states[frameIndex].covar[i], false});
				rect.center *= scalingFactor;
				rect.size *= scalingFactor;
				cv::ellipse(display, rect, Scalar{0, 0, 0}, 3);
				cv::ellipse(display, rect, Scalar{0, 200, 200}, 2);
			}
		}
		if (qShowTrackerCovar || qShowIds)
		{
			for (int i = 0; i < states[frameIndex].trackers.size(); i++)
			{
				for (const Tracker &tracker : states[frameIndex].trackers[i])
				{
					if (states[frameIndex].ids.find(tracker.id) == states[frameIndex].ids.end() ||
						states[frameIndex].ids[tracker.id] != i)
						continue;

					Point2f mean{tracker.kalman.mean[0], tracker.kalman.mean[1]};
					Matx22f covar{tracker.kalman.covar(0, 0), tracker.kalman.covar(0, 1), tracker.kalman.covar(1, 0), tracker.kalman.covar(1, 1)};

					if (qShowTrackerCovar)
					{
						float chi_square_95_percent = 2.4477f;
						cv::RotatedRect rect = getErrorEllipse(chi_square_95_percent + 1, mean, Mat{covar});
						rect.center *= scalingFactor;
						rect.size *= scalingFactor;
						cv::ellipse(display, rect, Scalar{0, 0, 0}, 3);
						cv::ellipse(display, rect, Scalar{244, 80, 66}, 2);

						cv::arrowedLine(display, mean * scalingFactor,
							(mean + Point2f{tracker.kalman.mean[2], tracker.kalman.mean[3]} *4) * scalingFactor, Scalar{225, 142, 170}, 2);
					}

					if (qShowIds)
					{
						Scalar color;

						if (tracker.id >= states[frameIndex].firstNewId)
						{
							color = Scalar{44, 160, 44};
						}
						else if (frameIndex + 1 != frameCount && states[frameIndex + 1].ids.find(tracker.id) == states[frameIndex + 1].ids.end())
						{
							color = Scalar{40, 39, 214};
						}
						else
						{
							color = Scalar{255, 255, 255};
						}

						cv::putText(display, std::to_string(tracker.id), mean * scalingFactor, cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar{0, 0, 0}, 3);
						cv::putText(display, std::to_string(tracker.id), mean * scalingFactor, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
					}
				}
			}
		}
		if (qShowMarkers)
		{
			for (int i = 0; i < currMarkers.size(); i++)
			{
				cv::circle(display, currMarkers[i] * scalingFactor, 4, Scalar{0, 0, 255}, -1);
			}
		}
		if (qShowTransitions && frameIndex > 0)
		{
			drawStateTransitions(display, states[frameIndex - 1], states[frameIndex], scalingFactor);
		}
		if (qShowMarkerIds)
		{
			for (int i = 0; i < (int)states[frameIndex].markers.size(); i++)
			{
				Point2f marker = states[frameIndex].markers[i];
				cv::putText(display, std::to_string(i), marker * scalingFactor, cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar{0, 0, 0}, 3);
				cv::putText(display, std::to_string(i), marker * scalingFactor, cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar{0, 255, 255});
			}
		}
		if (qShowVelocities)
		{
			for (int i = 0; i < (int)states[frameIndex].trackers.size(); i++)
			{
				Point2f marker = states[frameIndex].markers[i];

				for (Tracker t : states[frameIndex].trackers[i])
				{
					if (states[frameIndex].ids.find(t.id) != states[frameIndex].ids.end())
					{
						cv::arrowedLine(display, marker * scalingFactor, (marker + t.dir * 20) * scalingFactor, Scalar{244, 134, 66}, std::floor(scalingFactor));
					}
				}
			}
		}

		cv::imshow("w", display);
		cv::setWindowTitle("w", "Active trackers: " + std::to_string(states[frameIndex].ids.size()) + " Frame: " + std::to_string(frameIndex));

		if (playing)
			frameIndex = std::min(frameIndex + 1, frameCount - 1);

		int pressedKey = cv::waitKey(playing ? 30 : 0);

		switch (pressedKey)
		{
		case 'q':
			std::exit(1);
		case ' ':
			playing = !playing;
			break;
		case KEY_HOME:
			frameIndex = 0;
			break;
		case KEY_END:
			frameIndex = frameCount - 1;
			break;
		case KEY_ARROW_RIGHT:
			frameIndex = std::min(frameIndex + (playing ? 4 : 1), frameCount - 1);
			break;
		case KEY_ARROW_LEFT:
			frameIndex = std::max(frameIndex - (playing ? 4 : 1), 0);
			break;
		case KEY_ARROW_DOWN:
			mode++;
			break;
		case KEY_ARROW_UP:
			mode = std::max(mode - 1, 1);
			break;
		case 't':
			qShowTransitions = !qShowTransitions;
			break;
		case 'n':
			qShowIds = !qShowIds;
			break;
		case 'm':
			qShowMarkers = !qShowMarkers;
			break;
		case 'r':
			qShowTrails = !qShowTrails;
			break;
		case 'd':
			qShowMarkerIds = !qShowMarkerIds;
			break;
		case 'v':
			qShowVelocities = !qShowVelocities;
			break;
		case 'c':
			qShowMarkerCovar = !qShowMarkerCovar;
			break;
		case 'C':
			qShowTrackerCovar = !qShowTrackerCovar;
			break;
		default:
			break;
		}

		if ('0' <= pressedKey && pressedKey <= '9')
			mode = pressedKey - '0';
	}

	return 0;
}
