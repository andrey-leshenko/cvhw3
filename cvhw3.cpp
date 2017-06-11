#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp>

using std::vector;
using std::string;
using std::map;
using std::multimap;
using std::pair;
using std::tuple;
using std::mutex;
using std::condition_variable;

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

#define IT_RANGE(X) (X).begin(), (X).end()

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

//
// Image Processing
//

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
		//std::cout << "Background of row " << r << std::endl;
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

Mat getForeground(const Mat &frame, const Mat &background)
{
	Mat foreground;
	cv::absdiff(frame, background, foreground);
	return foreground;
}

Mat getMarkerMask(const Mat &foreground)
{
	Mat markerMask = foreground >= 30;

	cv::erode(markerMask,
		markerMask,
		cv::getStructuringElement(cv::MORPH_ELLIPSE, Size{3, 3}),
		cv::Point{-1, -1},
		1);

	//cv::dilate(markerMask,
	//	markerMask,
	//	cv::getStructuringElement(cv::MORPH_ELLIPSE, Size{3, 3}),
	//	cv::Point{-1, -1},
	//	1);

	return markerMask;
}

void getFrameMarkers(const Mat &markerMask, vector<Point2f> &outMean, vector<Matx22f> &outCovar)
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
}

//
// Tracking
//

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
	float distanceToLastSureMarker;
	int prevMarker;

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

struct ProcessState
{
	mutex framesLock;
	condition_variable frameLoadedCond;
	vector<Mat> frames;
	bool finishedReadingFrames = false;
	int totalFrameCount = 0;

	Mat background;

	mutex statesLock;
	vector<FrameState> states;
	// The index of the earliest state that was changed when this state was computed
	vector<int> firstChangedState;
};

void loadVideoGrayscale(ProcessState *ps, VideoCapture *cap)
{
	Mat frame;
	Mat prev;
	Mat diff;

	double frameCount = cap->get(cv::CAP_PROP_FRAME_COUNT);
	ps->totalFrameCount = frameCount > 0 ? (int)(frameCount + 0.5) : 0;

	while (true)
	{
		if (!cap->read(frame))
			break;

		cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

		// Sometimes there are duplicated frames in the video.
		// Here we ignore these frames.

		if (!prev.empty())
		{
			cv::absdiff(frame, prev, diff);
			if (cv::countNonZero(diff > 10) <= 50)
				continue;
		}

		{
			std::lock_guard<std::mutex> lock(ps->framesLock);
			ps->frames.push_back(frame.clone());
		}

		ps->frameLoadedCond.notify_all();
	}

	{
		std::lock_guard<std::mutex> lock(ps->framesLock);
		ps->finishedReadingFrames = true;
	}

	ps->frameLoadedCond.notify_all();

	std::cout << "Loaded video" << std::endl;
}

void buildFrameStates(ProcessState *ps, int maxBackgroundFrames)
{
	//
	// Find Background
	//

	vector<Mat> backgroundFrames;

	{
		std::unique_lock<std::mutex>lk(ps->framesLock);

		while (!ps->finishedReadingFrames && ps->frames.size() < maxBackgroundFrames)
		{
			ps->frameLoadedCond.wait(lk);
		}

		backgroundFrames = vector<Mat>{ps->frames.begin(), ps->frames.begin() + std::min((int)ps->frames.size(), maxBackgroundFrames)};
	}

	std::cout << "Creating background model" << std::endl;

	Mat background = videoMedian(backgroundFrames);

	{
		std::lock_guard<std::mutex> lock(ps->framesLock);
		ps->background = background;
	}

	std::cout << "Created background model" << std::endl;

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

	int t = 0;
	int idCount = 0;

	while (true)
	{
		//
		// Make sure we have a new frame available (from the video loading thread)
		//

		Mat currFrame;

		{
			std::unique_lock<mutex>lk(ps->framesLock);

			while (ps->background.empty())
			{
				ps->frameLoadedCond.wait(lk);
			}

			// We must load at least one frame
			while (!ps->finishedReadingFrames && t >= ps->frames.size())
			{
				ps->frameLoadedCond.wait(lk);
			}

			if (ps->finishedReadingFrames && t >= ps->frames.size())
				break;

			currFrame = ps->frames[t];
		}

		//
		// Extract foreground and markers
		//

		Mat foreground = getForeground(currFrame, ps->background);
		Mat markerMask = getMarkerMask(foreground);

		FrameState currState;

		FrameState emptyState;
		const FrameState &lastState = t > 0 ? ps->states[t - 1] : emptyState;

		getFrameMarkers(markerMask, currState.markers, currState.covar);

		//
		// Create the transition graph
		//

		for (int i = 0; i < lastState.markers.size(); i++)
		{
			for (int k = 0; k < currState.markers.size(); k++)
			{
				float dx = lastState.markers[i].x - currState.markers[k].x;
				float dy = lastState.markers[i].y - currState.markers[k].y;

				float squareDist = dx * dx + dy * dy;

				const float lowThresh = 50 * 50;

				if (squareDist <= lowThresh)
				{
					currState.mov.insert({i, k});
					currState.imov.insert({k, i});
				}
			}
		}

		//
		// Simulate movement from previous frame
		//

		currState.trackers = vector<vector<Tracker>>(currState.markers.size());

		for (int marker = 0; marker < (int)lastState.trackers.size(); marker++)
		{
			auto edgeRange = currState.mov.equal_range(marker);

			vector<int> edges;

			for (auto e = edgeRange.first; e != edgeRange.second; e++)
				edges.push_back(e->second);

			vector<KalmanState> kalmanStates(edges.size());
			vector<float> weights(edges.size());

			for (Tracker tracker : lastState.trackers[marker])
			{
				float totalWeight = 0;

				for (int i = 0; i < (int)edges.size(); i++)
				{
					float distance;
					kalmanStates[i] = updateKalman(tracker.kalman, currState.markers[edges[i]], currState.covar[edges[i]], &distance);
					weights[i] = 1 / std::max(distance, 0.1f);
					totalWeight += weights[i];
				}

				for (int i = 0; i < edges.size(); i++)
				{
					int targetMarker = edges[i];
					float newProb = tracker.prob * weights[i] / totalWeight;
					newProb *= 0.97f;

					auto it = std::find_if(IT_RANGE(currState.trackers[targetMarker]), [tracker](Tracker other) { return other.id == tracker.id; });

					auto calcDist = [](KalmanState &a, KalmanState &b) -> float
					{
						return std::sqrt((a.mean[0] - b.mean[0]) * (a.mean[0] - b.mean[0]) + (a.mean[1] - b.mean[1]) * (a.mean[1] - b.mean[1]));
					};

					if (newProb >= 0.005)
					{
						if (it != currState.trackers[targetMarker].end())
						{
							it->kalman.mean = (it->kalman.mean * it->prob + kalmanStates[i].mean * newProb) / (it->prob + newProb);
							it->kalman.covar = (it->kalman.covar * it->prob + kalmanStates[i].covar * newProb) * (1 / (it->prob + newProb));
							it->prob += newProb;

							float newDist = tracker.distanceToLastSureMarker + calcDist(tracker.kalman, it->kalman);

							if (newDist < it->distanceToLastSureMarker)
							{
								it->distanceToLastSureMarker = newDist;
								it->prevMarker = marker;
							}
						}
						else
						{
							Tracker newTracker{tracker.id, newProb, tracker.dir};
							newTracker.kalman = kalmanStates[i];
							newTracker.distanceToLastSureMarker = tracker.distanceToLastSureMarker + calcDist(tracker.kalman, kalmanStates[i]);
							newTracker.prevMarker = marker;
							currState.trackers[targetMarker].push_back(newTracker);
						}
					}
				}
			}
		}

		vector<bool> markerUsed(currState.markers.size(), false);
		vector<int> usedIds;

		vector<tuple<float, int, Tracker>> allTrackers;

		for (int i = 0; i < (int)currState.trackers.size(); i++)
		{
			for (Tracker t : currState.trackers[i])
			{
				allTrackers.push_back({t.prob, i, t});
			}
		}

		std::sort(allTrackers.rbegin(), allTrackers.rend());

		vector<pair<int, Tracker>> newTrackers;

		{
			std::lock_guard<mutex> lock(ps->statesLock);
			ps->firstChangedState.push_back(t);

			for (auto item : allTrackers)
			{
				float prob;
				int marker;
				Tracker tracker;

				std::tie(prob, marker, tracker) = item;



				if (!markerUsed[marker] && std::find(usedIds.begin(), usedIds.end(), tracker.id) == usedIds.end())
				{
					// Backpatch - put the id in all previous frames where
					// it was just a probability.

					{
						Tracker currTracker = tracker;
						int i = t;


						while (currTracker.prevMarker >= 0)
						{
							ps->states[i - 1].ids[currTracker.id] = currTracker.prevMarker;
							for (auto &newTracker : ps->states[i - 1].trackers[currTracker.prevMarker])
							{
								if (newTracker.id == currTracker.id)
								{
									currTracker = newTracker;
									break;
								}
							}
							i--;
						}
						ps->firstChangedState.back() = std::min(ps->firstChangedState.back(), i);
					}

					tracker.prob = 1;
					tracker.distanceToLastSureMarker = 0;
					tracker.prevMarker = -1;
					newTrackers.push_back({marker, tracker});
					markerUsed[marker] = true;
					usedIds.push_back(tracker.id);
					currState.ids[tracker.id] = marker;
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

		for (auto &v : currState.trackers)
			v.resize(0);
		for (pair<int, Tracker> t : newTrackers)
			currState.trackers[t.first].push_back(t.second);

		//
		// Assign ids to empty markers
		//

		currState.firstNewId = idCount;

		for (int i = 0; i < (int)currState.trackers.size(); i++)
		{
			if (currState.trackers[i].empty())
			{
				int newId = idCount++;
				Tracker tracker{newId};
				tracker.kalman.mean = Vec4f{currState.markers[i].x, currState.markers[i].y, 0, 0};
				tracker.kalman.covar = Matx44f::zeros();
				tracker.distanceToLastSureMarker = 0;
				tracker.prevMarker = -1;
				currState.trackers[i].push_back(tracker);
				currState.ids[newId] = i;
			}
		}

		{
			std::lock_guard<mutex> lock(ps->statesLock);
			ps->states.push_back(std::move(currState));
			t++;
		}
	}
}

//
// GUI
//

void drawStateTransitions(Mat &on, const FrameState& first, const FrameState& second, float scalingFactor = 1)
{
	for (pair<int, int> transition : second.mov)
	{
		cv::arrowedLine(on, first.markers[transition.first] * scalingFactor, second.markers[transition.second] * scalingFactor, Scalar{0, 255, 0}, std::floor(scalingFactor));
	}
}

// Taken from: http://www.visiondummy.com/2014/04/draw-error-ellipse-representing-covariance-matrix/

// Chi-Square table is: https://people.richland.edu/james/lecture/m170/tbl-chi.html
// The square root of the probability is passed.

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

#ifdef _WIN32
enum KeyCodes
{
	KEY_ARROW_LEFT = 0x250000,
	KEY_ARROW_UP = 0x260000,
	KEY_ARROW_RIGHT = 0x270000,
	KEY_ARROW_DOWN = 0x280000,
	KEY_HOME = 0x240000,
	KEY_END = 0x230000,
};
#else
enum KeyCodes
{
	KEY_ARROW_LEFT = 65361,
	KEY_ARROW_UP = 65362,
	KEY_ARROW_RIGHT = 65363,
	KEY_ARROW_DOWN = 65364,
	KEY_HOME = 65360,
	KEY_END = 65367,
};
#endif

void drawGUI(ProcessState *otherPs)
{
	ProcessState ps;
	vector<FrameState> &states = ps.states;
	size_t syncedStatesCount = 0;

	int frameIndex = 0;
	bool playing = true;
	int mode = 4;

	bool qShowMarkers		= true;
	bool qShowTransitions	= false;
	bool qShowIds			= true;
	bool qShowTrails		= false;
	bool qShowMarkerIds		= false;
	bool qShowVelocities	= false;
	bool qShowMarkerCovar	= false;
	bool qShowTrackerCovar	= false;

	float scalingFactor = 1;

	//
	// OnClick Information Printing
	//

	std::tuple<int*, float*, vector<FrameState>*> guiState{&frameIndex, &scalingFactor, &ps.states};

	cv::namedWindow("w");
	cv::setMouseCallback("w",
		[](int eventType, int x, int y, int flags, void *guiStateUntyped)
	{
		if (eventType == cv::EVENT_LBUTTONDOWN)
		{
			int *frameIndexPtr;
			float *scalingFactorPtr;
			vector<FrameState> *statesPtr;

			auto guiState = static_cast<std::tuple<int*, float*, vector<FrameState>*>*>(guiStateUntyped);
			std::tie(frameIndexPtr, scalingFactorPtr, statesPtr) = *guiState;

			int frameIndex = *frameIndexPtr;
			float scalingFactor = *scalingFactorPtr;

			FrameState &currState = (*statesPtr)[frameIndex];

			int closestMarker = -1;
			float minSqrDistance = FLT_MAX;

			for (int i = 0; i < (int)currState.markers.size(); i++)
			{
				float dx = currState.markers[i].x - x / scalingFactor;
				float dy = currState.markers[i].y - y / scalingFactor;

				float sqrDistance = dx * dx + dy * dy;

				if (sqrDistance < minSqrDistance)
				{
					minSqrDistance = sqrDistance;
					closestMarker = i;
				}
			}

			if (closestMarker >= 0)
			{
				std::cout << "================" << std::endl;
				std::cout << "FRAME: " << frameIndex << " MARKER: " << closestMarker << " " << currState.markers[closestMarker] << std::endl;
				std::cout << "TRACKERS:" << std::endl;
				for (Tracker &t : currState.trackers[closestMarker])
				{
					int sureMarker = currState.ids.find(t.id) != currState.ids.end() &&
						currState.ids[t.id] == closestMarker;
					std::cout << "- Tracker id " << t.id << (sureMarker ? " <OFFICIAL>" : "") << std::endl;
					std::cout << "\tprob: " << t.prob << std::endl;
					std::cout << "\tdistanceToLastSureMarker: " << t.distanceToLastSureMarker << std::endl;
					std::cout << "\tprevMarker: " << t.prevMarker << std::endl;
					std::cout << "\tkalman.mean: " << t.kalman.mean << std::endl;
					//std::cout << "\tkalman.covar: " << t.kalman.covar << std::endl;
				}
			}
		}
	},
		&guiState);

	while (true)
	{
		//
		// Sync process state with worker thread
		//

		{
			std::unique_lock<mutex>lk(otherPs->framesLock);

			// We must load at least one frame
			while (!otherPs->finishedReadingFrames && otherPs->frames.size() == 0)
			{
				otherPs->frameLoadedCond.wait(lk);
			}
			if (otherPs->frames.size() == 0)
				return;

			// Bring the changes
			if (otherPs->frames.size() > ps.frames.size())
			{
				ps.frames.insert(ps.frames.end(), otherPs->frames.begin() + ps.frames.size(), otherPs->frames.end());
			}

			ps.background = otherPs->background.empty() ? ps.frames[0] : otherPs->background;
			ps.totalFrameCount = otherPs->totalFrameCount;
			ps.states.resize(ps.frames.size());
		}

		{
			std::lock_guard<mutex> lock(otherPs->statesLock);

			if (syncedStatesCount < otherPs->states.size())
			{
				int startIndex = *std::min_element(otherPs->firstChangedState.begin() + syncedStatesCount, otherPs->firstChangedState.end());
				int endIndex = otherPs->states.size();

				ps.states.resize(std::max((int)ps.states.size(), endIndex));

				for (int i = startIndex; i < endIndex; i++)
					ps.states[i] = otherPs->states[i];

				syncedStatesCount = endIndex;
			}
		}

		Mat frame = ps.frames[frameIndex];
		Mat foreground = getForeground(frame, ps.background);
		Mat markerMask = getMarkerMask(foreground);

		auto &currMarkers = states[frameIndex].markers;

		Mat display;
		Mat controlBar;

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

				Mat frame = ps.frames[frameIndex - historyLength + t];
				Mat foreground = getForeground(frame, ps.background);
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
				cv::absdiff(ps.frames[frameIndex], ps.frames[frameIndex - 1], display);
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
						else if (frameIndex + 1 != ps.frames.size() &&
							syncedStatesCount == ps.frames.size() &&
							states[frameIndex + 1].ids.find(tracker.id) == states[frameIndex + 1].ids.end())
						{
							color = Scalar{40, 39, 214};
						}
						else if (tracker.prob != 1)
						{
							color = Scalar{255, 0, 255};
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

		{
			controlBar.create(30, display.cols, display.type());
			controlBar.setTo(Scalar{51, 52, 52});
			controlBar.row(0).setTo(Scalar{30, 30, 30});
			controlBar.row(controlBar.rows - 1).setTo(Scalar{30, 30, 30});

			int totalFrameCount = std::max(ps.totalFrameCount, (int)ps.frames.size());

			int loadedFramesEnd = controlBar.cols * ps.frames.size() / totalFrameCount;
			cv::rectangle(controlBar, Point2i{loadedFramesEnd, 0}, Point2i{controlBar.cols, controlBar.rows}, Scalar{30, 30, 30}, -1);

			int processedStatesEnd = controlBar.cols * syncedStatesCount / totalFrameCount;
			cv::rectangle(controlBar, Point2i{0, controlBar.rows - 5}, Point2i{processedStatesEnd, controlBar.rows}, Scalar{44, 160, 44}, -1);

			int sliderLength = controlBar.cols;
			int hangleWidth = 10;
			int sliderActiveRange = sliderLength - hangleWidth + 1;

			int sliderPos = sliderActiveRange * frameIndex / (std::max(totalFrameCount, 2) - 1);

			Rect hangleRect{sliderPos, 0, hangleWidth, controlBar.rows};
			cv::rectangle(controlBar, hangleRect, Scalar{147, 150, 150}, -1);
			cv::rectangle(controlBar, hangleRect, Scalar{30, 30, 30}, 1);

			cv::vconcat(display, controlBar, display);
		}

		cv::imshow("w", display);
		cv::setWindowTitle("w", "Active trackers: " + std::to_string(states[frameIndex].ids.size()) + " Frame: " + std::to_string(frameIndex));

		if (playing && (frameIndex != syncedStatesCount && frameIndex != syncedStatesCount - 1))
			frameIndex = std::min(frameIndex + 1, (int)ps.frames.size() - 1);

		int pressedKey = cv::waitKey(30);

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
			frameIndex = (int)ps.frames.size() - 1;
			frameIndex = std::max(frameIndex, 0);
			break;
		case KEY_ARROW_RIGHT:
			frameIndex = std::min(frameIndex + (playing ? 4 : 1), (int)ps.frames.size() - 1);
			frameIndex = std::max(frameIndex, 0);
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
}


int main(int argc, char *argv[])
{
	VideoCapture vid;

	//
	// Get video name from the command line and load it
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

	//
	// Load video in one thread, process in second thread,
	// draw a GUI in a third thread.
	//

	ProcessState ps;

	std::thread videoLoadingThread(loadVideoGrayscale, &ps, &vid);
	std::thread processingThread(buildFrameStates, &ps, 100);
	std::thread guiThread(drawGUI, &ps);

	guiThread.join();
	videoLoadingThread.join();
	processingThread.join();

	return 0;
}
