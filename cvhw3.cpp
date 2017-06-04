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

using cv::Scalar;
using cv::InputArray;
using cv::OutputArray;
using cv::Point2f;
using cv::Point2i;
using cv::Rect;
using cv::Size;
using cv::String;
using cv::VideoCapture;

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

struct Edge
{
	int from;
	int to;
	Point2f dir;
	float weight;

	Edge(int from, int to, Point2f dir, float weight = 1)
		: from{from}, to{to}, dir{dir}, weight{weight}
	{ }
};

struct Tracker
{
	int id;
	float prob;
	Point2f dir;

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

	// How the markers from the last frame moved
	multimap<int, Edge> mov;
	multimap<int, Edge> imov;

	//multimap<int, Tracker> trackers;
	vector<vector<Tracker>> trackers;
	map<int, int> ids;
};

void drawStateTransitions(Mat &on, const FrameState& first, const FrameState& second, float scalingFactor = 1)
{
	for (pair<int, Edge> transition : second.mov)
	{
		Edge e = transition.second;
		cv::arrowedLine(on, first.markers[e.from] * scalingFactor, second.markers[e.to] * scalingFactor, Scalar{0, 255, 0}, std::floor(scalingFactor));
	}
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

	auto getFrameMarkers = [](const Mat &markerMask) -> vector<Point2f>
	{
		vector<Point2f> markers;

		Mat labels;
		Mat stats;
		Mat1d centroids;

		cv::connectedComponentsWithStats(
			markerMask,
			labels,
			stats,
			centroids);

		// NOTE: label 0 is the background. We ignore it here.

		for (int r = 1; r < centroids.rows; r++)
		{
			if (stats.at<int>(r, cv::CC_STAT_AREA) > 20)
			{
				markers.push_back(Point2f{(float)centroids(r, 0), (float)centroids(r, 1)});
			}
		}

		return markers;
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
			frames.push_back(frame);
			std::cout << "Loaded frame " << i++ << std::endl;
		}
	}

	Mat background = videoMedian(vector<Mat>{frames.begin(), frames.begin() + std::min(frames.size(), (size_t)100)});

	int frameCount = frames.size();
	int frameIndex = 0;

	vector<FrameState> states(frames.size());
	int idCount = 0;

	for (int t = 0; t < frameCount; t++)
	{
		//
		// Extract foreground and markers
		//

		Mat foreground = getForeground(frames[t], background);
		Mat markerMask = getMarkerMask(foreground);

		vector<Point2f> currMarkers = getFrameMarkers(markerMask);
		vector<Point2f> lastMarkers = t > 0 ? states[t - 1].markers : vector<Point2f>{};

		vector<vector<Tracker>> emptyTrackers;

		auto &currTrackers = states[t].trackers;
		auto &lastTrackers = t > 0 ? states[t - 1].trackers : emptyTrackers;

		//
		// Create the transition graph
		//

		vector<std::tuple<float, int, int>> distList;

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
					distList.push_back({squareDist, i, k});
				}
			}
		}

		std::sort(distList.begin(), distList.end());

		vector<bool> lastConnected(lastMarkers.size(), false);
		vector<bool> currConnected(currMarkers.size(), false);

		for (int iter = 0; iter < 2; iter++)
		{
			for (tuple<float, int, int> d : distList)
			{
				float distance;
				int i, k;

				std::tie(distance, i, k) = d;

				if ((iter == 0 && !lastConnected[i]) ||
					(iter == 1 && !currConnected[k]))
				{
					Edge edge{i, k, currMarkers[k] - lastMarkers[i]};
					states[t].mov.insert({i, edge});

					Edge inverseEdge{k, i, lastMarkers[i] - currMarkers[k]};
					states[t].imov.insert({k, inverseEdge});

					lastConnected[i] = true;
					currConnected[k] = true;
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

			vector<Edge> edges;

			for (auto e = edgeRange.first; e != edgeRange.second; e++)
			{
				edges.push_back(e->second);
			}

			for (Tracker tracker : lastTrackers[i])
			{
				// An actual tracker and not just a probability
				bool sureTracker = states[t - 1].ids.find(tracker.id) != states[t - 1].ids.end();

				for (Edge e : edges)
				{
					float newProb = tracker.prob / edges.size();

					newProb *= 0.97f;

					auto it = std::find_if(currTrackers[e.to].begin(), currTrackers[e.to].end(), [tracker](Tracker other) { return other.id == tracker.id; });

					if (newProb >= 0.05)
					{
						if (it != currTrackers[e.to].end())
						{
							it->prob += newProb;
						}
						else
						{
							Tracker newTracker{tracker.id, newProb, tracker.dir};
							if (sureTracker)
							{
								newTracker.dir = 0.4 * newTracker.dir + 0.6 * e.dir;
								newTracker.dir /= std::max(std::max(std::abs(newTracker.dir.x), std::abs(newTracker.dir.y)), 0.1f);
							}
							currTrackers[e.to].push_back(newTracker);
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
						Edge edge = p->second;
						int targetMarker = edge.to;

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

		for (auto t : allTrackers)
		{
			float prob;
			int marker;
			Tracker tracker;

			std::tie(prob, marker, tracker) = t;

			if (std::find(usedIds.begin(), usedIds.end(), tracker.id) == usedIds.end())
				newTrackers.push_back({marker, tracker});
		}

		for (auto &v : currTrackers)
			v.resize(0);

		for (pair<int, Tracker> t : newTrackers)
		{
			currTrackers[t.first].push_back(t.second);
		}

		//
		// Assign ids to empty markers
		//

		for (int i = 0; i < (int)currTrackers.size(); i++)
		{
			if (currTrackers[i].empty())
			{
				int newId = idCount++;
				currTrackers[i].push_back(Tracker{newId});
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
	int mode = 5;

	bool qShowMarkers = true;
	bool qShowTransitions = true;
	bool qShowIds = true;
	bool qShowTrails = true;
	bool qShowMarkerIds = false;
	bool qShowVelocities = false;

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
						Point2f from = states[k + 1].markers[states[k + 1].ids[id]];
						Point2f to = states[k].markers[states[k].ids[id]];
						Scalar color = colorPalette[id % colorPalette.size()];
						cv::line(display, from * scalingFactor, to * scalingFactor, color, std::floor(scalingFactor));
						newIds.push_back(id);
					}
				}

				activeIds = newIds;
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
		if (qShowIds)
		{
			for (auto p : states[frameIndex].ids)
			{
				int id = p.first;
				Point2f position = states[frameIndex].markers[p.second];
				cv::putText(display, std::to_string(id), position * scalingFactor, cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar{0, 0, 0}, 3);
				cv::putText(display, std::to_string(id), position * scalingFactor, cv::FONT_HERSHEY_SIMPLEX, 0.5, Scalar{255, 255, 255});
			}
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
		default:
			break;
		}

		if ('0' <= pressedKey && pressedKey <= '9')
			mode = pressedKey - '0';
	}

	return 0;
}
