#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/opencv.hpp>

using std::vector;
using std::string;
using std::map;
using std::pair;

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

struct frame_state
{
	vector<Point2f> markers;
	vector<vector<int>> markerIds;
	map<int, int> ids;
};

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

	auto getForeground = [] (const Mat &frame, const Mat &background) -> Mat
	{
		Mat foreground;
		cv::absdiff(frame, background, foreground);
		return foreground;
	};

	auto getMarkerMask = [] (const Mat &foreground) -> Mat
	{
		Mat markerMask = foreground >= 30;

		cv::erode(markerMask,
			markerMask,
			cv::getStructuringElement(cv::MORPH_ELLIPSE, Size {3, 3}),
			cv::Point {-1, -1},
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

	auto getFrameMarkers = [] (const Mat &markerMask) -> vector<Point2f>
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
				markers.push_back(Point2f {(float)centroids(r, 0), (float)centroids(r, 1)});
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

	Mat background = videoMedian(vector<Mat>{frames.begin(), frames.begin() + std::max(frames.size(), (size_t)20)});

	int frameCount = frames.size();
	int frameIndex = 0;

	vector<frame_state> states(frames.size());
	int idCount = 0;

	for (int t = 0; t < frameCount; t++)
	{
		Mat foreground = getForeground(frames[t], background);
		Mat markerMask = getMarkerMask(foreground);

		vector<Point2f> currMarkers = getFrameMarkers(markerMask);
		vector<vector<int>> currMarkerIds(currMarkers.size());
		map<int, int> currIds;

		if (t > 0)
		{
			vector<Point2f> lastMarkers = states[t - 1].markers;
			vector<vector<int>> lastMarkerIds = states[t - 1].markerIds;
			map<int, int> lastIds = states[t - 1].ids;

			auto findClosest = [&currMarkers] (Point2f p, float distThreshold) -> vector<int>
			{
				float squareDistThreshold = distThreshold * distThreshold;

				vector<pair<float, int>> pointsByDistance;

				for (int i = 0; i < currMarkers.size(); i++)
				{
					float dx = p.x - currMarkers[i].x;
					float dy = p.y - currMarkers[i].y;

					float squareDist = dx * dx + dy * dy;

					if (squareDist < squareDistThreshold)
						pointsByDistance.push_back({squareDist, i});
				}

				std::sort(pointsByDistance.begin(), pointsByDistance.end());

				vector<int> points;

				for (auto p : pointsByDistance)
					points.push_back(p.second);

				return points;
			};

			for (int i = 0; i < lastMarkers.size(); i++)
			{
				if (lastMarkerIds[i].size() != 1)
					continue;

				vector<int> closest = findClosest(lastMarkers[i], 50);

				if (closest.size() > 0)
				{
					int lastId = lastMarkerIds[i][0];
					int newPoint = closest[0];
					currMarkerIds[newPoint].push_back(lastId);
					currIds[lastId] = newPoint;
				}
			}

			for (int i = 0; i < lastMarkers.size(); i++)
			{
				if (lastMarkerIds[i].size() == 1)
					continue;

				vector<int> closest = findClosest(lastMarkers[i], 50);

				vector<int> closestFull;
				vector<int> closestEmpty;

				for (auto p : closest)
				{
					if (currMarkerIds[p].size() > 0)
						closestFull.push_back(p);
					else
						closestEmpty.push_back(p);
				}

				if (closest.size() == 0)
					continue;

				for (int k = 0; k < lastMarkerIds[i].size(); k++)
				{
					int lastId = lastMarkerIds[i][k];
					int newPoint;

					if (closestEmpty.size() != 0)
						newPoint = closestEmpty[k % closestEmpty.size()];
					else
						newPoint = closestFull[0];

					currMarkerIds[newPoint].push_back(lastId);
					currIds[lastId] = newPoint;
				}
			}
		}

		for (int i = 0; i < currMarkerIds.size(); i++)
		{
			if (currMarkerIds[i].size() == 0)
			{
				currMarkerIds[i].push_back(idCount);
				currIds[idCount] = i;

				idCount++;
			}
		}

		CV_Assert(currMarkers.size() == currMarkerIds.size());

		states[t].markers = currMarkers;
		states[t].markerIds = currMarkerIds;
		states[t].ids = currIds;

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

	bool playing = false;
	int mode = 1;

	while (true)
	{
		Mat frame = frames[frameIndex];
		Mat foreground = getForeground(frame, background);
		Mat markerMask = getMarkerMask(foreground);

		auto &currMarkers = states[frameIndex].markers;
		auto &currMarkerIds = states[frameIndex].markerIds;
		auto &currIds = states[frameIndex].ids;

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
			for (int i = 0; i < currMarkers.size(); i++)
			{
				Point2f pos = currMarkers[i];
				int id = states[frameIndex].markerIds[i][0];
				Scalar c = currMarkerIds[i].size() > 1 ? Scalar {0, 255, 255} : Scalar {0, 0, 255};
				cv::circle(display, pos, 4, c, -1);
				cv::rectangle(display, pos + Point2f {5, -7}, pos + Point2f {15, 7}, Scalar {255, 255, 255}, -1);
				cv::putText(display, std::to_string(id),
					pos + Point2f{5, 5}, cv::FONT_HERSHEY_PLAIN, 1, Scalar{0});
			}
			break;
		case 4:
			cv::cvtColor(frame, display, cv::COLOR_GRAY2BGR);
			for (int i = 0; i < currMarkers.size(); i++)
			{
				Point2f pos = currMarkers[i];
				int id = states[frameIndex].markerIds[i][0];
				Scalar c = currMarkerIds[i].size() > 1 ? Scalar {0, 255, 255} : Scalar {0, 0, 255};
				cv::circle(display, pos, 4, c, -1);
				cv::rectangle(display, pos + Point2f {5, -7}, pos + Point2f {15, 7}, Scalar {255, 255, 255}, -1);
				cv::putText(display, std::to_string(id),
					pos + Point2f {5, 5}, cv::FONT_HERSHEY_PLAIN, 1, Scalar {0});
			}
			break;
		case 5:
			if (frameIndex > 0)
				cv::absdiff(frames[frameIndex], frames[frameIndex - 1], display);
			else
				display = frame.clone();
			break;
		default:
			display = frame.clone();
		}

		{
			if (display.type() == CV_8U)
				cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);

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

			for (pair<int, int> p : currIds)
				activeIds.push_back(p.first);

			cv::setWindowTitle("w", "Active bugs: " + std::to_string(activeIds.size()));

			for (int k = frameIndex - 1; k >= 0 && frameIndex - k < 30; k--)
			{
				if (activeIds.size() == 0)
					break;

				vector<int> newIds;

				for (int id : activeIds)
				{
					if (states[k].ids.find(id) != states[k].ids.end())
					{
						cv::line(display, states[k + 1].markers[states[k + 1].ids[id]], states[k].markers[states[k].ids[id]], colorPalette[id % colorPalette.size()]);
						newIds.push_back(id);
					}
				}

				activeIds = newIds;
			}
		}

		cv::imshow("w", display);

		if (playing)
			frameIndex = std::min(frameIndex + 1, frameCount - 1);

		int pressedKey = cv::waitKey(playing ? 29 : 0);

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
			mode = std::max(mode - 1, 0);
			break;
		default:
			break;
		}

		if ('0' <= pressedKey && pressedKey <= '9')
			mode = pressedKey - '0';
	}

	return 0;
}
