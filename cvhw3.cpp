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

Mat videoMedia(const vector<Mat> &images)
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

int main(int argc, char *argv[])
{
	VideoCapture vid;

	//
	// Load the video
	//

	{
		string fileName;

		if (argc == 2)
		{
			fileName = string{argv[1]};
		}

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
			{
				fileName += ".mp4";
			}

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

	vector<Mat> frames;

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

	Mat background = videoMedia(frames);

	int frameCount = frames.size();
	int frameIndex = 0;

	const int ARROW_LEFT = 65361;
	const int ARROW_UP = 65362;
	const int ARROW_RIGHT = 65363;
	const int ARROW_DOWN = 65364;
	const int KEY_HOME = 65360;
	const int KEY_END = 65367;

	bool playing = false;
	int mode = 1;

	while (true)
	{
		Mat frame = frames[frameIndex];

		Mat foreground;
		cv::absdiff(frame, background, foreground);

		Mat markerMask = foreground >= 30;

		cv::erode(markerMask,
				markerMask,
				cv::getStructuringElement(cv::MORPH_ELLIPSE, Size{3, 3}),
				cv::Point{-1, -1},
				1);

		/*
		cv::dilate(markerMask,
				markerMask,
				cv::getStructuringElement(cv::MORPH_ELLIPSE, Size{3, 3}),
				cv::Point{-1, -1},
				1);
				*/

		/*
		cv::erode(markerMask,
				markerMask,
				cv::getStructuringElement(cv::MORPH_ELLIPSE, Size{3, 3}),
				cv::Point{-1, -1},
				1);

		cv::morphologyEx(markerMask,
				markerMask,
				cv::MORPH_CLOSE,
				cv::getStructuringElement(cv::MORPH_ELLIPSE, Size{15, 15}));
				*/

		/*
		cv::morphologyEx(markerMask,
				markerMask,
				cv::MORPH_OPEN,
				cv::getStructuringElement(cv::MORPH_ELLIPSE, Size{7, 7}));
				*/

		vector<Point2f> markers;

		{
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
		}

		if (mode == 1)
		{
			imshow("w", frame);
		}
		else if (mode == 2)
		{
			imshow("w", foreground);
		}
		else if (mode == 3)
		{
			Mat display;
			cv::cvtColor(markerMask, display, cv::COLOR_GRAY2BGR);

			for (auto m : markers)
			{
				cv::circle(display, m, 4, Scalar{0, 0, 255}, -1);
			}
			imshow("w", display);
		}
		else if (mode == 4)
		{
			Mat display;
			cv::cvtColor(frame, display, cv::COLOR_GRAY2BGR);

			for (auto m : markers)
			{
				cv::circle(display, m, 4, Scalar{0, 0, 255}, -1);
			}
			imshow("w", display);
		}
		else if (mode == 5)
		{
			if (frameIndex > 0)
			{
				Mat movement;
				cv::absdiff(frames[frameIndex], frames[frameIndex - 1], movement);
				imshow("w", movement);
			}
		}

		if (playing)
		{
			if (frameIndex < frameCount - 1)
			{
				frameIndex++;
			}
		}

		int pressedKey = cv::waitKey(playing ? 30 : 0);

		if (pressedKey == ' ')
		{
			playing = !playing;
		}
		else if (pressedKey == ARROW_RIGHT)
		{
			int step = playing ? 4 : 1;

			if (frameIndex < frameCount - step)
			{
				frameIndex += step;
			}
		}
		else if (pressedKey == ARROW_LEFT)
		{
			int step = playing ? 4 : 1;

			if (frameIndex - step >= 0)
			{
				frameIndex -= step;
			}
		}
		else if (pressedKey == KEY_HOME)
		{
			frameIndex = 0;
		}
		else if (pressedKey == KEY_END)
		{
			frameIndex = frameCount - 1;
		}
		else if (pressedKey == 'q')
		{
			break;
		}
		else if (pressedKey >='0' && pressedKey <= '9')
		{
			mode = pressedKey - '0';
		}
		else if (pressedKey == ARROW_DOWN)
		{
			mode++;
		}
		else if (pressedKey == ARROW_UP)
		{
			if (mode > 1)
			{
				mode--;
			}
		}
	}

	return 0;
}
