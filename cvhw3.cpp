#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

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

float r()
{
	static std::default_random_engine generator;
	static std::uniform_real_distribution<float> distribution(0, 1);

	return distribution(generator);
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

struct Pt
{
	Point2f p;
	Point2f v;

	Pt(Point2f p)
	{
		this->p = p;
		this->v = Point2f {(r() - 0.5f) * 20, (r() - 0.5f) * 20};
	}

	Pt(Point2f p, Point2f v)
	{
		this->p = p;
		this->v = v;
	}

	Pt(const Pt &other)
	{
		this->p = other.p;
		this->v = other.v;
	}

	Pt next()
	{
		return Pt(p + v, v + Point2f {(r() - 0.5f) * 10, (r() - 0.5f) * 10});
	}
};

template <typename T>
void increment(Mat& m, Pt p)
{
	int r = p.p.y;
	int c = p.p.x;

	if (r >= 0 && r < m.rows && c >= 0 && c < m.cols)
		m.at<T>(r, c)++;
};

template <typename T>
int sample(const Mat& m, Pt p)
{
	int r = p.p.y;
	int c = p.p.x;

	if (r >= 0 && r < m.rows && c >= 0 && c < m.cols)
		return m.at<T>(r, c);
	else
		return -1;
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
		Mat markerMask = (foreground - 20) * 2;

		return markerMask;
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

	Mat background = videoMedian(vector<Mat>{frames.begin(), frames.begin() + std::max(frames.size(), (size_t)10)});

	int frameCount = frames.size();
	int frameIndex = 0;
	int rows = frames[0].rows;
	int cols = frames[0].cols;

	//const float splitProb = 0.5;
	const float maxDensity = 4;

	vector<vector<Pt>> particles(frames.size());
	vector<Mat> heatMaps(frames.size());

	for (int t = 0; t < frameCount; t++)
	{
		Mat foreground = getForeground(frames[t], background);
		Mat markerMask = getMarkerMask(foreground);

		if (t > 1)
		{
			for (Pt p : particles[t - 1])
			{
				//do
				//{
				//	Pt next = p.next();

				//	if (sample<unsigned char>(markerMask, next))
				//	{
				//		particles[t].push_back(next);
				//	}
				//} while (r() < splitProb);

				for (int i = 0; i < 16; i++)
				{
					Pt next = p.next();

					if (sample<unsigned char>(markerMask, next))
					{
						particles[t].push_back(next);
					}
				}
			}
		}

		if (cv::countNonZero(markerMask) > 1 && t < 50)
		{
			int added = 0;

			while (added < 100)
			{
				Pt pt {Point2f{r() * cols, r() * rows}};

				if (sample<unsigned char>(markerMask, pt))
				{
					particles[t].push_back(pt);
					added++;
				}
			}
		}

		heatMaps[t] = Mat {frames[t].size(), CV_32S};
		heatMaps[t].setTo(0);

		for (Pt p : particles[t])
		{
			increment<int>(heatMaps[t], p);
		}

		particles[t].erase(
			std::remove_if(
				particles[t].begin(),
				particles[t].end(),
				[&heatMaps, t, maxDensity] (Pt p) {
					float density = sample<int>(heatMaps[t], p);

					if (density < 0)
						return true;

					float extra = std::max(density - maxDensity, 0.0f) / std::max(density, 1.0f);
					return r() < extra;
			}),
			particles[t].end());

		std::cout << particles[t].size() << std::endl;

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
	bool showingParticles = true;
	int mode = 1;

	while (true)
	{
		Mat frame = frames[frameIndex];
		Mat foreground = getForeground(frame, background);
		Mat markerMask = getMarkerMask(foreground);

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
			heatMaps[frameIndex].convertTo(display, CV_8U, 256 / maxDensity);
			break;
		default:
			display = frame.clone();
		}

		if (showingParticles)
		{
			if (display.type() == CV_8U)
				cv::cvtColor(display, display, cv::COLOR_GRAY2BGR);

			for (Pt p : particles[frameIndex])
			{
				cv::circle(display, p.p, 1, Scalar {0, 0, 255}, -1);
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
		case 'p':
			showingParticles = !showingParticles;
			break;
		default:
			break;
		}

		if ('0' <= pressedKey && pressedKey <= '9')
			mode = pressedKey - '0';
	}

	return 0;
}
