#ifndef UTILITY_H
#define UTILITY_H

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudaimgproc.hpp"
#include "opencv2/cudawarping.hpp"

class DetectionInferenceResult;

class MatUtility
{
public:
	static bool CenterCrop(const cv::Mat& in,
		cv::Mat& out,
		int outWidth,
		int outHeight);

	static bool ResizeWithAspectRatio(const cv::Mat& in,
		cv::Mat& out,
		int outWidth,
		int outHeight,
		float scale = 87.5);

	static bool LetterBoxResize(const cv::Mat& _imgSource, cv::Size _newSize, cv::Mat& _imgOut, int _stride = 32, bool _auto = false, bool _scaleFill = false, bool _scaleUp = true, bool _center = true);

	static bool ClassificationPreprocess(const cv::Mat& in, int modelInputSize, cv::cuda::GpuMat& out);
	static bool DetectionPreprocess(const cv::Mat& in, int modelInputSize, cv::cuda::GpuMat& out);

	template<typename T>
	static uint64_t CalculateDim(const T* const input, int s)
	{
		uint64_t res = 1;
		for (int i = 0; i < s; i++)
		{
			if(input[i])
				res *= input[i];
		}

		return res;
	}
};

class ModelUtility
{
public:
	static bool ClassificationPostprocess(const std::vector<float>& input, std::vector<int>& out, int topK = 3);
	
	// do non-max suppression
	static bool DetectionPostprocess(const std::vector<DetectionInferenceResult>& input, std::vector<int>& out,
		float threshold = 0.5);

	static std::vector<std::string> LoadLabelData(const std::string& path);
};

#endif // !UTILITY_H