#include "Utility.h"
#include <tuple>
#include <numeric>
#include <fstream>

#include "NvInferRuntime.h"
#include "inferenceservice.h"

bool MatUtility::CenterCrop(const cv::Mat& in, cv::Mat& out, int outWidth, int outHeight)
{
	auto [w, h] = std::make_tuple(in.cols, in.rows);
	auto left = static_cast<int>((w - outWidth) / 2);
	auto right = static_cast<int>((w + outWidth) / 2);
	auto top = static_cast<int>((h - outHeight) / 2);
	auto bottom = static_cast<int>((h + outHeight) / 2);
	
	out = std::move(cv::Mat(in, cv::Rect(cv::Point(left, top), cv::Point(right, bottom))));

	return true;
}

bool MatUtility::ResizeWithAspectRatio(const cv::Mat& in, cv::Mat& out, int outWidth, int outHeight, float scale)
{
	auto [w_img, h_img] = std::make_tuple(in.cols, in.rows);
	auto newHeight = static_cast<int>((100.0 * outHeight / scale));
	auto newWidth = static_cast<int>((100.0 * outWidth / scale));

	int w, h = 0;
	if (h_img > w_img)
	{
		w = newWidth;
		h = static_cast<int>((newHeight * static_cast<double>(h_img) / w_img));
	}
	else
	{
		h = newHeight;
		w = static_cast<int>((newWidth * static_cast<double>(w_img) / h_img));
	}

	cv::resize(in, out, cv::Size(w, h));

	return true;
}

bool MatUtility::LetterBoxResize(const cv::Mat& _imgSource, cv::Size _newSize, cv::Mat& _imgOut, int _stride, bool _auto, bool _scaleFill, bool _scaleUp, bool _center)
{
	//https://github.com/autogyro/yolo-V8/blob/main/ultralytics/yolo/data/augment.py#L767

	const auto& imw = _imgSource.rows;
	const auto& imh = _imgSource.cols;


	// Calculate the scaling factor
	float r = std::min((float)_newSize.height / imh,
		(float)_newSize.width / imw);

	if (!_scaleUp)
	{
		r = std::min(r, 1.0f);
	}

	float rW = r;
	float rH = r;

	int new_unpad_width = std::round(imw * r);
	int new_unpad_height = std::round(imh * r);

	int dw = _newSize.width - new_unpad_width;
	int dh = _newSize.height - new_unpad_height;

	if (_auto) {
		dw = dw % _stride;
		dh = dh % _stride;
	}
	else if (_scaleFill) {
		dw = 0;
		dh = 0;
		new_unpad_width = _newSize.width;
		new_unpad_height = _newSize.height;
	}

	int top = _center ? std::round(dh / 2.0f - 0.1f) : 0;
	int bottom = _center ? std::round(dh / 2.0f + 0.1f) : dh;
	int left = _center ? std::round(dw / 2.0f - 0.1f) : 0;
	int right = _center ? std::round(dw / 2.0f + 0.1f) : dw;

	if (imw != new_unpad_width || imh != new_unpad_height)
	{
		cv::resize(_imgSource, _imgOut, cv::Size(new_unpad_width, new_unpad_height), 0, 0, cv::INTER_LINEAR);
	}
	else
	{
		_imgOut = _imgSource.clone();
	}

	cv::copyMakeBorder(_imgOut, _imgOut, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

	return true;
}


bool MatUtility::ClassificationPreprocess(const cv::Mat& in, int modelInputSize, cv::cuda::GpuMat& out)
{
	//https://github.com/onnx/models/tree/main/validated/vision/classification/resnet

	cv::Mat img;
	MatUtility::ResizeWithAspectRatio(in, img, modelInputSize, modelInputSize);
	MatUtility::CenterCrop(img, img, modelInputSize, modelInputSize);

	// normalize
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);
	img -= cv::Scalar(0.485, 0.456, 0.406);
	img /= cv::Scalar(0.229, 0.224, 0.225);

	// convert to CxHxW layout
	cv::cuda::GpuMat gpuImg(img);
	const auto channelImg = gpuImg.channels();
	std::vector<cv::cuda::GpuMat> vectorCHW(channelImg);

	for (int i = 0; i < channelImg; ++i) {
		cv::cuda::createContinuous(gpuImg.size(), CV_32F, vectorCHW[i]);
	}

	cv::cuda::split(gpuImg, vectorCHW);

	try
	{
		// flatten data
		cv::cuda::GpuMat blue = std::move(vectorCHW[0].reshape(1, 1));
		cv::cuda::GpuMat green = std::move(vectorCHW[1].reshape(1, 1));
		cv::cuda::GpuMat red = std::move(vectorCHW[2].reshape(1, 1));

		auto wXh = gpuImg.rows * gpuImg.cols;

		// merge to one image
		cv::cuda::GpuMat gpuImgContinous;
		cv::cuda::createContinuous(1, channelImg * wXh, CV_32FC1, gpuImgContinous);

		cudaError_t cudaError = cudaError_t::cudaSuccess;
		cudaError = cudaMemcpy(gpuImgContinous.ptr<float>(0), red.ptr<float>(0), wXh * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaError = cudaMemcpy((gpuImgContinous.ptr<float>(0) + wXh), green.ptr<float>(0), wXh * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaError = cudaMemcpy((gpuImgContinous.ptr<float>(0) + 2 * wXh), blue.ptr<float>(0), wXh * sizeof(float), cudaMemcpyDeviceToDevice);

		
		out.release();
		out = std::move(gpuImgContinous);

	}
	catch (const std::exception&)
	{

	}
	catch (...)
	{
		return false;
	}

	return true;
}

bool MatUtility::DetectionPreprocess(const cv::Mat& in, int modelInputSize, cv::cuda::GpuMat& out)
{
	cv::Mat img;
	LetterBoxResize(in, cv::Size(modelInputSize, modelInputSize), img);

	// normalize
	img.convertTo(img, CV_32FC3, 1.0 / 255.0);

	// convert to CxHxW layout
	cv::cuda::GpuMat gpuImg(img);
	const auto channelImg = gpuImg.channels();
	std::vector<cv::cuda::GpuMat> vectorCHW(channelImg);

	for (int i = 0; i < channelImg; i++) {
		cv::cuda::createContinuous(gpuImg.size(), CV_32F, vectorCHW[i]);
	}

	cv::cuda::split(gpuImg, vectorCHW);

	try
	{
		// flatten data
		cv::cuda::GpuMat blue = std::move(vectorCHW[0].reshape(1, 1));
		cv::cuda::GpuMat green = std::move(vectorCHW[1].reshape(1, 1));
		cv::cuda::GpuMat red = std::move(vectorCHW[2].reshape(1, 1));

		auto wXh = gpuImg.rows * gpuImg.cols;

		// merge to one image
		cv::cuda::GpuMat gpuImgContinous;
		cv::cuda::createContinuous(1, channelImg * wXh, CV_32FC1, gpuImgContinous);

		cudaError_t cudaError = cudaError_t::cudaSuccess;
		cudaError = cudaMemcpy(gpuImgContinous.ptr<float>(0), red.ptr<float>(0), wXh * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaError = cudaMemcpy((gpuImgContinous.ptr<float>(0) + wXh), green.ptr<float>(0), wXh * sizeof(float), cudaMemcpyDeviceToDevice);
		cudaError = cudaMemcpy((gpuImgContinous.ptr<float>(0) + 2 * wXh), blue.ptr<float>(0), wXh * sizeof(float), cudaMemcpyDeviceToDevice);


		out.release();
		out = std::move(gpuImgContinous);

	}
	catch (const std::exception&)
	{

	}
	catch (...)
	{
		return false;
	}


	return true;
}

bool ModelUtility::ClassificationPostprocess(const std::vector<float>& input, std::vector<int>& out, int topK)
{
	if (input.empty()) return false;

	if (input.size() < topK) return false;

	// fill data from 0
	std::vector<int> indices(input.size());
	std::iota(indices.begin(), indices.end(), 0);

	// just need to get top k element
	std::partial_sort(indices.begin(),
		indices.begin() + topK,
		indices.end(),
		[&input](int a, int b) {
			return input[a] > input[b];
		});

	out.clear();
	out.reserve(topK);
	std::copy(indices.begin(), indices.begin() + topK, std::back_inserter(out));
		
	return true;
}

bool ModelUtility::DetectionPostprocess(const std::vector<DetectionInferenceResult>& input, std::vector<int>& out, float threshold)
{
	if (input.empty()) return false;

	std::vector<int> sortedIndices(input.size());
	std::iota(sortedIndices.begin(), sortedIndices.end(), 0);

	// sort according to score
	std::sort(sortedIndices.begin(), sortedIndices.end(), [&input](int a, int b) {
		return input[a].GetScore() > input[b].GetScore();
		});

	std::vector<bool> suppressed(input.size(), false);
	auto sortedIndiceSize = sortedIndices.size();

	for (int i = 0; i < sortedIndiceSize; i++)
	{
		int current = sortedIndices[i];
		if (suppressed[current]) continue;

		out.emplace_back(current);

		// compare current with the rest
		for (int j = i; j < sortedIndiceSize; j++)
		{
			int other = sortedIndices[j];

			if (other == current || suppressed[other]) continue;

			auto x1Inter = std::max(input[current].GetCoordinate().left(), input[other].GetCoordinate().left());
			auto y1Inter = std::max(input[current].GetCoordinate().top(), input[other].GetCoordinate().top());
			auto x2Inter = std::max(input[other].GetCoordinate().right(), input[other].GetCoordinate().right());
			auto y2Inter = std::max(input[other].GetCoordinate().bottom(), input[other].GetCoordinate().bottom());
			auto areaInter = std::max(0, x2Inter - x1Inter) * std::max(0, y2Inter - y1Inter);

			auto areaCurrent = input[current].GetCoordinate().width() * input[current].GetCoordinate().height();
			auto areaOther = input[other].GetCoordinate().width() * input[other].GetCoordinate().height();

			auto iou = (1.0f * areaInter) / (areaCurrent + areaOther - areaInter);
			if (iou > threshold)
				suppressed[other] = true;
		}
	}

	return true;
}

std::vector<std::string> ModelUtility::LoadLabelData(const std::string& path)
{

	std::vector<std::string> labelInfor;
	std::ifstream f(path);

	if (f.is_open()) {

		std::string line;
		while (std::getline(f, line)) {
			labelInfor.emplace_back(line);
		}

		f.close();
	}

	return labelInfor;
}
