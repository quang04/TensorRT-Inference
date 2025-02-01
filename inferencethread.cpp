#include "inferencethread.h"
#include <cmath>


ClassificationInferenceThread::ClassificationInferenceThread(int modelID, int cpuID, int gpuID, QObject *parent)
    : InferenceThreadBase(MODE::CLASSIFICATION, modelID, cpuID, gpuID, parent)
{
}

ClassificationInferenceThread::~ClassificationInferenceThread()
{
}




bool ClassificationInferenceThread::DoInfer()
{
    // make sure tensortRT object is valid
    auto context = this->m_ptrContext.lock();
    if (!context) return false;

    // clear result
    this->m_vectorResult.clear();

    // make sure input layer/output layer is valid
    if (this->m_vectorModelLayerInputInfor.empty() || this->m_vectorModelLayerOutputInfor.empty()) return false;

    if (m_imageInput.empty()) return false;


    const auto inputLayerSize = this->m_vectorModelLayerInputInfor.size();
    const auto outputLayerSize = this->m_vectorModelLayerOutputInfor.size();

    std::vector<void*> inputBuffer(inputLayerSize);
    std::vector<void*> outputBuffer(outputLayerSize);

    float* gpuData = m_imageInput.ptr<float>();
    
    // right now only have 1 input, 1 output
    for (int i = 0; i < inputLayerSize; i++)
    {
        cudaMallocAsync(&inputBuffer[i], this->m_vectorModelLayerInputInfor[i].GetTotalShape() * sizeof(float), m_cudaStream);
        cudaMemcpyAsync(inputBuffer[i], gpuData, this->m_vectorModelLayerInputInfor[i].GetTotalShape() * sizeof(float), cudaMemcpyDeviceToDevice, m_cudaStream);
        context->setTensorAddress(this->m_vectorModelLayerInputInfor[i].GetLayerName().c_str(), inputBuffer[i]);
    }

    for (int i = 0; i < outputLayerSize; i++)
    {
        cudaMallocAsync(&outputBuffer[i], this->m_vectorModelLayerOutputInfor[i].GetTotalShape() * sizeof(float), m_cudaStream);
        context->setTensorAddress(this->m_vectorModelLayerOutputInfor[i].GetLayerName().c_str(), outputBuffer[i]);

    }

    bool isOk = context->enqueueV3(m_cudaStream);
    if (!isOk)
    {
        for (int i = 0; i < inputLayerSize; i++)
        {
            cudaFreeAsync(inputBuffer[i], m_cudaStream);
        }

        for (int i = 0; i < outputLayerSize; i++)
        {
            cudaFreeAsync(outputBuffer[i], m_cudaStream);
        }

        return false;
    }

    // transfer result from device to host(gpu to cpu)
    std::vector<std::vector<float>> dataOutputs(outputLayerSize);
    for (int i = 0; i < outputLayerSize; i++)
    {
        dataOutputs[i].resize(this->m_vectorModelLayerOutputInfor[i].GetTotalShape());
        cudaMemcpyAsync(dataOutputs[i].data(),
            outputBuffer[i * this->m_vectorModelLayerOutputInfor[i].GetTotalShape() * sizeof(float)],
            this->m_vectorModelLayerOutputInfor[i].GetTotalShape() * sizeof(float),
            cudaMemcpyDeviceToHost,
            m_cudaStream);
    }

    cudaStreamSynchronize(m_cudaStream);

    // due to 1 output only take into account at index 0
    std::vector<int> selectedIndex;
    if (ModelUtility::ClassificationPostprocess(dataOutputs[0], selectedIndex))
    {   
        //do softmax to get actual score
        auto sum = std::accumulate(dataOutputs[0].begin(), dataOutputs[0].end(), 0.0f, [](float accumulator, float item) {
            return accumulator + std::expf(item);
            });

        for (const auto& item : selectedIndex)
        {
            m_vectorResult.emplace_back(item, (std::expf(dataOutputs[0][item])) / sum);
        }
    }


    for (int i = 0; i < inputLayerSize; i++)
    {
        cudaFreeAsync(inputBuffer[i], m_cudaStream);
    }

    for (int i = 0; i < outputLayerSize; i++)
    {
        cudaFreeAsync(outputBuffer[i], m_cudaStream);
    }


    return true;
}

void ClassificationInferenceThread::SaveResult()
{
    if (this->m_vectorResult.empty()) return;

    static const cv::Scalar& colorOverlay = cv::Scalar(0, 0, 255);
    static const int& thickness = 1;
    static const int& fontFace = cv::FONT_HERSHEY_SIMPLEX;
    static const double& fontScale = 1.0;
    static const int& xPos = 10;
    static const int& yPos = 30;


    auto label = std::move(this->m_modelInferenceInfor.GetLabel());

    int pos = yPos;
    for (auto const& item : this->m_vectorResult) {

        std::string str = std::format("{} {:.3f}", label[item.GetClassID()], item.GetScore());

        cv::putText(m_imageOriginal, str, cv::Point(xPos, pos),
            fontFace, fontScale, colorOverlay, thickness);

        pos += yPos;

    }

    std::string imgName = std::format("ClassificationResult_modelID{}_gpu{}.jpg", m_modelID, m_gpuID);
    cv::imwrite(imgName, m_imageOriginal);
    
}



bool ClassificationInferenceThread::SupplyImageInput(const cv::Mat& input)
{
    m_sizeImageOriginal = cv::Size(0, 0);

    // imread default format BGR
    cv::Mat cvImg(input.size(), input.type());
    if (input.channels() == 3)
        cv::cvtColor(input, cvImg, cv::COLOR_BGR2RGB);
    else
        cv::cvtColor(input, cvImg, cv::COLOR_GRAY2RGB);

    if (this->m_vectorModelLayerInputInfor.empty()) return false;

    // only have one input
    auto s = this->m_vectorModelLayerInputInfor[0].GetImageSize();

    if (!s.has_value()) return false;

    if (!MatUtility::ClassificationPreprocess(cvImg, *s, m_imageInput))
    {
        // error handler at here
        return false;
    }
    // store orignal size and image for later processing
    // expensive operation
    //TODO: if not need save result, can ignore store orignal image
    m_imageOriginal = input.clone();
    m_sizeImageOriginal = input.size();

    return true;
}


DetectionInferenceThread::DetectionInferenceThread(int modelID, int cpuID, int gpuID, QObject *parent)
    : InferenceThreadBase(MODE::DETECTION, modelID, cpuID, gpuID, parent)
{
   
}

DetectionInferenceThread::~DetectionInferenceThread()
{

}

bool DetectionInferenceThread::DoInfer()
{
    // make sure tensortRT object is valid
    auto context = this->m_ptrContext.lock();
    if (!context) return false;

    // clear result
    this->m_vectorResult.clear();

    // make sure input layer/output layer is valid
    if (this->m_vectorModelLayerInputInfor.empty() || this->m_vectorModelLayerOutputInfor.empty()) return false;

    if (m_imageInput.empty()) return false;

    const auto& inputLayerSize = this->m_vectorModelLayerInputInfor.size();
    const auto& outputLayerSize = this->m_vectorModelLayerOutputInfor.size();

    std::vector<void*> inputBuffer(inputLayerSize);
    std::vector<void*> outputBuffer(outputLayerSize);

    float* gpuData = m_imageInput.ptr<float>();

    // right now only have 1 input, 1 output
    for (int i = 0; i < inputLayerSize; i++)
    {
        cudaMallocAsync(&inputBuffer[i], this->m_vectorModelLayerInputInfor[i].GetTotalShape() * sizeof(float), m_cudaStream);
        cudaMemcpyAsync(inputBuffer[i], gpuData, this->m_vectorModelLayerInputInfor[i].GetTotalShape() * sizeof(float), cudaMemcpyDeviceToDevice, m_cudaStream);
        context->setTensorAddress(this->m_vectorModelLayerInputInfor[i].GetLayerName().c_str(), inputBuffer[i]);
    }

    for (int i = 0; i < outputLayerSize; i++)
    {
        cudaMallocAsync(&outputBuffer[i], this->m_vectorModelLayerOutputInfor[i].GetTotalShape() * sizeof(float), m_cudaStream);
        context->setTensorAddress(this->m_vectorModelLayerOutputInfor[i].GetLayerName().c_str(), outputBuffer[i]);

    }

    bool isOk = context->enqueueV3(m_cudaStream);
    if (!isOk)
    {
        for (auto& item : inputBuffer)
        {
            cudaFreeAsync(item, m_cudaStream);
        }

        for (auto& item : outputBuffer)
        {
            cudaFreeAsync(item, m_cudaStream);
        }

 
        return false;
    }

   
    // due to 1 output only, take into account at index 0
    // create data infor
    const int numOfFeature = m_vectorModelLayerOutputInfor[0].GetShape().at(1);
    const int numOfBox = m_vectorModelLayerOutputInfor[0].GetShape().at(2);

    std::vector<cv::Mat1f> dataOutputs(outputLayerSize);
    for (int i = 0; i < outputLayerSize; i++)
    {
        dataOutputs[i].create(numOfFeature, numOfBox);
        cudaMemcpyAsync(dataOutputs[i].data,
            outputBuffer[i * this->m_vectorModelLayerOutputInfor[i].GetTotalShape() * sizeof(float)],
            this->m_vectorModelLayerOutputInfor[i].GetTotalShape() * sizeof(float),
            cudaMemcpyDeviceToHost,
            m_cudaStream);

        //tranpose data for take advantage of cache locality
        cv::transpose(dataOutputs[i], dataOutputs[i]);

    }


    cudaStreamSynchronize(m_cudaStream);

    std::vector<DetectionInferenceResult> results;

    const auto& ratioX = (m_sizeImageOriginal.width * 1.0f) / (*m_vectorModelLayerInputInfor[0].GetImageSize());
    const auto& ratioY = (m_sizeImageOriginal.height * 1.0f) / (*m_vectorModelLayerInputInfor[0].GetImageSize());

    //// x y w h -> 4 field
    auto rwData = reinterpret_cast<float*>(dataOutputs[0].ptr());
    for (int i = 0; i < numOfBox; i++)
    {
        int pos = i * numOfFeature;

        std::vector<float> infor(numOfFeature);

        std::memcpy(infor.data(), rwData + pos, (numOfFeature) * sizeof(float));

        auto maxScoreIt = std::max_element(infor.begin() + 4, infor.end());

        if (*maxScoreIt < m_modelInferenceInfor.GetConfidenceThreshold()) continue;

        // due to use vector, so that can achieve 0(1) for std::distance
        auto index = std::distance(infor.begin() + 4, maxScoreIt);
        
        auto centerX = infor[0];
        auto centerY = infor[1];
        auto width = infor[2];
        auto height = infor[3];

        int l = static_cast<int>((centerX - 0.5 * width) * ratioX);
        int t = static_cast<int>((centerY - 0.5 * height) * ratioY);
        int w = width * ratioX;
        int h = height * ratioY;

        results.emplace_back(index, *maxScoreIt, QRect(l, t, w, h));
    }


    std::vector<int> selectedIndex;
    if (ModelUtility::DetectionPostprocess(results, selectedIndex, m_modelInferenceInfor.GetIouThreshold()))
    {
        this->m_vectorResult.reserve(selectedIndex.size());
        for (const auto& index : selectedIndex)
        {
            this->m_vectorResult.emplace_back(results[index]);
        }
    }


    for (auto& item : inputBuffer)
    {
        cudaFreeAsync(item, m_cudaStream);
    }

    for (auto& item : outputBuffer)
    {
        cudaFreeAsync(item, m_cudaStream);
    }


    return true;
}

void DetectionInferenceThread::SaveResult()
{
    if (this->m_vectorResult.empty()) return;

    static const cv::Scalar& colorOverlay = cv::Scalar(0, 0, 255);
    static const int& thickness = 1;
    static const int& marginTop = 5;
    static const int& fontFace = cv::FONT_HERSHEY_SIMPLEX;
    static const double& fontScale = 1.0;

    //draw box to overlay for rechecking
    auto label = std::move(this->m_modelInferenceInfor.GetLabel());
    for (auto const& item : this->m_vectorResult) {

        int xPos = item.GetCoordinate().left();
        int yPos = item.GetCoordinate().top();

        cv::putText(m_imageOriginal, label[item.GetClassID()], cv::Point(xPos, yPos - marginTop),
            fontFace, fontScale, colorOverlay, thickness);

        cv::rectangle(m_imageOriginal, cv::Rect(xPos, yPos, item.GetCoordinate().width(), item.GetCoordinate().height()), colorOverlay);
    }

    std::string imgName = std::format("DetectionResult_modelID{}_gpu{}.jpg", m_modelID, m_gpuID);
    cv::imwrite(imgName, m_imageOriginal);
    

}



bool DetectionInferenceThread::SupplyImageInput(const cv::Mat& input)
{
    m_sizeImageOriginal = cv::Size(0, 0);

    // imread default format BGR
    cv::Mat cvImg(input.size(), input.type());
    if(input.channels() == 3)
        cv::cvtColor(input, cvImg, cv::COLOR_BGR2RGB);
    else
        cv::cvtColor(input, cvImg, cv::COLOR_GRAY2RGB);

    if (this->m_vectorModelLayerInputInfor.empty()) return false;

    // only have one input
    auto s = this->m_vectorModelLayerInputInfor[0].GetImageSize();

    if (!s.has_value()) return false;

    if (!MatUtility::DetectionPreprocess(cvImg, *s, m_imageInput))
    {
        // error handler at here
        return false;
    }

    //TODO: if not need save result, can ignore store orignal image
    // store orignal size and image for later processing
    // expensive operation
    m_imageOriginal = input.clone();
    m_sizeImageOriginal = input.size();

    return true;
}


