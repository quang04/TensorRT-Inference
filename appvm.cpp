#include <QDebug>
#include <QDir>
#include <QUrl>


#include <fstream>
#include <format>
#include <chrono>

#include <NvInfer.h>
#include <NvOnnxConfig.h>
#include <NvOnnxParser.h>

#include "appvm.h"
#include "tensorRTLogger.h"



TensorRTLogger theTensorRTLogger{ TensorRTLogger ::Severity::kINFO};

AppVM::AppVM(QObject *parent)
    : QObject{parent}
    , m_bIsConvertingModel(false)
    , m_bIsUsedFP16(true)
    , m_envOnnx(ORT_LOGGING_LEVEL_INFO, "App")
    , m_bIsRunningInference(false)
{

    createOnnxSession();


    // create classification, detection thread
    // default using at gpu0
    int cpuID = 0;


    for (int i = 0; i < MAX_MODEL_SUPPORT; i++, cpuID++)
    {
        m_arrayModelThreadManagerClassification[i] = std::make_unique<ClassificationInferenceThread>(i, cpuID, 0, this);
        m_arrayModelThreadManagerClassification[i].get()->start();

        // bind to inference complete slot
        connect(m_arrayModelThreadManagerClassification[i].get(), &ThreadService::infereneCompletedSignal, this, &AppVM::inferenceCompletedSlot);
    }

    for (int j = 0; j < MAX_MODEL_SUPPORT; j++, cpuID++)
    {
        m_arrayModelThreadManagerDetection[j] = std::make_unique<DetectionInferenceThread>(j, cpuID, 0, this);
        m_arrayModelThreadManagerDetection[j].get()->start();

        // bind to inference complete slot
        connect(m_arrayModelThreadManagerDetection[j].get(), &ThreadService::infereneCompletedSignal, this, &AppVM::inferenceCompletedSlot);
    }

}

AppVM::~AppVM()
{
    // wait all thread exit
    for (const auto& t : m_arrayModelThreadManagerClassification)
    {
        t.get()->SendExitCommand();
        t.get()->wait();
    }
    for (const auto& t : m_arrayModelThreadManagerDetection)
    {
        t.get()->SendExitCommand();
        t.get()->wait();
    }

    // need to manual reset this, if not will cause sw crash when exit
    m_pSessioOptionsOnnx.reset();
    m_pSessioOptionsOnnx.release();
}



void AppVM::loadModelClassificationAllSlot(const QString &modelPath)
{
    QString actualPath;
    const QUrl url(modelPath);
    if (url.isLocalFile()) {
        actualPath = QDir::toNativeSeparators(url.toLocalFile());
    }
    else {
        actualPath = modelPath;
    }


    emit addDataDebugLog("Load Model Classification");

    // load for all thread
    auto const& s = m_arrayModelThreadManagerClassification.size();
    for (int i=0; i<s; i++)
    {
        loadModelTensorRTClassification(actualPath.toStdString(), i);
    }

    emit addDataDebugLog("Load Model Classification Done");
}

void AppVM::waitInferenceAllModelComplete()
{
    std::unique_lock lk(m_mutexInferenceAllModelComplete);
    m_waitConditionInferenceAllModelComplete.wait(lk, [this]() {
        return !m_bIsRunningInference;
        });
}

void AppVM::releaseModelClassificationAllSlot()
{
    for (auto& item : m_arrayTensorRTObjClassification)
        item.ResetResource();
}

void AppVM::loadModelDetectionAllSlot(const QString &modelPath)
{
    QString actualPath;
    const QUrl url(modelPath);
    if (url.isLocalFile()) {
        actualPath = QDir::toNativeSeparators(url.toLocalFile());
    }
    else {
        actualPath = modelPath;
    }


    emit addDataDebugLog("Load Model Detection");

    // load for all thread
    auto const& s = m_arrayModelThreadManagerDetection.size();
    for (int i = 0; i < s; i++)
    {
        loadModelTensorRTDetection(actualPath.toStdString(), i);
    }

    emit addDataDebugLog("Load Model Detection Done");

}

void AppVM::releaseModelDetectionAllSlot()
{
    for (auto& item : m_arrayTensorRTObjDetection)
        item.ResetResource();
}

void AppVM::convertModelClassificationToTensorRTSlot(const QString &modelOnnxPath)
{
    // convert from onnx to tensorRT model
    if(m_bIsConvertingModel) return;

    std::thread launchConvert(&AppVM::convertModelToTensorRT, this, modelOnnxPath, MODE::CLASSIFICATION);
    launchConvert.detach();
}

void AppVM::convertModelDetectionToTensorRTSlot(const QString& modelOnnxPath)
{
    // convert from onnx to tensorRT model
    if (m_bIsConvertingModel) return;

    std::thread launchConvert(&AppVM::convertModelToTensorRT, this, modelOnnxPath, MODE::DETECTION);
    launchConvert.detach();
}

void AppVM::convertModelToTensorRT(const QString &modelOnnxPath, MODE modelMode)
{

    m_bIsConvertingModel = true;

    emit addDataDebugLog("convertModelToTensorRT start");

    QString actualPath;
    const QUrl url(modelOnnxPath);
    if (url.isLocalFile()) {
        actualPath = QDir::toNativeSeparators(url.toLocalFile());

    }
    else {
        actualPath = modelOnnxPath;
    }

    // load model as onnx to get model infor
    std::vector<std::unique_ptr<OnnxLayerParam>> infors;
    loadModelOnnx(actualPath, modelMode, infors);

    if (infors.empty())
    {
        emit addDataDebugLog("convertModelToTensorRT failed");
        m_bIsConvertingModel = false;

        return;
    }

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(theTensorRTLogger.GetTrtLogger()));
    if (builder == nullptr)
    {
        emit addDataDebugLog("convertModelToTensorRT::createInferBuilder failed");
        m_bIsConvertingModel = false;
        return;
    }

    //using explicit batch rather than implicit batch
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (network.get() == nullptr)
    {
        emit addDataDebugLog("convertModelToTensorRT::createNetworkV2 failed");
        m_bIsConvertingModel = false;
        return;
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, theTensorRTLogger.GetTrtLogger()));
    if (parser.get() == nullptr)
    {
        emit addDataDebugLog("convertModelToTensorRT::createParser failed");
        m_bIsConvertingModel = false;
        return;
    }

    auto modelParser = parser->parseFromFile(actualPath.toStdString().c_str(), static_cast<int>(theTensorRTLogger.GetReportSeverity()));
    if (modelParser == false) {

        for (int32_t i = 0; i < parser->getNbErrors(); ++i)
        {
            qDebug() << parser->getError(i)->desc();
        }

        emit addDataDebugLog("convertModelToTensorRT::parseFromFile failed");
        m_bIsConvertingModel = false;
        return;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (config.get() == nullptr)
    {
        emit addDataDebugLog("convertModelToTensorRT::createParser failed");
        m_bIsConvertingModel = false;
        return;
    }

    auto profile = builder->createOptimizationProfile();


    // fix input tensor name and image size
    // may need modify at here for flex batch size
    for (const auto& infor : infors)
    {
        auto name = infor.get()->GetLayerName();
        auto batch = infor.get()->GetNumBatch();
        auto imgSize = infor.get()->GetImageSize();
        auto imgChannel = infor.get()->GetImageChannel();

        if (!batch.has_value() || !imgSize.has_value() || !imgChannel.has_value()) continue;

        // fix 1 batch
        batch = 1;

        // depend on framework need to set correct dimension
        switch (modelMode)
        {
        case CLASSIFICATION:
            profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(*batch, *imgChannel, *imgSize, *imgSize));
            profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(*batch, *imgChannel, *imgSize, *imgSize));
            profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(*batch, *imgChannel, *imgSize, *imgSize));
            break;
        case DETECTION:
            profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(*batch, *imgChannel, *imgSize, *imgSize));
            profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(*batch, *imgChannel, *imgSize, *imgSize));
            profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(*batch, *imgChannel, *imgSize, *imgSize));
            break;
        default:
            break;
        }
    }


    config->addOptimizationProfile(profile);

    // 4GB workspace for aggressive optimize
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 4U << 30);


    // using time cache
    std::unique_ptr<nvinfer1::ITimingCache> timingCache(config->createTimingCache(nullptr, false));
    config->setTimingCache(*(timingCache.get()), false);

    // default using fp16
    if (m_bIsUsedFP16)
    {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
        config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);

        emit addDataDebugLog("fp16 using");

    }

    std::unique_ptr<nvinfer1::IHostMemory> serializeModel(builder->buildSerializedNetwork(*network, *config));
    if (serializeModel.get() == nullptr)
    {
        emit addDataDebugLog("convertModelToTensorRT::buildSerializedNetwork failed");
        m_bIsConvertingModel = false;
        return;
    }

    // save data to a file
    QFileInfo fileInfo(actualPath);

    auto fName = fileInfo.baseName();
    const static QString tensorRTExetension = ".engine";

    if (m_bIsUsedFP16)
        fName = QString("%1_fp16%2").arg(fName).arg(tensorRTExetension);
    else
        fName = QString("%1%2").arg(fName).arg(tensorRTExetension);


   
    std::ofstream outFile(fName.toStdString(), std::ios::binary);
    if (!outFile) {
        emit addDataDebugLog("Error: Could not open file");
        m_bIsConvertingModel = false;
        return;
    }

    // Get the pointer to the memory and its size.
    const void* data = serializeModel->data();
    size_t size = serializeModel->size();

    outFile.write(static_cast<const char*>(data), size);
    outFile.close();

    emit addDataDebugLog("convertModelToTensorRT success");

    m_bIsConvertingModel = false;
}

void AppVM::readEngineFileAsBuffer(const std::string& modelTensorRTPath, std::vector<char>& buffer)
{
    std::ifstream engineFile(modelTensorRTPath, std::ios::binary);
    if (engineFile.is_open() == false) return;

    // calculate file size
    engineFile.seekg(0, std::ifstream::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    buffer.resize(fileSize);
    engineFile.read(buffer.data(), fileSize);

    engineFile.close();
}

void AppVM::createOnnxSession()
{
    m_pSessioOptionsOnnx.reset();
    m_pSessioOptionsOnnx = std::make_unique<Ort::SessionOptions>();

    m_pSessioOptionsOnnx->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    OrtCUDAProviderOptionsV2* cudaOptions = nullptr;

    Ort::GetApi().CreateCUDAProviderOptions(&cudaOptions);

    if (cudaOptions == nullptr)
    {
        // error when creating
        qDebug() << "Error when create cuda provider";
    }

    // default using gpu 0
    std::vector<const char*> keys{ "device_id" };
    std::vector<const char*> values{ "0" };
    Ort::GetApi().UpdateCUDAProviderOptions(cudaOptions, keys.data(), values.data(), keys.size());

    m_pSessioOptionsOnnx->AppendExecutionProvider_CUDA_V2(*cudaOptions);

    // Release CUDA provider options after appending
    Ort::GetApi().ReleaseCUDAProviderOptions(cudaOptions);

    // Check if CUDA execution provider was enabled successfully
    bool isCudaEnabled = false;
    auto providers = Ort::GetAvailableProviders();
    for (const auto& provider : providers) {
        if (provider == "CUDAExecutionProvider") {
            isCudaEnabled = true;
            break;
        }
    }

    // throw error at starting
    if (!isCudaEnabled) {
        throw std::runtime_error("CUDA Execution Provider not enabled");
    }
}

void AppVM::loadModelOnnx(const QString &modelOnnxPath, MODE mode, std::vector<std::unique_ptr<OnnxLayerParam>>& modelInformations)
{
    modelInformations.clear();
    modelInformations.shrink_to_fit();

    std::unique_ptr<Ort::Session> pSession(new Ort::Session(m_envOnnx, modelOnnxPath.toStdWString().c_str(), *m_pSessioOptionsOnnx.get()));
    auto inputCount = pSession->GetInputCount();

    modelInformations.reserve(inputCount);

    Ort::AllocatorWithDefaultOptions ortAllocator;

    for(int i=0; i<inputCount; i++)
    {
        char* inputName = pSession->GetInputName(i, ortAllocator);

        Ort::TypeInfo info = pSession->GetInputTypeInfo(i);
        auto tensorInfo = info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> dims = tensorInfo.GetShape();
        ONNXTensorElementDataType type = tensorInfo.GetElementType();
        
        // make sure not out of range
        assert(dims.size() == 4);

        switch (mode)
        {
        case CLASSIFICATION:
            modelInformations.emplace_back(std::make_unique<OnnxClassificationLayerParam>(std::string(inputName), dims, LAYER_TYPE::INPUT));
            break;
        case DETECTION:
            modelInformations.emplace_back(std::make_unique<OnnxDetectionLayerParam>(std::string(inputName), dims, LAYER_TYPE::INPUT));
            break;
        case ABNORMAL:
            throw std::exception("Not support");
            break;
        default:
            break;
        }

        ortAllocator.Free(inputName);
    }

}

bool AppVM::loadModelTensorRTClassification(const std::string& modelTensorRTPath, int modelID)
{

    if (modelID >= m_arrayTensorRTObjClassification.size()) return false;

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(theTensorRTLogger.GetTrtLogger()));
    if (runtime.get() == nullptr) return false;

    std::vector<char> modelBuffer;
    readEngineFileAsBuffer(modelTensorRTPath, modelBuffer);
    if (modelBuffer.empty()) return false;

    // release all resource
    m_arrayTensorRTObjClassification[modelID].ResetResource();

    auto& e = m_arrayTensorRTObjClassification[modelID].engine;
    auto& c = m_arrayTensorRTObjClassification[modelID].context;
    auto& t = m_arrayModelThreadManagerClassification[modelID];

    e = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelBuffer.data(), modelBuffer.size()));
    if (e == nullptr) return false;

    c = std::shared_ptr<nvinfer1::IExecutionContext>(e->createExecutionContext());
    if (c == nullptr) return false;

    auto totalIO = e->getNbIOTensors();

    // accumulate layer infor
    std::vector<TensorRTClassificationLayerParam> layerInforInput;
    std::vector<TensorRTClassificationLayerParam> layerInforOutput;

    for (int i = 0; i < totalIO; i++)
    {
        auto name = e->getIOTensorName(i);
        auto dim = e->getTensorShape(name);
        auto mode = static_cast<LAYER_TYPE>(e->getTensorIOMode(name));
        
        TensorRTClassificationLayerParam infor(std::string(name), std::vector<int32_t>{dim.d, dim.d + dim.MAX_DIMS}, mode);

        if (mode == LAYER_TYPE::INPUT) layerInforInput.emplace_back(infor);
        else if (mode == LAYER_TYPE::OUTPUT) layerInforOutput.emplace_back(infor);
    }

    // passing layer infor to thread
    t.get()->SupplyLayerInformation(layerInforInput, layerInforOutput);

    // load label
    const std::string& labelPath = std::format("{}.txt", modelTensorRTPath);
    
    // passing inference information to thread
    t.get()->SupplyInferenceInformation(ClassificationInferenceInformation(CLASSIFY_THRESHOLD, ModelUtility::LoadLabelData(labelPath)));

    // passing tensorRT obj to thread
    t.get()->SuppyTensorRTCoreObj(c);

    // create dummy image for running warmup
    cv::Mat dummyImg(256, 256, CV_8UC3, cv::Scalar(1,1,1));
    t.get()->SupplyImageInput(dummyImg);

    // do warmup
    t.get()->SendWarmUpCommand();
    return true;
}

bool AppVM::loadModelTensorRTDetection(const std::string& modelTensorRTPath, int modelID)
{
    if (modelID >= m_arrayTensorRTObjDetection.size()) return false;

    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(theTensorRTLogger.GetTrtLogger()));
    if (runtime.get() == nullptr) return false;

    std::vector<char> modelBuffer;
    readEngineFileAsBuffer(modelTensorRTPath, modelBuffer);
    if (modelBuffer.empty()) return false;

    // release all resource
    m_arrayTensorRTObjDetection[modelID].ResetResource();

    auto& e = m_arrayTensorRTObjDetection[modelID].engine;
    auto& c = m_arrayTensorRTObjDetection[modelID].context;
    auto& t = m_arrayModelThreadManagerDetection[modelID];

    e = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(modelBuffer.data(), modelBuffer.size()));
    if (e == nullptr) return false;

    c = std::shared_ptr<nvinfer1::IExecutionContext>(e->createExecutionContext());
    if (c == nullptr) return false;

    auto totalIO = e->getNbIOTensors();

    // accumulate layer infor
    std::vector<TensorRTDetectionLayerParam> layerInforInput;
    std::vector<TensorRTDetectionLayerParam> layerInforOutput;

    for (int i = 0; i < totalIO; i++)
    {
        auto name = e->getIOTensorName(i);
        auto dim = e->getTensorShape(name);
        auto mode = static_cast<LAYER_TYPE>(e->getTensorIOMode(name));

        TensorRTDetectionLayerParam infor(std::string(name), std::vector<int32_t>{dim.d, dim.d + dim.MAX_DIMS}, mode);

        if (mode == LAYER_TYPE::INPUT) layerInforInput.emplace_back(infor);
        else if (mode == LAYER_TYPE::OUTPUT) layerInforOutput.emplace_back(infor);
    }

    // passing layer infor to thread
    t.get()->SupplyLayerInformation(layerInforInput, layerInforOutput);

    // load label
    const std::string& labelPath = std::format("{}.txt", modelTensorRTPath);

    // passing inference information to thread
    t.get()->SupplyInferenceInformation(DetectionInferenceInformation(DETECTION_THREASHOLD, DETECTION_IOU_THRESHOLD, ModelUtility::LoadLabelData(labelPath)));

    // passing tensorRT obj to thread
    t.get()->SuppyTensorRTCoreObj(c);

    // create dummy image for running warmup
    cv::Mat dummyImg(1024, 1024, CV_8UC3, cv::Scalar(1, 1, 1));
    t.get()->SupplyImageInput(dummyImg);

    // do warmup
    t.get()->SendWarmUpCommand();
    return true;
}

void AppVM::inferenceAll()
{
    m_bIsRunningInference = true;

    auto start = std::chrono::steady_clock::now();

    for (const auto& item : m_arrayModelThreadManagerClassification)
    {
        item->SendInferCommand();
    }

    for (const auto& item : m_arrayModelThreadManagerDetection)
    {
        item->SendInferCommand();
    }

    //wait at here to make sure all model complete
    waitInferenceAllModelComplete();

    auto finish = std::chrono::steady_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<
        std::chrono::milliseconds>(finish - start).count();

    emit addDataDebugLog(QString("Inference Time: %1 ms").arg(elapsed_seconds));

 
}

void AppVM::loadImageClassificationSlots(const QString &imgPath)
{

    emit addDataDebugLog("Load Image Classification");

    QString actualPath;
    const QUrl url(imgPath);
    if (url.isLocalFile()) {
        actualPath = QDir::toNativeSeparators(url.toLocalFile());

    }
    else {
        actualPath = imgPath;
    }

    cv::Mat img = cv::imread(actualPath.toStdString());

    for (const auto& item : m_arrayModelThreadManagerClassification)
    {
        item->SupplyImageInput(img);
    }

    emit addDataDebugLog("Load Image Classification Done");
}

void AppVM::loadImageDetectionSlots(const QString& imgPath)
{
    emit addDataDebugLog("Load Image Detection");

    QString actualPath;
    const QUrl url(imgPath);
    if (url.isLocalFile()) {
        actualPath = QDir::toNativeSeparators(url.toLocalFile());

    }
    else {
        actualPath = imgPath;
    }

    cv::Mat img = cv::imread(actualPath.toStdString());

    // supply input for all thread
    for (const auto& item : m_arrayModelThreadManagerDetection)
    {
        item->SupplyImageInput(img);
    }

    emit addDataDebugLog("Load Image Detection Done");
}

void AppVM::infereceAllSlots()
{
    // avoid mutiple click
    if (m_bIsRunningInference) return;

    std::thread launchInferAll(&AppVM::inferenceAll, this);
    launchInferAll.detach();
}


void AppVM::inferenceCompletedSlot(int modelID)
{
    static int countModelComplete = 0;
    static int maxModelToWait = m_arrayModelThreadManagerClassification.size() + m_arrayModelThreadManagerDetection.size();

    if (++countModelComplete == maxModelToWait)
    {
        // reset for next counting
        countModelComplete = 0;
        
        // print out result from all thread

        emit addDataDebugLog("Classification Result");
        for (const auto& item : m_arrayModelThreadManagerClassification)
        {
            const auto& obj = item.get();
            const auto& res = obj->GetInferenceResult();
            const auto& label = obj->GetModelInferenceInfor().GetLabel();
           
            QString mess = QString("ModelID: %1 GPUID: %2").arg(obj->m_modelID).arg(obj->m_gpuID);
            emit addDataDebugLog(mess);
            for (const auto& r : res) {
                mess = QString("Class: %1, Score: %2")
                    .arg(label[r.GetClassID()].c_str())
                    .arg(r.GetScore());

                emit addDataDebugLog(mess);
            }
        }


        emit addDataDebugLog("Detection Result");
        for (const auto& item : m_arrayModelThreadManagerDetection)
        {
            const auto& obj = item.get();
            const auto& res = obj->GetInferenceResult();
            const auto& label = obj->GetModelInferenceInfor().GetLabel();
            

            QString mess = QString("ModelID: %1 GPUID: %2").arg(obj->m_modelID).arg(obj->m_gpuID);
            emit addDataDebugLog(mess);
            for (const auto& r : res) {

                const auto& box = r.GetCoordinate();
                mess = QString("Class: %1, Score: %2, X: %3, Y: %4, W: %5, H: %6")
                    .arg(label[r.GetClassID()].c_str())
                    .arg(r.GetScore())
                    .arg(box.x())
                    .arg(box.y())
                    .arg(box.width())
                    .arg(box.height());

                emit addDataDebugLog(mess);
            }
        }
 
        emit addDataDebugLog("Inference All Done");

        // notify wait condition
        {
            std::lock_guard lk(m_mutexInferenceAllModelComplete);
            m_bIsRunningInference = false;
        }

        m_waitConditionInferenceAllModelComplete.notify_one();
    }
}
