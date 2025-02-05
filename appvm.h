#ifndef APPVM_H
#define APPVM_H

#include <QObject>
#include <array>
#include <appdef.h>
#include <memory>
#include <mutex>
#include <condition_variable>

#include "NvInferRuntime.h"
#include "onnxruntime_cxx_api.h"
#include "inferencethread.h"



struct TensorRTCoreObj
{
    std::shared_ptr<nvinfer1::ICudaEngine> engine;
    std::shared_ptr<nvinfer1::IExecutionContext> context;

    void ResetResource()
    {
        context.reset();
        engine.reset();
    }
};


class AppVM : public QObject
{
    Q_OBJECT
public:
    explicit AppVM(QObject *parent = nullptr);
    ~AppVM();

private:
    // TensorRT
    std::array<TensorRTCoreObj, MAX_MODEL_SUPPORT> m_arrayTensorRTObjClassification;
    std::array<TensorRTCoreObj, MAX_MODEL_SUPPORT> m_arrayTensorRTObjDetection;

    std::array<std::unique_ptr<ClassificationInferenceThread>, MAX_MODEL_SUPPORT> m_arrayModelThreadManagerClassification;
    std::array<std::unique_ptr<DetectionInferenceThread>, MAX_MODEL_SUPPORT> m_arrayModelThreadManagerDetection;

    // Onnx object
    // use for to load model onnx to get needed param for converting tensorRT
    std::unique_ptr<Ort::SessionOptions> m_pSessioOptionsOnnx;
    Ort::Env m_envOnnx;



private:
    void convertModelToTensorRT(const QString& modelOnnxPath, MODE modelMode);
    static void readEngineFileAsBuffer(const std::string& modelTensorRTPath, std::vector<char>& buffer);
    // Onnx
    void createOnnxSession();
    void loadModelOnnx(const QString& modelOnnxPath, MODE mode, std::vector<std::unique_ptr<OnnxLayerParam>>& modelInformations);

    
    bool loadModelTensorRTClassification(const std::string& modelTensorRTPath, int modelID);
    bool loadModelTensorRTDetection(const std::string& modelTensorRTPath, int modelID);

    void inferenceAll();
private:
    bool m_bIsConvertingModel;
    const bool m_bIsUsedFP16;

    bool m_bIsRunningInference;
    std::condition_variable m_waitConditionInferenceAllModelComplete;
    std::mutex m_mutexInferenceAllModelComplete;

    //blocking function
    void waitInferenceAllModelComplete();
public slots:
    void loadModelClassificationAllSlot(const QString& modelPath);
    void releaseModelClassificationAllSlot();

    void loadModelDetectionAllSlot(const QString& modelPath);
    void releaseModelDetectionAllSlot();

    void convertModelClassificationToTensorRTSlot(const QString& modelOnnxPath);
    void convertModelDetectionToTensorRTSlot(const QString& modelOnnxPath);

    void loadImageClassificationSlots(const QString& imgPath);
    void loadImageDetectionSlots(const QString& imgPath);
    void infereceAllSlots();

    void inferenceCompletedSlot(int modelID);
signals:
      void addDataDebugLog(const QString& mess);
};


// the only one
extern AppVM theApp;

#endif // APPVM_H
