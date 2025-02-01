#ifndef INFERENCESERVICE_H
#define INFERENCESERVICE_H

#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <algorithm>

#include <QtCore/qobject.h>
#include <QMutexLocker>
#include <QWaitCondition>
#include <QRect>

#include "NvInferRuntime.h"
#include "appdef.h"
#include "opencv2/dnn.hpp"
#include "utility.h"


enum class LAYER_TYPE : uint8_t
{
    NONE = 0,
    INPUT,
    OUTPUT,

};

// in onnx use int64 while tensorRT use int32
template<typename T>
class LayerParam
{
public:
    LayerParam()
        : name("")
        , type(LAYER_TYPE::NONE)
        , totalShape(0)
    {
    }

    LayerParam(const std::string& _name,
        const std::vector<T>& _shape,
        LAYER_TYPE _type)
        : name(_name)
        , shape(_shape)
        , type(_type)
        , totalShape(MatUtility::CalculateDim(_shape.data(), _shape.size()))
    {
    }

    LayerParam(const LayerParam& other)
        : name(other.name)
        , shape(other.shape)
        , type(other.type)
        , totalShape(other.totalShape)
    {
    }

    LayerParam& operator=(const LayerParam& other)
    {
        if (this != &other)
        {
            name = other.name;
            shape = other.shape;
            type = other.type;
            totalShape = other.totalShape;
        }

        return *this;
    }

    LayerParam& operator=(LayerParam&& other)
    {
        name = std::move(other.name);
        shape = std::move(other.shape);
        type = std::move(other.type);
        totalShape = std::move(other.totalShape);
        return *this;
    }

    virtual ~LayerParam()
    {
    }

public:
    inline std::string GetLayerName() const { return name; }
    constexpr std::vector<T> GetShape() const { return shape; }
    constexpr uint64_t GetTotalShape() const { return totalShape; }
    constexpr LAYER_TYPE GetLayerType() const { return type; }

    virtual inline std::optional<T> const GetNumBatch() = 0;
    virtual inline std::optional<T> const GetImageSize() = 0;
    virtual inline std::optional<T> const GetImageChannel() = 0;

protected:
    // name of layer
    std::string name;

    // shape
    std::vector<T> shape;

    // total shape
    uint64_t totalShape;

    // indicate input/output
    LAYER_TYPE type;
};

//shape classification will be [N C H W]
template<typename T>
class ClassificationLayerParam : public LayerParam<T>
{
public:
    ClassificationLayerParam()
        : LayerParam<T>()
    {
    }

    ClassificationLayerParam(const std::string& _name,
        const std::vector<T>& _shape,
        LAYER_TYPE _type)
        : LayerParam<T>(_name, _shape, _type)
    {
    }

    ClassificationLayerParam(const ClassificationLayerParam& other)
        : LayerParam<T>(other)
    {
    }

    ClassificationLayerParam& operator=(const ClassificationLayerParam& other)
    {
        LayerParam<T>::operator=(other);

        return *this;
    }
    ClassificationLayerParam& operator=(ClassificationLayerParam&& other)
    {

        LayerParam<T>::operator=(std::move(other));

        return *this;
    }


public:
    inline std::optional<T> const GetNumBatch()
    {
        // only use as input layer
        if (this->type != LAYER_TYPE::INPUT) return std::nullopt;

        // avoid out of range
        if (this->shape.size() < 4) return std::nullopt;

        return this->shape[0];
    };
    inline std::optional<T> const GetImageSize() {

        // only use as input layer
        if (this->type != LAYER_TYPE::INPUT) return std::nullopt;

        // avoid out of range
        if (this->shape.size() < 4) return std::nullopt;

        return this->shape[2];
    }
    inline std::optional<T> const GetImageChannel() {
  
        // only use as input layer
        if (this->type != LAYER_TYPE::INPUT) return std::nullopt;

        // avoid out of range
        if (this->shape.size() < 4) return std::nullopt;

        return this->shape[1];
    }
};

//shape detection will be [N C W H]
template<typename T>
class DetectionLayerParam : public LayerParam<T>
{
public:
    DetectionLayerParam()
        : LayerParam<T>()
    {
    }

    DetectionLayerParam(const std::string& _name,
        const std::vector<T>& _shape,
        LAYER_TYPE _type)
        : LayerParam<T>(_name, _shape, _type)
    {
    }

    DetectionLayerParam(const DetectionLayerParam& other)
        : LayerParam<T>(other)
    {
    }

    DetectionLayerParam& operator=(const DetectionLayerParam& other)
    {
        LayerParam<T>::operator=(other);
        return *this;
    }

    DetectionLayerParam& operator=(DetectionLayerParam&& other)
    {
        LayerParam<T>::operator=(std::move(other));
        return *this;
    }
public:
    inline std::optional<T> const GetNumBatch() {

        // only use as input layer
        if (this->type != LAYER_TYPE::INPUT) return std::nullopt;

        // avoid out of range
        if (this->shape.size() < 4) return std::nullopt;

        return this->shape[0];
    };
    inline std::optional<T> const GetImageSize() {

        // only use as input layer
        if (this->type != LAYER_TYPE::INPUT) return std::nullopt;

        // avoid out of range
        if (this->shape.size() < 4) return std::nullopt;

        return this->shape[2];
    }
    inline std::optional<T> const GetImageChannel() {

        // only use as input layer
        if (this->type != LAYER_TYPE::INPUT) return std::nullopt;

        // avoid out of range
        if (this->shape.size() < 4) return std::nullopt;

        return this->shape[1];
    }
};

class ModelInferenceInformation
{
protected:
    ModelInferenceInformation()
        : m_confidenceThreshold(0.0f)
    {
    }
    ModelInferenceInformation(float _confidenceScore, const std::vector<std::string>& _labelInfor)
        : m_confidenceThreshold(_confidenceScore)
        , m_vectorLabelInfor(_labelInfor)
    {
    }

    ModelInferenceInformation(const ModelInferenceInformation& other)
    {
        m_confidenceThreshold = other.m_confidenceThreshold;
        m_vectorLabelInfor = other.m_vectorLabelInfor;
    }

    ModelInferenceInformation& operator=(const ModelInferenceInformation& other)
    {
        m_confidenceThreshold = other.m_confidenceThreshold;
        m_vectorLabelInfor = other.m_vectorLabelInfor;
        return *this;
    }
    virtual ~ModelInferenceInformation()
    {
    }
public:
    constexpr float GetConfidenceThreshold() const { return m_confidenceThreshold; }

    // expensive operation
    inline  std::vector<std::string> GetLabel() const { return m_vectorLabelInfor;}
    
protected:
    float m_confidenceThreshold;
    std::vector<std::string> m_vectorLabelInfor;
};


class ClassificationInferenceInformation : public ModelInferenceInformation
{
public:

    ClassificationInferenceInformation()
        : ModelInferenceInformation()
    {
    }

    ClassificationInferenceInformation(float _confidenceScore, const std::vector<std::string>& _labelInfor)
        : ModelInferenceInformation(_confidenceScore, _labelInfor)
    {
    }


    ClassificationInferenceInformation(const ClassificationInferenceInformation& other)
        : ModelInferenceInformation(other)
    {
    }

    ClassificationInferenceInformation& operator=(const ClassificationInferenceInformation& other)
    {
        m_confidenceThreshold = other.m_confidenceThreshold;
        m_vectorLabelInfor = other.m_vectorLabelInfor;

        return *this;
    }

    ClassificationInferenceInformation& operator=(ClassificationInferenceInformation&& other) noexcept
    {
        m_confidenceThreshold = std::move(other.m_confidenceThreshold);
        m_vectorLabelInfor = std::move(other.m_vectorLabelInfor);
        return *this;
    }

    ~ClassificationInferenceInformation(){}

};

class DetectionInferenceInformation: public ModelInferenceInformation
{
public:

    DetectionInferenceInformation()
        : ModelInferenceInformation()
        , m_iouThreshold(0.0f)
    {
    }

    DetectionInferenceInformation(float _confThreshold,
        float _iouThreshold,
        const std::vector<std::string>& _labelInfor)
        : ModelInferenceInformation(_confThreshold, _labelInfor)
        , m_iouThreshold(_iouThreshold)
    {
    }

    DetectionInferenceInformation(const DetectionInferenceInformation& other)
        : ModelInferenceInformation(other)
        , m_iouThreshold(other.m_iouThreshold)
    {
    }

    DetectionInferenceInformation& operator=(const DetectionInferenceInformation& other)
    {

        m_confidenceThreshold = other.m_confidenceThreshold;
        m_iouThreshold = other.m_iouThreshold;
        m_vectorLabelInfor = other.m_vectorLabelInfor;

        return *this;
    }

    DetectionInferenceInformation& operator=(DetectionInferenceInformation&& other) noexcept
    {

        m_confidenceThreshold = std::move(other.m_confidenceThreshold);
        m_iouThreshold = std::move(other.m_iouThreshold);
        m_vectorLabelInfor = std::move(other.m_vectorLabelInfor);

        return *this;
    }
    ~DetectionInferenceInformation(){}
public:
    constexpr float GetIouThreshold() const { return m_iouThreshold;}
private:
    float m_iouThreshold;
};

class InferenceResult
{
protected:
    InferenceResult()
        : m_classID(0)
        , m_score(0.0f)
    {
    }

    InferenceResult(int _classID, float _score)
        : m_classID(_classID)
        , m_score(_score)
    {
    }

    virtual ~InferenceResult(){}

public:
    constexpr int GetClassID() const { return m_classID;}
    constexpr float GetScore() const { return m_score;}
 
protected:
    int m_classID;
    float m_score;
};

class ClassificationInferenceResult : public InferenceResult
{
public:
    ClassificationInferenceResult()
    {
    }

    ClassificationInferenceResult(int _classID, float _score)
        : InferenceResult(_classID, _score)
    {
    }

    ~ClassificationInferenceResult(){}

};

class DetectionInferenceResult: public InferenceResult {
public:
    DetectionInferenceResult()
    {
    }
    DetectionInferenceResult(int _classID, float _score, QRect _coordinate)
        : InferenceResult(_classID, _score)
        , m_rectCoordinate(_coordinate)
    {
    }
    ~DetectionInferenceResult() {}
public:
    constexpr QRect GetCoordinate() const { return m_rectCoordinate;}
protected:
    QRect m_rectCoordinate;
};

typedef LayerParam<int64_t> OnnxLayerParam;
typedef ClassificationLayerParam<int64_t> OnnxClassificationLayerParam;
typedef DetectionLayerParam<int64_t> OnnxDetectionLayerParam;

typedef LayerParam<int32_t> TensorRTLayerParam;
typedef ClassificationLayerParam<int32_t> TensorRTClassificationLayerParam;
typedef DetectionLayerParam<int32_t> TensorRTDetectionLayerParam;


template<typename ModelInferenceInforType,
    typename InferenceResultType, 
    typename TensorRTLayerParamType>
class TRTService : public QObject
{
protected:
    TRTService(){}
    virtual ~TRTService(){}

protected:
    // for using main purpose
    std::vector<InferenceResultType> m_vectorResult;
    std::vector<TensorRTLayerParamType> m_vectorModelLayerInputInfor;
    std::vector<TensorRTLayerParamType> m_vectorModelLayerOutputInfor;
    ModelInferenceInforType m_modelInferenceInfor;
    std::weak_ptr<nvinfer1::IExecutionContext> m_ptrContext;

    // the ready buffer image that already goes through preprocess
    cv::cuda::GpuMat m_imageInput;

    // the orignal size of image and image when loading from external
    cv::Size m_sizeImageOriginal;
    cv::Mat m_imageOriginal;
public:
    inline std::vector<InferenceResultType> GetInferenceResult() const { return m_vectorResult; };
    inline ModelInferenceInforType GetModelInferenceInfor() const { return m_modelInferenceInfor; };
};

#endif // INFERENCESERVICE_H
