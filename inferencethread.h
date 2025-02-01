#ifndef INFERENCETHREAD_H
#define INFERENCETHREAD_H

#include <QObject>
#include <QThread>

#include "inferenceservice.h"
#include "threadservice.h"
#include <array>

template<typename ModelInferenceInforType,
    typename InferenceResultType,
    typename TensorRTLayerParamType>
class InferenceThreadBase : public ThreadService
    , public TRTService<ModelInferenceInforType, InferenceResultType, TensorRTLayerParamType>
{
protected:

    InferenceThreadBase(MODE mode, int modelID, int cpuID, int gpuID = 0, QObject* parent = nullptr)
        : ThreadService(mode, modelID, cpuID, gpuID, parent)
    {

    }

    ~InferenceThreadBase()
    {
  
    }

    void run() override
    {
        while(true)
        {
            std::unique_lock<std::mutex> locker(m_mutex);
            
            // wait at here till have valid command
            m_waitCondtion.wait(locker, [this] { return m_actionCommand != RUN_COMMAND::WAIT; });

            // command for exit when closing
            if(m_actionCommand == RUN_COMMAND::EXIT) break;

            //TODO: make sure inference context valid
            //if (this->m_ptrContext.expired()) continue;

 
            switch (m_actionCommand) {
            case RUN_COMMAND::WARM_UP:
                DoWarmUp();
                break;
            case RUN_COMMAND::INFER:
                DoInfer();
                //TODO: can remove save result for better performance
                SaveResult();
 
                emit infereneCompletedSignal(m_modelID);
 
                break;
            default:
                break;
            }

            // reset command for next time
            m_actionCommand = RUN_COMMAND::WAIT;
            QThread::msleep(1);
        }
    }

public:
    void SupplyLayerInformation(const std::vector<TensorRTLayerParamType>& input, const std::vector<TensorRTLayerParamType>& output)
    {

        this->m_vectorModelLayerInputInfor.clear();
        this->m_vectorModelLayerInputInfor.resize(input.size());
        std::copy(input.begin(), input.end(), this->m_vectorModelLayerInputInfor.begin());

        this->m_vectorModelLayerOutputInfor.clear();
        this->m_vectorModelLayerOutputInfor.resize(output.size());

        std::copy(output.begin(), output.end(), this->m_vectorModelLayerOutputInfor.begin());

    }

    inline void SupplyInferenceInformation(const ModelInferenceInforType& infor)
    {
        this->m_modelInferenceInfor = infor;
    }
    inline void SuppyTensorRTCoreObj(std::weak_ptr<nvinfer1::IExecutionContext> obj)
    {
        this->m_ptrContext = obj;
    }
protected:
    // for rechecking result
    virtual void SaveResult() = 0;

};


class ClassificationInferenceThread
    : public InferenceThreadBase<ClassificationInferenceInformation, ClassificationInferenceResult, TensorRTClassificationLayerParam>
{
public:
    ClassificationInferenceThread(int modelID, int cpuID, int gpuID, QObject *parent);
    ~ClassificationInferenceThread();

private:
    bool DoInfer() override;

    void SaveResult() override;
public:
    virtual bool SupplyImageInput(const cv::Mat& input) override;

};

class DetectionInferenceThread
    : public InferenceThreadBase<DetectionInferenceInformation, DetectionInferenceResult, TensorRTDetectionLayerParam>
{
public:
    DetectionInferenceThread(int modelID, int cpuID, int gpuID, QObject *parent);
    ~DetectionInferenceThread();

private:
    bool DoInfer() override;
    void SaveResult() override;
public:
    virtual bool SupplyImageInput(const cv::Mat& input) override;
};

#endif // INFERENCETHREAD_H

