#ifndef THREADSERVICE_H
#define THREADSERVICE_H

#include <QThread>
#include <mutex>
#include <condition_variable>

#include "appdef.h"
#include "NvInferRuntime.h"
#include "opencv2/opencv.hpp"

// do to QObject not support template so need create threadservice
class ThreadService : public QThread
{
    Q_OBJECT
public:
    explicit ThreadService(MODE mode, int modelID, int cpuID, int gpuID = 0, QObject *parent = nullptr);
    virtual ~ThreadService();

    inline void SendWarmUpCommand()
    {
        std::lock_guard<std::mutex> l(m_mutex);
        m_actionCommand = RUN_COMMAND::WARM_UP;
        m_waitCondtion.notify_one();
    }
    inline void SendInferCommand()
    {
        std::lock_guard<std::mutex> l(m_mutex);
        m_actionCommand = RUN_COMMAND::INFER;
        m_waitCondtion.notify_one();
    }
    inline void SendExitCommand()
    {
        std::lock_guard<std::mutex> l(m_mutex);
        m_actionCommand = RUN_COMMAND::EXIT;
        m_waitCondtion.notify_one();
    }


public:
    virtual bool SupplyImageInput(const cv::Mat& input) = 0;
protected:
    virtual bool DoInfer() = 0;
    bool DoWarmUp();
protected:
    // for run/pause thread
    RUN_COMMAND m_actionCommand;
    std::mutex m_mutex;
    std::condition_variable m_waitCondtion;

    //
    cudaStream_t m_cudaStream;
public:
    // for tracking model only
    const MODE m_mode;
    const int m_modelID;
    const int m_cpuID;
    const int m_gpuID;

signals:
    // signal when run inference complete
    void infereneCompletedSignal(int modelID);
};

#endif // THREADSERVICE_H
