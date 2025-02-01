#include "threadservice.h"

#if defined(Q_OS_WIN)
#include <windows.h>
#elif defined Q_OS_LINUX
#include <pthread.h>
#include <sched.h>
#else
#endif // Q_OS_WIN

ThreadService::ThreadService(MODE mode, int modelID, int cpuID, int gpuID, QObject *parent)
    : QThread{parent}
    , m_modelID(modelID)
    , m_gpuID(gpuID)
    , m_mode(mode)
    , m_cpuID(cpuID)
    , m_actionCommand(RUN_COMMAND::WAIT)
{

#if defined(Q_OS_WIN)
    int affinityMask = 1 << cpuID;
    auto h = this->currentThreadId();
    SetThreadAffinityMask(h, affinityMask);
#elif defined(Q_OS_LINUX)
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuID, &cpuset);
    pthread_t pthread = pthread_self();
    pthread_setaffinity_np(pthread, sizeof(cpu_set_t), &cpuset);
#else
#endif
    cudaSetDevice(gpuID);
    cudaStreamCreate(&m_cudaStream);
}

ThreadService::~ThreadService()
{
    cudaStreamDestroy(m_cudaStream);
}

bool ThreadService::DoWarmUp()
{
    for (int i = 0; i < WARM_UP_TIMES; i++)
    {
        DoInfer();
    }

    return true;
}


