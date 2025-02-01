#ifndef APPDEF_H
#define APPDEF_H

#define MAX_MODEL_SUPPORT 5
#define WARM_UP_TIMES 5

#define CLASSIFY_THRESHOLD 0.5

#define DETECTION_THREASHOLD 0.7
#define DETECTION_IOU_THRESHOLD 0.9

enum class RUN_COMMAND : uint8_t
{
    INFER,
    WARM_UP,
    EXIT,
    WAIT
};

enum MODE : uint8_t
{
    CLASSIFICATION,
    DETECTION,
    ABNORMAL,
};

#endif // APPDEF_H
