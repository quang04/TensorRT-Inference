#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include "appvm.h"

AppVM theApp;

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    // binding to UI
    qmlRegisterSingletonInstance<AppVM>("app.theApp", 1, 0, "TheApp", &theApp);

    QQmlApplicationEngine engine;
    QObject::connect(
        &engine,
        &QQmlApplicationEngine::objectCreationFailed,
        &app,
        []() { QCoreApplication::exit(-1); },
        Qt::QueuedConnection);
    engine.loadFromModule("App", "Main");

    return app.exec();
}
