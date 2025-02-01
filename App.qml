import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Dialogs

import app.theApp 1.0

Item {

    readonly property int buttonHeight: 50
    readonly property int buttonWidth: 200
    readonly property int spacingElement: 10
    readonly property int debugLogItemHeight: 20
    readonly property int debugLogItemTextFont: 12

    readonly property string engineExtensionFile: "Model files (*.engine)"
    readonly property string onnxExtensionFile: "Model files (*.onnx)"
    readonly property string imageExtensionFile: "Image files (*.jpg *.png *.bmp)"

    readonly property int modeClassification: 0
    readonly property int modeDetection: 1

    readonly property int loadModelClassification: 0
    readonly property int loadModelDetection: 1
    readonly property int convertModelClassificationTensorRT: 2
    readonly property int convertModelDetectionTensorRT: 3

    Connections {
        target: TheApp
        function onAddDataDebugLog(mess) {
            addDataLog(mess)
        }
    }


    ListModel {
        id: listDebugModelInfor
        ListElement {
            infor: "Logging"
        }
    }

    function clearDataLog() {
        listDebugModelInfor.clear()
    }

    function addDataLog(mess) {
        listDebugModelInfor.append({
                                  "infor": mess
                              })
    }

    // File image dialog
    FileDialog {
        id: imageFileSelectDialog
        nameFilters: [imageExtensionFile]
        property int mode: modeClassification

        title: "Select a image file"
        fileMode: FileDialog.OpenFile
        onAccepted: {

            if(mode == modeClassification) {
                TheApp.loadImageClassificationSlots(selectedFile)
            } else if(mode == modeDetection) {
                TheApp.loadImageDetectionSlots(selectedFile)
            }
        }
        onRejected: {
        }
    }

    // File model dialog
    FileDialog {
        id: fileModelDialog
        property string extension: engineExtensionFile

        property int mode: loadModelClassification

        nameFilters: [extension]
        title: "Select a model file"
        fileMode: FileDialog.OpenFile

        onAccepted: {
    
            if (mode === loadModelClassification) {
                TheApp.loadModelClassificationAllSlot(selectedFile)
            } else if (mode === loadModelDetection) {
                TheApp.loadModelDetectionAllSlot(selectedFile)
            } else if (mode === convertModelClassificationTensorRT) {
                TheApp.convertModelClassificationToTensorRTSlot(selectedFile)
            } else if (mode === convertModelDetectionTensorRT) {
                TheApp.convertModelDetectionToTensorRTSlot(selectedFile)
            }
            else {

            }
        }

        onRejected: {
        }
    }

    //Main Layout
    Rectangle {
        id: leftGroup
        anchors.left: parent.left
        anchors.top: parent.top
        width: parent.width * 0.5
        height: parent.height

        Flickable {
            anchors.fill: parent
            contentWidth: leftGroup.width
            contentHeight: leftGroup.height
            clip: true

            Flow {
                id: flowLayout
                spacing: spacingElement

                width: parent.width

                Button {
                    width: buttonWidth
                    height: buttonHeight

                    text: "Load Model Classification All"
                    onClicked: {
                        fileModelDialog.extension = engineExtensionFile
                        fileModelDialog.mode = loadModelClassification
                        fileModelDialog.open()
                    }
                }

                Button {
                    width: buttonWidth
                    height: buttonHeight

                    text: "Load Model Detection All"
                    onClicked: {
                        fileModelDialog.extension = engineExtensionFile
                        fileModelDialog.mode = loadModelDetection
                        fileModelDialog.open()
                    }
                }

                Button {
                    width: buttonWidth
                    height: buttonHeight

                    text: "Release Model Classification All"
                    onClicked: TheApp.releaseModelClassificationAllSlot()
                }

                Button {
                    width: buttonWidth
                    height: buttonHeight

                    text: "Release Model Detection All"
                    onClicked: TheApp.releaseModelDetectionAllSlot()
                }


                Button {
                    width: buttonWidth
                    height: buttonHeight

                    text: "Load Image Classification"
                    onClicked: {
                        imageFileSelectDialog.mode = modeClassification
                        imageFileSelectDialog.open()
                    }
                }

                Button {
                    width: buttonWidth
                    height: buttonHeight

                    text: "Load Image Detection"
                    onClicked: {
                        imageFileSelectDialog.mode = modeDetection
                        imageFileSelectDialog.open()
                    }
                }

                Button {
                    width: buttonWidth
                    height: buttonHeight

                    text: "Infer All"
                    onClicked: TheApp.infereceAllSlots()
                }



                Button {
                    width: buttonWidth
                    height: buttonHeight

                    text: "Convert model Classification"
                    onClicked: {

                        fileModelDialog.extension = onnxExtensionFile
                        fileModelDialog.mode = convertModelClassificationTensorRT
                        fileModelDialog.open()

                    }
                }

                 Button {
                    width: buttonWidth
                    height: buttonHeight

                    text: "Convert model Detection"
                    onClicked: {

                        fileModelDialog.extension = onnxExtensionFile
                        fileModelDialog.mode = convertModelDetectionTensorRT
                        fileModelDialog.open()

                    }
                }
            }
        }
    }

    Rectangle {
        id: rightGroup
        anchors.left: leftGroup.right
        anchors.top: leftGroup.top
        anchors.right: parent.right
        anchors.bottom: parent.bottom

        ColumnLayout {
            spacing: 10
            anchors.fill: parent

            Button {
                width: buttonWidth
                height: buttonHeight
                text: "Clear log"
                onClicked: clearDataLog()
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true

                color: "black"

                ListView {
                    id: debugLogListView
                    anchors.fill: parent
                    spacing: spacingElement

                    clip: true
                    model: listDebugModelInfor

                    ScrollBar.vertical: ScrollBar {
                        policy: ScrollBar.AlwaysOn
                    }

                    delegate: Item {
                        width: ListView.width
                        height: debugLogItemHeight

                        // display infor property in model
                        Text {
                            font.pointSize: debugLogItemTextFont
                            text: model.infor
                            color: "white"
                        }
                    }

                    // Scroll to the end when a new item is added to the model
                    Component.onCompleted: {
                        listDebugModelInfor.onCountChanged.connect(function () {
                            debugLogListView.positionViewAtIndex(
                                        listDebugModelInfor.count - 1,
                                        ListView.End)
                        })
                    }
                }
            }
        }
    }
}
