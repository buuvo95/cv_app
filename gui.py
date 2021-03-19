import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class Window(QWidget):

    def __init__(self):

        super().__init__()

        self.showMaximized()
        self.setWindowTitle("Computer Vision GUI")

        # Set title and defaultPath
        self.title = "Open Path"
        self.defaultPath = "/home/ubuntu/Desktop"
        
        ################################
        #### Define the Left layout ####
        ################################
        
        imageLabel = QLabel(self)
        pixelMap = QPixmap('/home/ubuntu/Downloads/predictions.jpg') # Enter image path
        imageLabel.setPixmap(pixelMap)
        imageLabel.setMaximumSize(200,200)  
        # imageLabel.setScaledContents(True)
        imageLabel.setObjectName("photo")

        hboxLayoutLeft = QVBoxLayout()
        hboxLayoutLeft.addWidget(imageLabel)
        hboxLayoutLeftWG = QWidget()
        hboxLayoutLeftWG.setLayout(hboxLayoutLeft)
        
        ###############################
        ####Define the Right layout####
        ###############################

            ###Training###
            ##############
            
        # Create the image path of train data and test data
        dataDir = QHBoxLayout()
        dataDirWG = QWidget()
        # Input data dir from keyboard
        keyboardInput = QFormLayout()
        keyboardInputWG = QWidget()
        self.trainDir, self.testDir = QLineEdit(), QLineEdit()
        keyboardInput.addRow("Traning dir:", self.trainDir)
        keyboardInput.addRow("Validation dir:", self.testDir)
        keyboardInputWG.setLayout(keyboardInput)
        # Input data dir by clicking mouse
        mouseInput =  QVBoxLayout()
        self.trainFolder = QToolButton() # -> thay QPushButton by another one
        self.testFolder = QToolButton() # -> thay QPushButton by another one
        self.testFolder.setIcon(QIcon("./icon/open-folder.svg"))
        self.trainFolder.setIcon(QIcon("./icon/open-folder.svg"))
        self.trainFolder.clicked.connect(self._openTrainDirectory)
        self.testFolder.clicked.connect(self._openTestDirectory)
        mouseInput.addWidget(self.trainFolder)
        mouseInput.addWidget(self.testFolder)
        mouseInputWG = QWidget()
        mouseInputWG.setLayout(mouseInput)

        dataDir.addWidget(keyboardInputWG)
        dataDir.addWidget(mouseInputWG)
        dataDirWG.setLayout(dataDir)

        # Create and connect the combo box to switch between pages

        self.pageCombo = QComboBox()
        self.pageCombo.addItems(["AlexNet", "VGG", "InceptionNet", "XceptionNet", "ResNet"])
        self.pageCombo.activated.connect(self.switchPage)

        # Create the stacked layout

        # self.stackedLayout = QStackedLayout()

        # # Create the page for Alexnet
        # AlexNet = QWidget()
        #  # -> For define the hyparameter
        # AlexNetLayout = QFormLayout()
        # AlexNetLayout.setAlignment(Qt.AlignTop)
        # self.AlexNetFunctions, self.AlexNetLoss  = QLineEdit(), QLineEdit()
        # AlexNetLayout.addRow("Activate Functions:", self.AlexNetFunctions)
        # AlexNetLayout.addRow("Loss Functions:", self.AlexNetLoss)
        
        # AlexNet.setLayout(AlexNetLayout)
        # self.stackedLayout.addWidget(AlexNet)

        # # Create the page for VGG
        # VGG = QWidget()
        #  # -> For define the hyparameter
        # VGGLayout = QFormLayout()
        # VGGLayout.setAlignment(Qt.AlignTop)
        # self.VGGFunctions, self.VGGLoss  = QLineEdit(), QLineEdit()
        # VGGLayout.addRow("Activate Functions:", self.VGGFunctions)
        # VGGLayout.addRow("Loss Functions:", self.VGGLoss)
        
        # VGG.setLayout(VGGLayout)
        # self.stackedLayout.addWidget(VGG)

        # # Create the page for the InceptionNet
        # InceptionNet = QWidget()
        #  # -> For define the hyparameter
        # InceptionNetLayout = QFormLayout()
        # InceptionNetLayout.setAlignment(Qt.AlignTop)
        # self.InceptionNetFunctions, self.InceptionNetLoss  = QLineEdit(), QLineEdit()
        # InceptionNetLayout.addRow("Activate Functions:", self.InceptionNetFunctions)
        # InceptionNetLayout.addRow("Loss Functions:", self.InceptionNetLoss)
        
        # InceptionNet.setLayout(InceptionNetLayout)
        # self.stackedLayout.addWidget(InceptionNet)
        
        # # Create the page for the XceptionNet
        # XceptionNet = QWidget()
        #  # -> For define the hyparameter
        # XceptionNetLayout = QFormLayout()
        # XceptionNetLayout.setAlignment(Qt.AlignTop)
        # self.XceptionNetFunctions, self.XceptionNetLoss = QLineEdit(), QLineEdit()
        # XceptionNetLayout.addRow("Activate Functions:", self.XceptionNetFunctions)
        # XceptionNetLayout.addRow("Activate Functions:", self.XceptionNetLoss)

        # XceptionNet.setLayout(XceptionNetLayout)
        # self.stackedLayout.addWidget(XceptionNet)

        # # Create the page for the ResNet
        # ResNet = QWidget()
        #  # -> For define the hyparameter
        # ResNetLayout = QFormLayout()
        # ResNetLayout.setAlignment(Qt.AlignTop)
        # self.ResNetFunctions, self.ResNetLoss  = QLineEdit(), QLineEdit()
        # ResNetLayout.addRow("Activate Functions:", self.ResNetFunctions)
        # ResNetLayout.addRow("Loss Functions:", self.ResNetLoss)
        
        # ResNet.setLayout(ResNetLayout)
        # self.stackedLayout.addWidget(ResNet)

        # Create Vertical Box to store comboBox and stackedLayout
        comboBox = QVBoxLayout()
        comboBox.addWidget(self.pageCombo)
        # comboBox.addLayout(self.stackedLayout)
        comboBoxWG = QWidget()
        comboBoxWG.setLayout(comboBox)

        # Create Augmentation Widget
        AugmentationLabel = QLabel("Augmentation:")
            # Horizontal and Vertical Shift Augmentation
        HVShift = QHBoxLayout()
        HVShiftLabel = QLabel("H/V Shift:")
        HVShiftLabel.setMinimumWidth(140)
        self.HVShiftSlider = QSlider(Qt.Horizontal)
        self.HVShiftSlider.setMinimum(-200)
        self.HVShiftSlider.setMaximum(200)
        self.HVShiftSlider.setValue(0)
        HVShift.addWidget(HVShiftLabel)
        HVShift.addWidget(self.HVShiftSlider)
        HVShiftWG = QWidget()
        HVShiftWG.setLayout(HVShift)
            # Horizontal and Vertical Flip Augmentation
        HVFlip = QHBoxLayout()
        HVFlipLabel = QLabel("H/V Flip:")
        HVFlipLabel.setMinimumWidth(140)
        self.HVFlipSlider = QSlider(Qt.Horizontal)
        self.HVFlipSlider.setMinimum(-200)
        self.HVFlipSlider.setMaximum(200)
        self.HVFlipSlider.setValue(0)
        HVFlip.addWidget(HVFlipLabel)
        HVFlip.addWidget(self.HVFlipSlider)
        HVFlipWG = QWidget()
        HVFlipWG.setLayout(HVFlip)
            # Radnom Rotation Augmentation
        RRotation = QHBoxLayout()
        RRotationLabel = QLabel("Random Rotation:")
        RRotationLabel.setMinimumWidth(140)
        self.RRotationSlider = QSlider(Qt.Horizontal)
        self.RRotationSlider.setMinimum(-200)
        self.RRotationSlider.setMaximum(200)
        self.RRotationSlider.setValue(0)
        RRotation.addWidget(RRotationLabel)
        RRotation.addWidget(self.RRotationSlider)
        RRotationWG = QWidget()
        RRotationWG.setLayout(RRotation)
            # Random Brightness Augmentation
        RBrightness = QHBoxLayout()
        RBrightnessLabel = QLabel("Random Brightness:")
        RBrightnessLabel.setMinimumWidth(140)
        self.RBrightnessSlider = QSlider(Qt.Horizontal)
        self.RBrightnessSlider.setMinimum(-200)
        self.RBrightnessSlider.setMaximum(200)
        self.RBrightnessSlider.setValue(0)
        RBrightness.addWidget(RBrightnessLabel)
        RBrightness.addWidget(self.RBrightnessSlider)
        RBrightnessWG = QWidget()
        RBrightnessWG.setLayout(RBrightness)
            # Random Zoom Augmentation
        RZoom = QHBoxLayout()
        RZoomLabel = QLabel("Random Zoom:")
        RZoomLabel.setMinimumWidth(140)
        self.RZoomSlider = QSlider(Qt.Horizontal)
        self.RZoomSlider.setMinimum(-200)
        self.RZoomSlider.setMaximum(200)
        self.RZoomSlider.setValue(0)
        RZoom.addWidget(RZoomLabel)
        RZoom.addWidget(self.RZoomSlider)
        RZoomWG = QWidget()
        RZoomWG.setLayout(RZoom)

        augmentation = QVBoxLayout()
        augmentation.addWidget(AugmentationLabel)
        augmentation.addWidget(HVShiftWG)
        augmentation.addWidget(HVFlipWG)
        augmentation.addWidget(RRotationWG)
        augmentation.addWidget(RBrightnessWG)
        augmentation.addWidget(RZoomWG)
        augmentationWG = QWidget()
        augmentationWG.setLayout(augmentation)
        augmentationWG.setGeometry(0,0,300,300)

        # Create Train and Reset Button
        btnArea = QHBoxLayout()
        self.btnTrain = QPushButton("Train")
        self.btnTrain.clicked.connect(self._trainClicked)
        self.btnReset = QPushButton("Reset")
        self.btnReset.clicked.connect(self._resetClicked)
        btnArea.addWidget(self.btnTrain)
        btnArea.addWidget(self.btnReset)
        btnAreaWG = QWidget()
        btnAreaWG.setLayout(btnArea)

        trainingLayout = QVBoxLayout()
        trainingLayout.setAlignment(Qt.AlignTop)
        trainingLayout.addWidget(dataDirWG)
        trainingLayout.addWidget(comboBoxWG)
        trainingLayout.addWidget(augmentationWG)
        trainingLayout.addWidget(btnAreaWG)
        trainingLayoutWG = QWidget()
        trainingLayoutWG.setLayout(trainingLayout)

            ###Testing###
            #############
        # Create label name
        testingName = QLabel("Testing")
        # Create load weight
        testingLayout = QVBoxLayout()
        testingLayoutWG = QWidget()
        testDir = QHBoxLayout()
        testDirWG = QWidget()
        
        # Input Weights and Image
        testingKeyboard = QFormLayout()
        testingKeyboardWG = QWidget()
        self.inputWeight, self.inputImage = QLineEdit(), QLineEdit()
        testingKeyboard.addRow("Image Dir:", self.inputWeight)
        testingKeyboard.addRow("Weight Dir:", self.inputImage)
        testingKeyboardWG.setLayout(testingKeyboard)

        # Input data dir by clicking mouse
        testingMouse =  QVBoxLayout()
        self.imageFolder = QToolButton() # -> thay QPushButton by another one
        self.weightFolder = QToolButton() # -> thay QPushButton by another one
        self.imageFolder.setIcon(QIcon("./icon/open-folder.svg"))
        self.weightFolder.setIcon(QIcon("./icon/open-folder.svg"))
        #self.imageFolder.clicked.connect(self._openTrainDirectory)
        #self.weightFolder.clicked.connect(self._openTestDirectory)
        testingMouse.addWidget(self.imageFolder)
        testingMouse.addWidget(self.weightFolder)
        testingMouseWG = QWidget()
        testingMouseWG.setLayout(testingMouse)

        testDir.addWidget(testingKeyboardWG)
        testDir.addWidget(testingMouseWG)
        testDirWG.setLayout(testDir)

        # Testing Button
        testingBtn = QHBoxLayout()
        self.btnImage = QPushButton("Image")
        self.btnImage.clicked.connect(self._testImageClicked)
        self.btnWeight = QPushButton("Weight")
        self.btnWeight.clicked.connect(self._testWeightClicked)
        testingBtn.addWidget(self.btnImage)
        testingBtn.addWidget(self.btnWeight)
        testingBtnWG = QWidget()
        testingBtnWG.setLayout(testingBtn)

        testingLayout.addWidget(testingName)
        testingLayout.addWidget(testDirWG)
        testingLayout.addWidget(testingBtnWG)
        testingLayoutWG.setLayout(testingLayout)

        hboxLayoutRight = QVBoxLayout()
        hboxLayoutRight.setAlignment(Qt.AlignTop)
        hboxLayoutRight.addWidget(trainingLayoutWG)
        hboxLayoutRight.addWidget(testingLayoutWG)
        hboxLayoutRightWG = QWidget()
        hboxLayoutRight.setAlignment(Qt.AlignTop)
        hboxLayoutRightWG.setMaximumWidth(500)
        hboxLayoutRightWG.setLayout(hboxLayoutRight)

         # Create a top-level layout

        mainLayout = QHBoxLayout()
        mainLayout.addWidget(hboxLayoutLeftWG)
        mainLayout.addWidget(hboxLayoutRightWG)
        self.setLayout(mainLayout)

    # When switch page button click
    # def switchPage(self):
    #     self.stackedLayout.setCurrentIndex(self.pageCombo.currentIndex())
    
    # When train button clicked
    def _trainClicked(self):
        if self.pageCombo.currentText() == "AlexNet":
            # Make Data augmentation and train
        elif self.pageCombo.currentText() == "VGG":
            pass
        elif self.pageCombo.currentText() == "InceptionNet":
            pass
        elif self.pageCombo.currentText() == "XceptionNet":
        else:
            pass
        print(self.Classes)
        pass

    # When reset button clicked
    def _resetClicked(self):
        
        # DataSet Reset event
        self.trainDir.setText("")
        self.testDir.setText("")

        # ComboBox Reset event
        self.AlexNetFunctions.setText("")
        self.AlexNetLoss.setText("")
        self.InceptionNetFunctions.setText("")
        self.InceptionNetLoss.setText("")
        self.VGGFunctions.setText("")
        self.VGGLoss.setText("")
        self.ResNetFunctions.setText("")
        self.ResNetLoss.setText("")
        # Augmentation Reset event
        self.HVShiftSlider.setValue(0)
        self.HVFlipSlider.setValue(0)
        self.RRotationSlider.setValue(0)
        self.RBrightnessSlider.setValue(0)
        self.RZoomSlider.setValue(0)


    # Open Directory when button on click
    def _openTrainDirectory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,self.title, self.defaultPath,"All Files (*)", options=options)
        if len(fileName) < 1:
            return
        self.trainDir.setText(fileName)
    def _openTestDirectory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,self.title, self.defaultPath,"All Files (*)", options=options)
        if len(fileName) < 1:
            return
        self.testDir.setText(fileName)
    
    def _testImageClicked(self):
        pass

    def _testWeightClicked(self):
        pass

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = Window()

    window.show()

    sys.exit(app.exec_())