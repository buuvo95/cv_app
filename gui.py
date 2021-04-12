import sys
import os
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from datagen import *
from model import *


##test
class Window(QWidget):

    def __init__(self):

        super().__init__()

        self.showMaximized()
        self.setWindowTitle("Computer Vision GUI")

        # Set title and defaultPath
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

        # Create the page for Activate and Loss function
        self.ActLoss = QWidget()
         # -> For define the hyparameter
        ActLossLayout = QFormLayout()
        ActLossLayout.setAlignment(Qt.AlignTop)
        self.Functions, self.Loss  = QLineEdit(), QLineEdit()
        ActLossLayout.addRow("Activate Functions:", self.Functions)
        ActLossLayout.addRow("Loss Functions:", self.Loss)
        
        self.ActLoss.setLayout(ActLossLayout)
       
        # Create Vertical Box to store comboBox
        comboBox = QVBoxLayout()
        comboBox.addWidget(self.pageCombo)
        comboBox.addWidget(self.ActLoss)
        comboBoxWG = QWidget()
        comboBoxWG.setLayout(comboBox)

        # Create Augmentation Widget
        AugmentationLabel = QLabel("Augmentation:")
        # Define label_mode : 'binary' or 'categorical' --- batch_size: size of batches of data. default=32 --- image_size
            # Horizontal and Vertical Shift Augmentation
        WHShift = QHBoxLayout()
                # Horizontal
        WidthLayout = QFormLayout()
        self.WidthShift = QLineEdit()
        WidthLayout.addRow("Width Shift: ", self.WidthShift)

                # Vertical
        HeightLayout = QFormLayout()
        self.HeightShift = QLineEdit()
        HeightLayout.addRow("Height Shift: ", self.HeightShift)

        WHShift.addLayout(WidthLayout)
        WHShift.addLayout(HeightLayout)
        WHShiftWG = QWidget()
        WHShiftWG.setLayout(WHShift)
            # Horizontal and Vertical Flip Augmentation
        HVFlip = QHBoxLayout()
                 # Horizontal
        HFlipLabel = QLabel("Horizontal Flip:")
        self.HFlipCBB = QComboBox()
        self.HFlipCBB.addItems(["True", "False"])
    
                # Vertical
        VFlipLabel = QLabel("Vertical Flip:")
        self.VFlipCBB = QComboBox()
        self.VFlipCBB.addItems(["True", "False"])

        HVFlip.addWidget(HFlipLabel)
        HVFlip.addWidget(self.HFlipCBB)
        HVFlip.addWidget(VFlipLabel)
        HVFlip.addWidget(self.VFlipCBB)
        HVFlipWG = QWidget()
        HVFlipWG.setLayout(HVFlip)
    

            # Radnom Rotation Augmentation
        RRotation = QHBoxLayout()
        RRotationLabel = QLabel("Rotation Range:")
        RRotationLabel.setMinimumWidth(140)
        
            # add slider
        self.RRotationSlider = QSlider(Qt.Horizontal)
        self.RRotationSlider.setMinimum(0)
        self.RRotationSlider.setMaximum(180)
        self.RRotationSlider.setValue(0)
        self.RRotationSlider.valueChanged.connect(self.valueChange)
            # Add label to displate value
        self.RRotationDisplay  = QLabel("0")

        RRotation.addWidget(RRotationLabel)
        RRotation.addWidget(self.RRotationSlider)
        RRotation.addWidget(self.RRotationDisplay)
        RRotationWG = QWidget()
        RRotationWG.setLayout(RRotation)

        #     # Random Brightness Augmentation
        # RBrightness = QHBoxLayout()
        # RBrightnessLabel = QLabel("Brightness Range:")
        # RBrightnessLabel.setMinimumWidth(140)
        # self.RBrightnessSlider = QSlider(Qt.Horizontal)
        # self.RBrightnessSlider.setMinimum(-200)
        # self.RBrightnessSlider.setMaximum(200)
        # self.RBrightnessSlider.setValue(0)
        # RBrightness.addWidget(RBrightnessLabel)
        # RBrightness.addWidget(self.RBrightnessSlider)
        # RBrightnessWG = QWidget()
        # RBrightnessWG.setLayout(RBrightness)

            # Random Zoom Augmentation
        RZoom = QHBoxLayout()
        RZoomLabel = QLabel("Zoom Range:")
        RZoomLabel.setMinimumWidth(140)
        self.RZoomSlider = QSlider(Qt.Horizontal)
        self.RZoomSlider.setMinimum(0)
        self.RZoomSlider.setMaximum(100)
        self.RZoomSlider.setValue(0) # -> Remember to devide by 100 for this variable
        self.RZoomSlider.setTickPosition(QSlider.TicksBelow)
        self.RZoomSlider.setTickInterval(10)
        self.RZoomSlider.valueChanged.connect(self.valueChange)

        self.RZoomDisplay  = QLabel("0.0")


        RZoom.addWidget(RZoomLabel)
        RZoom.addWidget(self.RZoomSlider)
        RZoom.addWidget(self.RZoomDisplay)
        RZoomWG = QWidget()
        RZoomWG.setLayout(RZoom)

            # Random Shear Augmentation
        RShear = QHBoxLayout()
        RShearLabel = QLabel("Shear Range:")
        RShearLabel.setMinimumWidth(140)
        self.RShearSlider = QSlider(Qt.Horizontal)
        self.RShearSlider.setMinimum(0)
        self.RShearSlider.setMaximum(100)
        self.RShearSlider.setValue(0) # -> Remember to devide by 100 for this variable
        self.RShearSlider.setTickPosition(QSlider.TicksBelow)
        self.RShearSlider.setTickInterval(10)
        self.RShearSlider.valueChanged.connect(self.valueChange)

        self.RShearDisplay  = QLabel("0.0")
        
        RShear.addWidget(RShearLabel)
        RShear.addWidget(self.RShearSlider)
        RShear.addWidget(self.RShearDisplay)
        RShearWG = QWidget()
        RShearWG.setLayout(RShear)

        augmentation = QVBoxLayout()
        augmentation.addWidget(AugmentationLabel)
        augmentation.addWidget(WHShiftWG)
        augmentation.addWidget(HVFlipWG)
        augmentation.addWidget(RRotationWG)
        # augmentation.addWidget(RBrightnessWG)
        augmentation.addWidget(RZoomWG)
        augmentation.addWidget(RShearWG)
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
        testingName = QLabel("Testing: ")
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
        self.imageFolder.clicked.connect(self._openTestImage)
        self.weightFolder.clicked.connect(self._openTestWeight)
        testingMouse.addWidget(self.imageFolder)
        testingMouse.addWidget(self.weightFolder)
        testingMouseWG = QWidget()
        testingMouseWG.setLayout(testingMouse)

        testDir.addWidget(testingKeyboardWG)
        testDir.addWidget(testingMouseWG)
        testDirWG.setLayout(testDir)

        # Testing Button
        testingBtn = QHBoxLayout()
        self.btnImage = QPushButton("Test")
        self.btnImage.clicked.connect(self._testImageClicked)
        self.btnWeight = QPushButton("Save")
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

    def valueChange(self):
        self.RRotationDisplay.setText(str(self.RRotationSlider.value()))
        self.RZoomDisplay.setText(str(round(self.RZoomSlider.value()/100,1)))
        self.RShearDisplay.setText(str(round(self.RShearSlider.value()/100,1)))
  
    # When train button clicked
    def _trainClicked(self):
        msg = QMessageBox()

        # Get training dir and validation dir
        trainingPath = self.trainDir.text()
        validPath = self.testDir.text()
        
        if trainingPath == '' or validPath == '':
            msg.setWindowTitle("Path Error!")
            msg.setText("Please re-enter folder path!")
            x = msg.exec_()
            return
        else:
            # Check training path and valid path
            if not (os.path.exists(trainingPath) and os.path.exists(validPath)):
                msg.setWindowTitle("Path Error!")
                msg.setText("Please check folder path again!")
                x = msg.exec_()
                return
            else:
                # Count folder
                trainCount = testCount = 0


                for foldername in os.listdir(trainingPath):
                    trainCount += 1
                for foldername in os.listdir(validPath):
                    testCount += 1
                # Check trainCount is equal to testCount
                if trainCount != testCount:
                    msg.setWindowTitle("Data Error!")
                    msg.setText("Please check test and validation sets again!")
                    x = msg.exec_()
                    return
                else:
                    # Check the number of class
                    class_mode = ""
                    if trainCount == 2:
                        class_mode = 'binary'
                        loss = 'binary_crossentropy'
                    elif trainCount > 2:
                        class_mode = 'categorical'
                        loss = 'categorical_crossentropy'
                    else:
                        msg.setWindowTitle("Data Error!")
                        msg.setText("Please check test and validation sets again!")
                        x = msg.exec_()

        # Define activat function and loss function
            # Ignore
        
        # The augmentation value:
            # Width shift
        if self.WidthShift.text() == '':
            width_shift = 0.0
        else:
            try:
                width_shift = float(self.WidthShift.text())
            except:
                msg.setWindowTitle("Value Eror")
                msg.setText("Please enter float number!")
                x = msg.exec_()
                return
        
            # Height Shift
        if self.HeightShift.text() == '':
            height_shift = 0.0
        else:
            try:
                height_shift = float(self.HeightShift.text())
            except:
                msg.setWindowTitle("Value Eror")
                msg.setText("Please enter float number!")
                x = msg.exec_()
                return

            # Check Witdh Shift and Height Shift in range
        if width_shift < -1.0 or width_shift >= 1.0 or height_shift < -1.0 or height_shift >= 1.0:
            msg.setWindowTitle("Value Eror")
            msg.setText("Please enter float number in range [-1.0,1.0) !")
            x = msg.exec_()
            return

            # Horizontal and vertical Flip
        horizontal_flip = self.HFlipCBB.currentText()
        vertical_flip  = self.VFlipCBB.currentText()

            # Rotation, Zoom, Shear Range values:
        rotationValue = self.RRotationSlider.value()
        zoomValue = round(self.RZoomSlider.value()/100,1)
        shearValue = round(self.RShearSlider.value()/100,1)
        
        # Get the target_size, model
        if self.pageCombo.currentText() == "AlexNet":
            target_size = (256,256)
            input_shape = (256,256,3)
            model = AlexNet(input_shape, trainCount)
        elif self.pageCombo.currentText() == "VGG":
            target_size = (224,224)
            input_shape = (224,224,3)
            model = VGGNet(input_shape, trainCount)
        elif self.pageCombo.currentText() == "InceptionNet":
            target_size = (224,224)
            input_shape = (224,224,3)
            model = InceptionNet(input_shape, trainCount)
        elif self.pageCombo.currentText() == "XceptionNet":
            target_size = (299,299)
            input_shape = (299,299,3)
            model = XceptionNet(input_shape, trainCount)
        else:
            target_size = (224,224)
            input_shape = (224,224,3)
            num_res_block = 5
            model = ResNet(input_shape, num_res_block, trainCount)
        
        # Define train and test datagenerator
        trainData = trainDatagen(
                                    width_shift,
                                    height_shift,
                                    rotationValue,
                                    zoomValue,
                                    shearValue,
                                    bool(horizontal_flip),
                                    bool(vertical_flip)
        )
        testData = testDatagen()

        trainGenerator = trainData.generator(
                                    trainingPath,
                                    target_size = target_size,
                                    batch_size = 16,
                                    class_mode = class_mode
        )
        
        testGenerator = testData.generator(
                                    validPath,
                                    target_size = target_size,
                                    batch_size = 16,
                                    class_mode = class_mode
        )

        optimizer = tf.keras.optimizers.SGD(lr = 0.01, momentum = 0.9)
        model.compile(optimizer = optimizer, loss = loss, metrics = ['accuracy'])
        print(model.summary())
        # history = model.fit_generator(
        #                             trainGenerator,
        #                             steps_per_epoch=100,
        #                             epochs=30,
        #                             validation_data=testGenerator,
        #                             validation_steps=50
        # )

        
    # When reset button clicked
    def _resetClicked(self):
        # DataSet Reset event
        self.trainDir.setText("")
        self.testDir.setText("")

        # ComboBox Reset event
        self.Functions.setText("")
        self.Loss.setText("")
        # Augmentation Reset even
        self.WidthShift.setText("")
        self.HeightShift.setText("")
        self.RRotationSlider.setValue(0)
        self.RShearSlider.setValue(0)
        self.RZoomSlider.setValue(0)


    # Open Directory when button on click
    def _openTrainDirectory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderPath = QFileDialog.getExistingDirectory(self,"Open Path", options=options)
        if len(folderPath) < 1:
            return
        self.trainDir.setText(folderPath)
    def _openTestDirectory(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        folderPath = QFileDialog.getExistingDirectory(self,"Open Path", options=options)
        if len(folderPath) < 1:
            return
        self.testDir.setText(folderPath)
    
    def _openTestImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Image Files (*.jpg);;Image Files (*.png);;Image Files (*.jpeg)", options=options)


    def _openTestWeight(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", "","All Files (*);;Weight Files (*.h5);;Weight Files (*.pb)", options=options)

    def _testImageClicked(self):
        pass


    def _testWeightClicked(self):
        pass



if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = Window()

    window.show()

    sys.exit(app.exec_())