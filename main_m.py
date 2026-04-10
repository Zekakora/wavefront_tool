from PySide6 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1240, 800)
        MainWindow.setMinimumSize(QtCore.QSize(900, 620))

        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.mainStackedWidget = QtWidgets.QStackedWidget(parent=self.centralwidget)
        self.mainStackedWidget.setObjectName("mainStackedWidget")

        # =========================
        # Page 1: Wavefront detection
        # =========================
        self.pageWavefrontDetection = QtWidgets.QWidget()
        self.pageWavefrontDetection.setObjectName("pageWavefrontDetection")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.pageWavefrontDetection)
        self.horizontalLayout_3.setContentsMargins(16, 16, 16, 16)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")

        self.mainSplitter = QtWidgets.QSplitter(parent=self.pageWavefrontDetection)
        self.mainSplitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.mainSplitter.setChildrenCollapsible(False)
        self.mainSplitter.setHandleWidth(6)
        self.mainSplitter.setObjectName("mainSplitter")

        # -------------------------
        # Left side
        # -------------------------
        self.leftPanel = QtWidgets.QWidget(parent=self.mainSplitter)
        self.leftPanel.setObjectName("leftPanel")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.leftPanel)
        self.verticalLayout.setContentsMargins(0, 0, 10, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")

        self.leftVerticalSplitter = QtWidgets.QSplitter(parent=self.leftPanel)
        self.leftVerticalSplitter.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.leftVerticalSplitter.setChildrenCollapsible(False)
        self.leftVerticalSplitter.setHandleWidth(6)
        self.leftVerticalSplitter.setObjectName("leftVerticalSplitter")

        self.plotCard = QtWidgets.QFrame(parent=self.leftVerticalSplitter)
        self.plotCard.setObjectName("plotCard")
        self.plotCard.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.plotCardLayout = QtWidgets.QVBoxLayout(self.plotCard)
        self.plotCardLayout.setContentsMargins(12, 12, 12, 12)
        self.plotCardLayout.setSpacing(8)
        self.plotCardLayout.setObjectName("plotCardLayout")

        self.framePlotContainer = QtWidgets.QFrame(parent=self.plotCard)
        self.framePlotContainer.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.framePlotContainer.setFrameShadow(QtWidgets.QFrame.Shadow.Raised)
        self.framePlotContainer.setMinimumSize(QtCore.QSize(320, 220))
        self.framePlotContainer.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
        )
        self.framePlotContainer.setObjectName("framePlotContainer")
        self.plotCardLayout.addWidget(self.framePlotContainer)

        self.leftInputScrollArea = QtWidgets.QScrollArea(parent=self.leftVerticalSplitter)
        self.leftInputScrollArea.setWidgetResizable(True)
        self.leftInputScrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.leftInputScrollArea.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.leftInputScrollArea.setObjectName("leftInputScrollArea")

        self.leftInputScrollContents = QtWidgets.QWidget()
        self.leftInputScrollContents.setObjectName("leftInputScrollContents")
        self.leftInputScrollLayout = QtWidgets.QVBoxLayout(self.leftInputScrollContents)
        self.leftInputScrollLayout.setContentsMargins(0, 12, 0, 0)
        self.leftInputScrollLayout.setSpacing(0)
        self.leftInputScrollLayout.setObjectName("leftInputScrollLayout")

        self.inputCard = QtWidgets.QFrame(parent=self.leftInputScrollContents)
        self.inputCard.setObjectName("inputCard")
        self.inputCard.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.inputCard.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Maximum,
            )
        )
        self.inputCardLayout = QtWidgets.QVBoxLayout(self.inputCard)
        self.inputCardLayout.setContentsMargins(16, 16, 16, 16)
        self.inputCardLayout.setSpacing(12)
        self.inputCardLayout.setObjectName("inputCardLayout")

        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")

        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setHorizontalSpacing(12)
        self.gridLayout_3.setVerticalSpacing(10)
        self.gridLayout_3.setObjectName("gridLayout_3")

        self.labelInputSectionTitle = QtWidgets.QLabel(parent=self.inputCard)
        titleFont = QtGui.QFont()
        titleFont.setPointSize(11)
        titleFont.setBold(True)
        self.labelInputSectionTitle.setFont(titleFont)
        self.labelInputSectionTitle.setObjectName("labelInputSectionTitle")
        self.gridLayout_3.addWidget(self.labelInputSectionTitle, 0, 0, 1, 4)

        self.line = QtWidgets.QFrame(parent=self.inputCard)
        self.line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_3.addWidget(self.line, 1, 0, 1, 4)

        self.labelInputDirA = QtWidgets.QLabel(parent=self.inputCard)
        self.labelInputDirA.setObjectName("labelInputDirA")
        self.gridLayout_3.addWidget(self.labelInputDirA, 2, 0, 1, 1)

        self.editInputDirA = QtWidgets.QLineEdit(parent=self.inputCard)
        self.editInputDirA.setClearButtonEnabled(True)
        self.editInputDirA.setObjectName("editInputDirA")
        self.gridLayout_3.addWidget(self.editInputDirA, 2, 1, 1, 2)

        self.btnBrowseInputDirA = QtWidgets.QToolButton(parent=self.inputCard)
        self.btnBrowseInputDirA.setObjectName("btnBrowseInputDirA")
        self.gridLayout_3.addWidget(self.btnBrowseInputDirA, 2, 3, 1, 1)

        self.labelInputDirB = QtWidgets.QLabel(parent=self.inputCard)
        self.labelInputDirB.setObjectName("labelInputDirB")
        self.gridLayout_3.addWidget(self.labelInputDirB, 3, 0, 1, 1)

        self.editInputDirB = QtWidgets.QLineEdit(parent=self.inputCard)
        self.editInputDirB.setClearButtonEnabled(True)
        self.editInputDirB.setObjectName("editInputDirB")
        self.gridLayout_3.addWidget(self.editInputDirB, 3, 1, 1, 2)

        self.btnBrowseInputDirB = QtWidgets.QToolButton(parent=self.inputCard)
        self.btnBrowseInputDirB.setObjectName("btnBrowseInputDirB")
        self.gridLayout_3.addWidget(self.btnBrowseInputDirB, 3, 3, 1, 1)

        self.labelMatchMode = QtWidgets.QLabel(parent=self.inputCard)
        self.labelMatchMode.setObjectName("labelMatchMode")
        self.gridLayout_3.addWidget(self.labelMatchMode, 4, 0, 1, 1)

        self.comboMatchMode = QtWidgets.QComboBox(parent=self.inputCard)
        self.comboMatchMode.setObjectName("comboMatchMode")
        self.comboMatchMode.addItem("")
        self.comboMatchMode.addItem("")
        self.gridLayout_3.addWidget(self.comboMatchMode, 4, 1, 1, 3)

        self.labelPairKeyword = QtWidgets.QLabel(parent=self.inputCard)
        self.labelPairKeyword.setObjectName("labelPairKeyword")
        self.gridLayout_3.addWidget(self.labelPairKeyword, 5, 0, 1, 1)

        self.editPairKeyword = QtWidgets.QLineEdit(parent=self.inputCard)
        self.editPairKeyword.setClearButtonEnabled(True)
        self.editPairKeyword.setObjectName("editPairKeyword")
        self.gridLayout_3.addWidget(self.editPairKeyword, 5, 1, 1, 3)

        self.labelOutputDir = QtWidgets.QLabel(parent=self.inputCard)
        self.labelOutputDir.setObjectName("labelOutputDir")
        self.gridLayout_3.addWidget(self.labelOutputDir, 6, 0, 1, 1)

        self.editOutputDir = QtWidgets.QLineEdit(parent=self.inputCard)
        self.editOutputDir.setClearButtonEnabled(True)
        self.editOutputDir.setObjectName("editOutputDir")
        self.gridLayout_3.addWidget(self.editOutputDir, 6, 1, 1, 2)

        self.btnBrowseOutputDir = QtWidgets.QToolButton(parent=self.inputCard)
        self.btnBrowseOutputDir.setObjectName("btnBrowseOutputDir")
        self.gridLayout_3.addWidget(self.btnBrowseOutputDir, 6, 3, 1, 1)

        self.labelSensorDistance = QtWidgets.QLabel(parent=self.inputCard)
        self.labelSensorDistance.setObjectName("labelSensorDistance")
        self.gridLayout_3.addWidget(self.labelSensorDistance, 7, 0, 1, 1)

        self.editSensorDistanceM = QtWidgets.QLineEdit(parent=self.inputCard)
        self.editSensorDistanceM.setObjectName("editSensorDistanceM")
        self.gridLayout_3.addWidget(self.editSensorDistanceM, 7, 1, 1, 3)

        self.labelSamplingFreq = QtWidgets.QLabel(parent=self.inputCard)
        self.labelSamplingFreq.setObjectName("labelSamplingFreq")
        self.gridLayout_3.addWidget(self.labelSamplingFreq, 8, 0, 1, 1)

        self.editSamplingFreqMHz = QtWidgets.QLineEdit(parent=self.inputCard)
        self.editSamplingFreqMHz.setObjectName("editSamplingFreqMHz")
        self.gridLayout_3.addWidget(self.editSamplingFreqMHz, 8, 1, 1, 3)

        self.gridLayout_3.setColumnStretch(0, 0)
        self.gridLayout_3.setColumnStretch(1, 1)
        self.gridLayout_3.setColumnStretch(2, 1)
        self.gridLayout_3.setColumnStretch(3, 0)
        self.gridLayout_3.setColumnMinimumWidth(0, 96)
        self.gridLayout_3.setColumnMinimumWidth(3, 40)

        self.horizontalLayout_2.addLayout(self.gridLayout_3)
        self.inputCardLayout.addLayout(self.horizontalLayout_2)
        self.leftInputScrollLayout.addWidget(self.inputCard)
        self.leftInputScrollLayout.addStretch(1)
        self.leftInputScrollArea.setWidget(self.leftInputScrollContents)

        self.verticalLayout.addWidget(self.leftVerticalSplitter)
        self.leftVerticalSplitter.setSizes([560, 240])

        # -------------------------
        # Right side in scroll area
        # -------------------------
        self.rightScrollArea = QtWidgets.QScrollArea(parent=self.mainSplitter)
        self.rightScrollArea.setWidgetResizable(True)
        self.rightScrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.rightScrollArea.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.rightScrollArea.setObjectName("rightScrollArea")
        self.rightScrollArea.setMinimumWidth(350)
        self.rightScrollArea.setMaximumWidth(500)

        self.rightScrollContents = QtWidgets.QWidget()
        self.rightScrollContents.setObjectName("rightScrollContents")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.rightScrollContents)
        self.verticalLayout_2.setContentsMargins(10, 0, 0, 0)
        self.verticalLayout_2.setSpacing(14)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.resultCard = QtWidgets.QFrame(parent=self.rightScrollContents)
        self.resultCard.setObjectName("resultCard")
        self.resultCard.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.resultCardLayout = QtWidgets.QVBoxLayout(self.resultCard)
        self.resultCardLayout.setContentsMargins(16, 16, 16, 16)
        self.resultCardLayout.setSpacing(10)
        self.resultCardLayout.setObjectName("resultCardLayout")

        self.formLayout = QtWidgets.QFormLayout()
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setHorizontalSpacing(12)
        self.formLayout.setVerticalSpacing(10)
        self.formLayout.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.formLayout.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.formLayout.setObjectName("formLayout")

        self.labelResultSectionTitle = QtWidgets.QLabel(parent=self.resultCard)
        self.labelResultSectionTitle.setFont(titleFont)
        self.labelResultSectionTitle.setObjectName("labelResultSectionTitle")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.ItemRole.SpanningRole, self.labelResultSectionTitle)

        self.labelCurrentFile = QtWidgets.QLabel(parent=self.resultCard)
        self.labelCurrentFile.setObjectName("labelCurrentFile")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelCurrentFile)
        self.valueCurrentFile = QtWidgets.QLabel(parent=self.resultCard)
        self.valueCurrentFile.setWordWrap(True)
        self.valueCurrentFile.setObjectName("valueCurrentFile")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.valueCurrentFile)

        self.labelCurrentAlgorithm = QtWidgets.QLabel(parent=self.resultCard)
        self.labelCurrentAlgorithm.setObjectName("labelCurrentAlgorithm")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelCurrentAlgorithm)
        self.valueCurrentAlgorithm = QtWidgets.QLabel(parent=self.resultCard)
        self.valueCurrentAlgorithm.setWordWrap(True)
        self.valueCurrentAlgorithm.setObjectName("valueCurrentAlgorithm")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.valueCurrentAlgorithm)

        self.labelChannelAResult = QtWidgets.QLabel(parent=self.resultCard)
        self.labelChannelAResult.setObjectName("labelChannelAResult")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelChannelAResult)
        self.valueChannelAResult = QtWidgets.QLabel(parent=self.resultCard)
        self.valueChannelAResult.setWordWrap(True)
        self.valueChannelAResult.setObjectName("valueChannelAResult")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.FieldRole, self.valueChannelAResult)

        self.labelChannelBResult = QtWidgets.QLabel(parent=self.resultCard)
        self.labelChannelBResult.setObjectName("labelChannelBResult")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelChannelBResult)
        self.valueChannelBResult = QtWidgets.QLabel(parent=self.resultCard)
        self.valueChannelBResult.setWordWrap(True)
        self.valueChannelBResult.setObjectName("valueChannelBResult")
        self.formLayout.setWidget(4, QtWidgets.QFormLayout.ItemRole.FieldRole, self.valueChannelBResult)

        self.labelTimeDifference = QtWidgets.QLabel(parent=self.resultCard)
        self.labelTimeDifference.setObjectName("labelTimeDifference")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelTimeDifference)
        self.valueTimeDifference = QtWidgets.QLabel(parent=self.resultCard)
        self.valueTimeDifference.setWordWrap(True)
        self.valueTimeDifference.setObjectName("valueTimeDifference")
        self.formLayout.setWidget(5, QtWidgets.QFormLayout.ItemRole.FieldRole, self.valueTimeDifference)

        self.labelDistanceToA = QtWidgets.QLabel(parent=self.resultCard)
        self.labelDistanceToA.setObjectName("labelDistanceToA")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelDistanceToA)
        self.valueDistanceToA = QtWidgets.QLabel(parent=self.resultCard)
        self.valueDistanceToA.setWordWrap(True)
        self.valueDistanceToA.setObjectName("valueDistanceToA")
        self.formLayout.setWidget(6, QtWidgets.QFormLayout.ItemRole.FieldRole, self.valueDistanceToA)

        self.labelDistanceToB = QtWidgets.QLabel(parent=self.resultCard)
        self.labelDistanceToB.setObjectName("labelDistanceToB")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelDistanceToB)
        self.valueDistanceToB = QtWidgets.QLabel(parent=self.resultCard)
        self.valueDistanceToB.setWordWrap(True)
        self.valueDistanceToB.setObjectName("valueDistanceToB")
        self.formLayout.setWidget(7, QtWidgets.QFormLayout.ItemRole.FieldRole, self.valueDistanceToB)

        self.labelDetectionStatus = QtWidgets.QLabel(parent=self.resultCard)
        self.labelDetectionStatus.setObjectName("labelDetectionStatus")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelDetectionStatus)
        self.valueDetectionStatus = QtWidgets.QLabel(parent=self.resultCard)
        self.valueDetectionStatus.setWordWrap(True)
        self.valueDetectionStatus.setObjectName("valueDetectionStatus")
        self.formLayout.setWidget(8, QtWidgets.QFormLayout.ItemRole.FieldRole, self.valueDetectionStatus)

        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setSpacing(10)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")

        self.btnSaveResult = QtWidgets.QPushButton(parent=self.resultCard)
        self.btnSaveResult.setMinimumHeight(36)
        self.btnSaveResult.setObjectName("btnSaveResult")
        self.horizontalLayout_4.addWidget(self.btnSaveResult)

        self.btnClearResult = QtWidgets.QPushButton(parent=self.resultCard)
        self.btnClearResult.setMinimumHeight(36)
        self.btnClearResult.setObjectName("btnClearResult")
        self.horizontalLayout_4.addWidget(self.btnClearResult)

        self.formLayout.setLayout(9, QtWidgets.QFormLayout.ItemRole.SpanningRole, self.horizontalLayout_4)
        self.resultCardLayout.addLayout(self.formLayout)
        self.verticalLayout_2.addWidget(self.resultCard, 0)

        self.algorithmCard = QtWidgets.QFrame(parent=self.rightScrollContents)
        self.algorithmCard.setObjectName("algorithmCard")
        self.algorithmCard.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.algorithmCardLayout = QtWidgets.QVBoxLayout(self.algorithmCard)
        self.algorithmCardLayout.setContentsMargins(16, 16, 16, 16)
        self.algorithmCardLayout.setSpacing(12)
        self.algorithmCardLayout.setObjectName("algorithmCardLayout")

        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(12)
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        self.labelAlgorithmSectionTitle = QtWidgets.QLabel(parent=self.algorithmCard)
        self.labelAlgorithmSectionTitle.setFont(titleFont)
        self.labelAlgorithmSectionTitle.setObjectName("labelAlgorithmSectionTitle")
        self.verticalLayout_3.addWidget(self.labelAlgorithmSectionTitle)

        self.comboAlgorithm = QtWidgets.QComboBox(parent=self.algorithmCard)
        self.comboAlgorithm.setMinimumHeight(38)
        self.comboAlgorithm.setObjectName("comboAlgorithm")
        self.comboAlgorithm.addItem("")
        self.comboAlgorithm.addItem("")
        self.verticalLayout_3.addWidget(self.comboAlgorithm)

        self.stackedAlgorithmKeyParams = QtWidgets.QStackedWidget(parent=self.algorithmCard)
        self.stackedAlgorithmKeyParams.setObjectName("stackedAlgorithmKeyParams")
        self.stackedAlgorithmKeyParams.setSizePolicy(
            QtWidgets.QSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Maximum,
            )
        )

        self.pageRdpLocalAicKeyParams = QtWidgets.QWidget()
        self.pageRdpLocalAicKeyParams.setObjectName("pageRdpLocalAicKeyParams")
        self.formLayoutRdpLocalAicKeyParams = QtWidgets.QFormLayout(self.pageRdpLocalAicKeyParams)
        self.formLayoutRdpLocalAicKeyParams.setContentsMargins(0, 0, 0, 0)
        self.formLayoutRdpLocalAicKeyParams.setHorizontalSpacing(12)
        self.formLayoutRdpLocalAicKeyParams.setVerticalSpacing(10)
        self.formLayoutRdpLocalAicKeyParams.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.formLayoutRdpLocalAicKeyParams.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.formLayoutRdpLocalAicKeyParams.setObjectName("formLayoutRdpLocalAicKeyParams")

        self.labelRdpPreN = QtWidgets.QLabel(parent=self.pageRdpLocalAicKeyParams)
        self.labelRdpPreN.setObjectName("labelRdpPreN")
        self.formLayoutRdpLocalAicKeyParams.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelRdpPreN)
        self.editRdpLocalAic_pre_n = QtWidgets.QLineEdit(parent=self.pageRdpLocalAicKeyParams)
        self.editRdpLocalAic_pre_n.setObjectName("editRdpLocalAic_pre_n")
        self.formLayoutRdpLocalAicKeyParams.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.editRdpLocalAic_pre_n)

        self.labelRdpRoughK = QtWidgets.QLabel(parent=self.pageRdpLocalAicKeyParams)
        self.labelRdpRoughK.setObjectName("labelRdpRoughK")
        self.formLayoutRdpLocalAicKeyParams.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelRdpRoughK)
        self.editRdpLocalAic_rough_k = QtWidgets.QLineEdit(parent=self.pageRdpLocalAicKeyParams)
        self.editRdpLocalAic_rough_k.setObjectName("editRdpLocalAic_rough_k")
        self.formLayoutRdpLocalAicKeyParams.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.editRdpLocalAic_rough_k)

        self.labelRdpEpsilon = QtWidgets.QLabel(parent=self.pageRdpLocalAicKeyParams)
        self.labelRdpEpsilon.setObjectName("labelRdpEpsilon")
        self.formLayoutRdpLocalAicKeyParams.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelRdpEpsilon)
        self.editRdpLocalAic_rdp_epsilon = QtWidgets.QLineEdit(parent=self.pageRdpLocalAicKeyParams)
        self.editRdpLocalAic_rdp_epsilon.setObjectName("editRdpLocalAic_rdp_epsilon")
        self.formLayoutRdpLocalAicKeyParams.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.editRdpLocalAic_rdp_epsilon)

        self.labelRdpSearchLeftUs = QtWidgets.QLabel(parent=self.pageRdpLocalAicKeyParams)
        self.labelRdpSearchLeftUs.setObjectName("labelRdpSearchLeftUs")
        self.formLayoutRdpLocalAicKeyParams.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelRdpSearchLeftUs)
        self.editRdpLocalAic_search_left_us = QtWidgets.QLineEdit(parent=self.pageRdpLocalAicKeyParams)
        self.editRdpLocalAic_search_left_us.setObjectName("editRdpLocalAic_search_left_us")
        self.formLayoutRdpLocalAicKeyParams.setWidget(3, QtWidgets.QFormLayout.ItemRole.FieldRole, self.editRdpLocalAic_search_left_us)

        self.labelRdpSearchRightUs = QtWidgets.QLabel(parent=self.pageRdpLocalAicKeyParams)
        self.labelRdpSearchRightUs.setObjectName("labelRdpSearchRightUs")
        self.formLayoutRdpLocalAicKeyParams.setWidget(4, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelRdpSearchRightUs)
        self.editRdpLocalAic_search_right_us = QtWidgets.QLineEdit(parent=self.pageRdpLocalAicKeyParams)
        self.editRdpLocalAic_search_right_us.setObjectName("editRdpLocalAic_search_right_us")
        self.formLayoutRdpLocalAicKeyParams.setWidget(4, QtWidgets.QFormLayout.ItemRole.FieldRole, self.editRdpLocalAic_search_right_us)

        self.labelRdpAmpK = QtWidgets.QLabel(parent=self.pageRdpLocalAicKeyParams)
        self.labelRdpAmpK.setObjectName("labelRdpAmpK")
        self.formLayoutRdpLocalAicKeyParams.setWidget(5, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelRdpAmpK)
        self.editRdpLocalAic_amp_k = QtWidgets.QLineEdit(parent=self.pageRdpLocalAicKeyParams)
        self.editRdpLocalAic_amp_k.setObjectName("editRdpLocalAic_amp_k")
        self.formLayoutRdpLocalAicKeyParams.setWidget(5, QtWidgets.QFormLayout.ItemRole.FieldRole, self.editRdpLocalAic_amp_k)

        self.btnOpenRdpLocalAicParamDialog = QtWidgets.QPushButton(parent=self.pageRdpLocalAicKeyParams)
        self.btnOpenRdpLocalAicParamDialog.setMinimumHeight(36)
        self.btnOpenRdpLocalAicParamDialog.setObjectName("btnOpenRdpLocalAicParamDialog")
        self.formLayoutRdpLocalAicKeyParams.setWidget(6, QtWidgets.QFormLayout.ItemRole.FieldRole, self.btnOpenRdpLocalAicParamDialog)

        self.stackedAlgorithmKeyParams.addWidget(self.pageRdpLocalAicKeyParams)

        self.pageIceemdanTeoKeyParams = QtWidgets.QWidget()
        self.pageIceemdanTeoKeyParams.setObjectName("pageIceemdanTeoKeyParams")
        self.formLayoutIceemdanTeoKeyParams = QtWidgets.QFormLayout(self.pageIceemdanTeoKeyParams)
        self.formLayoutIceemdanTeoKeyParams.setContentsMargins(0, 0, 0, 0)
        self.formLayoutIceemdanTeoKeyParams.setHorizontalSpacing(12)
        self.formLayoutIceemdanTeoKeyParams.setVerticalSpacing(10)
        self.formLayoutIceemdanTeoKeyParams.setFieldGrowthPolicy(QtWidgets.QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.formLayoutIceemdanTeoKeyParams.setRowWrapPolicy(QtWidgets.QFormLayout.RowWrapPolicy.WrapLongRows)
        self.formLayoutIceemdanTeoKeyParams.setObjectName("formLayoutIceemdanTeoKeyParams")

        self.labelIceRdpEpsilon = QtWidgets.QLabel(parent=self.pageIceemdanTeoKeyParams)
        self.labelIceRdpEpsilon.setObjectName("labelIceRdpEpsilon")
        self.formLayoutIceemdanTeoKeyParams.setWidget(0, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelIceRdpEpsilon)
        self.editIceemdanTeo_rdp_epsilon = QtWidgets.QLineEdit(parent=self.pageIceemdanTeoKeyParams)
        self.editIceemdanTeo_rdp_epsilon.setObjectName("editIceemdanTeo_rdp_epsilon")
        self.formLayoutIceemdanTeoKeyParams.setWidget(0, QtWidgets.QFormLayout.ItemRole.FieldRole, self.editIceemdanTeo_rdp_epsilon)

        self.labelIceEnsembleSize = QtWidgets.QLabel(parent=self.pageIceemdanTeoKeyParams)
        self.labelIceEnsembleSize.setObjectName("labelIceEnsembleSize")
        self.formLayoutIceemdanTeoKeyParams.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelIceEnsembleSize)
        self.editIceemdanTeo_ensemble_size = QtWidgets.QLineEdit(parent=self.pageIceemdanTeoKeyParams)
        self.editIceemdanTeo_ensemble_size.setObjectName("editIceemdanTeo_ensemble_size")
        self.formLayoutIceemdanTeoKeyParams.setWidget(1, QtWidgets.QFormLayout.ItemRole.FieldRole, self.editIceemdanTeo_ensemble_size)

        self.labelIceNoiseStrength = QtWidgets.QLabel(parent=self.pageIceemdanTeoKeyParams)
        self.labelIceNoiseStrength.setObjectName("labelIceNoiseStrength")
        self.formLayoutIceemdanTeoKeyParams.setWidget(2, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelIceNoiseStrength)
        self.editIceemdanTeo_noise_strength = QtWidgets.QLineEdit(parent=self.pageIceemdanTeoKeyParams)
        self.editIceemdanTeo_noise_strength.setObjectName("editIceemdanTeo_noise_strength")
        self.formLayoutIceemdanTeoKeyParams.setWidget(2, QtWidgets.QFormLayout.ItemRole.FieldRole, self.editIceemdanTeo_noise_strength)

        self.labelIceSigmaK = QtWidgets.QLabel(parent=self.pageIceemdanTeoKeyParams)
        self.labelIceSigmaK.setObjectName("labelIceSigmaK")
        self.formLayoutIceemdanTeoKeyParams.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelIceSigmaK)
        self.editIceemdanTeo_sigma_k = QtWidgets.QLineEdit(parent=self.pageIceemdanTeoKeyParams)
        self.editIceemdanTeo_sigma_k.setObjectName("editIceemdanTeo_sigma_k")
        self.formLayoutIceemdanTeoKeyParams.setWidget(3, QtWidgets.QFormLayout.ItemRole.FieldRole, self.editIceemdanTeo_sigma_k)

        self.labelIcePickMode = QtWidgets.QLabel(parent=self.pageIceemdanTeoKeyParams)
        self.labelIcePickMode.setObjectName("labelIcePickMode")
        self.formLayoutIceemdanTeoKeyParams.setWidget(4, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelIcePickMode)
        self.comboIceemdanTeo_pick_mode = QtWidgets.QComboBox(parent=self.pageIceemdanTeoKeyParams)
        self.comboIceemdanTeo_pick_mode.setObjectName("comboIceemdanTeo_pick_mode")
        self.comboIceemdanTeo_pick_mode.addItem("")
        self.comboIceemdanTeo_pick_mode.addItem("")
        self.comboIceemdanTeo_pick_mode.addItem("")
        self.formLayoutIceemdanTeoKeyParams.setWidget(4, QtWidgets.QFormLayout.ItemRole.FieldRole, self.comboIceemdanTeo_pick_mode)

        self.labelIceCrossConsecutive = QtWidgets.QLabel(parent=self.pageIceemdanTeoKeyParams)
        self.labelIceCrossConsecutive.setObjectName("labelIceCrossConsecutive")
        self.formLayoutIceemdanTeoKeyParams.setWidget(5, QtWidgets.QFormLayout.ItemRole.LabelRole, self.labelIceCrossConsecutive)
        self.editIceemdanTeo_cross_consecutive = QtWidgets.QLineEdit(parent=self.pageIceemdanTeoKeyParams)
        self.editIceemdanTeo_cross_consecutive.setObjectName("editIceemdanTeo_cross_consecutive")
        self.formLayoutIceemdanTeoKeyParams.setWidget(5, QtWidgets.QFormLayout.ItemRole.FieldRole, self.editIceemdanTeo_cross_consecutive)

        self.btnOpenIceemdanTeoParamDialog = QtWidgets.QPushButton(parent=self.pageIceemdanTeoKeyParams)
        self.btnOpenIceemdanTeoParamDialog.setMinimumHeight(36)
        self.btnOpenIceemdanTeoParamDialog.setObjectName("btnOpenIceemdanTeoParamDialog")
        self.formLayoutIceemdanTeoKeyParams.setWidget(6, QtWidgets.QFormLayout.ItemRole.FieldRole, self.btnOpenIceemdanTeoParamDialog)

        self.stackedAlgorithmKeyParams.addWidget(self.pageIceemdanTeoKeyParams)
        self.verticalLayout_3.addWidget(self.stackedAlgorithmKeyParams)

        self.btnRunDetection = QtWidgets.QPushButton(parent=self.algorithmCard)
        self.btnRunDetection.setMinimumHeight(44)
        self.btnRunDetection.setObjectName("btnRunDetection")
        self.verticalLayout_3.addWidget(self.btnRunDetection)

        self.algorithmCardLayout.addLayout(self.verticalLayout_3)
        self.verticalLayout_2.addWidget(self.algorithmCard, 0)
        self.verticalLayout_2.addStretch(1)

        self.rightScrollArea.setWidget(self.rightScrollContents)

        self.horizontalLayout_3.addWidget(self.mainSplitter)
        self.mainSplitter.setStretchFactor(0, 1)
        self.mainSplitter.setStretchFactor(1, 0)
        self.mainSplitter.setSizes([820, 340])

        self.mainStackedWidget.addWidget(self.pageWavefrontDetection)

        self.pageReserved = QtWidgets.QWidget()
        self.pageReserved.setObjectName("pageReserved")
        self.mainStackedWidget.addWidget(self.pageReserved)

        self.horizontalLayout.addWidget(self.mainStackedWidget)
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 28))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(parent=self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)

        self.toolBar = QtWidgets.QToolBar(parent=MainWindow)
        self.toolBar.setMovable(False)
        self.toolBar.setFloatable(False)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.ToolBarArea.TopToolBarArea, self.toolBar)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        self.stackedAlgorithmKeyParams.setCurrentIndex(0)
        self.comboAlgorithm.currentIndexChanged.connect(self.stackedAlgorithmKeyParams.setCurrentIndex)
        self.comboMatchMode.currentIndexChanged.connect(self._sync_pair_keyword_enabled)
        self._sync_pair_keyword_enabled(self.comboMatchMode.currentIndex())
        self._apply_styles(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def _sync_pair_keyword_enabled(self, index: int):
        enabled = index == 1
        self.labelPairKeyword.setEnabled(enabled)
        self.editPairKeyword.setEnabled(enabled)

    def _apply_styles(self, MainWindow):
        MainWindow.setStyleSheet(
            """
            QWidget {
                background: #f2f2f2;
                color: #1c1c1c;
                font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
                font-size: 13px;
            }
            QMenuBar, QMenu, QToolBar {
                background: #ffffff;
                color: #1c1c1c;
            }
            QMenuBar {
                border-bottom: 1px solid #d8d8d8;
            }
            QToolBar {
                border: none;
                border-bottom: 1px solid #d8d8d8;
                spacing: 6px;
                padding: 6px 10px;
            }
            QSplitter::handle {
                background: #e1e1e1;
            }
            QSplitter::handle:hover {
                background: #cfcfcf;
            }
            #plotCard, #inputCard, #resultCard, #algorithmCard {
                background: #ffffff;
                border: 1px solid #dadada;
                border-radius: 12px;
            }
            #framePlotContainer {
                background: #ffffff;
                border: 1px solid #d2d2d2;
                border-radius: 10px;
            }
            QLabel {
                background: transparent;
            }
            QLabel#labelInputSectionTitle,
            QLabel#labelResultSectionTitle,
            QLabel#labelAlgorithmSectionTitle {
                color: #111111;
                font-weight: 700;
                font-size: 15px;
            }
            QLabel#valueCurrentFile,
            QLabel#valueCurrentAlgorithm,
            QLabel#valueChannelAResult,
            QLabel#valueChannelBResult,
            QLabel#valueTimeDifference,
            QLabel#valueDistanceToA,
            QLabel#valueDistanceToB,
            QLabel#valueDetectionStatus {
                color: #111111;
                font-weight: 600;
                padding: 6px 8px;
                background: #f7f7f7;
                border: 1px solid #dedede;
                border-radius: 8px;
            }
            QLineEdit, QComboBox {
                background: #ffffff;
                border: 1px solid #cfcfcf;
                border-radius: 8px;
                padding: 8px 10px;
                min-height: 20px;
                selection-background-color: #1c1c1c;
                selection-color: #ffffff;
            }
            QLineEdit:hover, QComboBox:hover {
                border-color: #a9a9a9;
            }
            QLineEdit:focus, QComboBox:focus {
                border: 1px solid #111111;
                background: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
                width: 28px;
            }
            QToolButton, QPushButton {
                background: #f5f5f5;
                border: 1px solid #d5d5d5;
                border-radius: 8px;
                padding: 8px 14px;
                font-weight: 600;
                color: #1c1c1c;
            }
            QToolButton {
                min-width: 34px;
                min-height: 34px;
                padding: 0px;
            }
            QToolButton:hover, QPushButton:hover {
                background: #ebebeb;
                border-color: #bfbfbf;
            }
            QToolButton:pressed, QPushButton:pressed {
                background: #e1e1e1;
            }
            QPushButton#btnRunDetection {
                background: #111111;
                color: #ffffff;
                border: 1px solid #111111;
                font-size: 14px;
                font-weight: 700;
                min-height: 42px;
            }
            QPushButton#btnRunDetection:hover {
                background: #000000;
                border-color: #000000;
            }
            QScrollArea {
                background: transparent;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #bcbcbc;
                border-radius: 5px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9f9f9f;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background: transparent;
                height: 0px;
            }
            QFrame#line {
                background: #dddddd;
                max-height: 1px;
            }
            """
        )

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "双端波头分析"))

        self.editInputDirA.setProperty("fieldKey", _translate("MainWindow", "input_dir_a"))
        self.editInputDirA.setPlaceholderText(_translate("MainWindow", "请选择 A 端数据目录"))
        self.editInputDirB.setProperty("fieldKey", _translate("MainWindow", "input_dir_b"))
        self.editInputDirB.setPlaceholderText(_translate("MainWindow", "请选择 B 端数据目录"))
        self.editPairKeyword.setProperty("fieldKey", _translate("MainWindow", "pair_keyword"))
        self.editPairKeyword.setPlaceholderText(_translate("MainWindow", "例如 11-52-46_933"))
        self.editOutputDir.setProperty("fieldKey", _translate("MainWindow", "output_dir"))
        self.editOutputDir.setPlaceholderText(_translate("MainWindow", "请选择结果保存目录"))
        self.editSensorDistanceM.setProperty("fieldKey", _translate("MainWindow", "sensor_distance_m"))
        self.editSensorDistanceM.setPlaceholderText(_translate("MainWindow", "m"))
        self.editSamplingFreqMHz.setProperty("fieldKey", _translate("MainWindow", "wave_speed_mps"))
        self.editSamplingFreqMHz.setPlaceholderText(_translate("MainWindow", "m/s"))

        self.labelInputSectionTitle.setText(_translate("MainWindow", "输入与数据配置"))
        self.labelInputDirA.setText(_translate("MainWindow", "数据 A 目录"))
        self.labelInputDirB.setText(_translate("MainWindow", "数据 B 目录"))
        self.labelMatchMode.setText(_translate("MainWindow", "配对模式"))
        self.labelPairKeyword.setText(_translate("MainWindow", "配对关键词"))
        self.labelOutputDir.setText(_translate("MainWindow", "结果保存目录"))
        self.labelSensorDistance.setText(_translate("MainWindow", "设备间距 (m)"))
        self.labelSamplingFreq.setText(_translate("MainWindow", "波速 (m/s)"))

        self.btnBrowseInputDirA.setText(_translate("MainWindow", "..."))
        self.btnBrowseInputDirA.setProperty("targetFieldKey", _translate("MainWindow", "input_dir_a"))
        self.btnBrowseInputDirB.setText(_translate("MainWindow", "..."))
        self.btnBrowseInputDirB.setProperty("targetFieldKey", _translate("MainWindow", "input_dir_b"))
        self.btnBrowseOutputDir.setText(_translate("MainWindow", "..."))
        self.btnBrowseOutputDir.setProperty("targetFieldKey", _translate("MainWindow", "output_dir"))

        self.comboMatchMode.setProperty("fieldKey", _translate("MainWindow", "pair_mode"))
        self.comboMatchMode.setItemText(0, _translate("MainWindow", "自动匹配"))
        self.comboMatchMode.setItemText(1, _translate("MainWindow", "指定关键词"))

        self.labelResultSectionTitle.setText(_translate("MainWindow", "结果展示"))
        self.labelCurrentFile.setText(_translate("MainWindow", "当前文件"))
        self.valueCurrentFile.setText(_translate("MainWindow", "-"))
        self.labelCurrentAlgorithm.setText(_translate("MainWindow", "当前算法"))
        self.valueCurrentAlgorithm.setText(_translate("MainWindow", "-"))
        self.labelChannelAResult.setText(_translate("MainWindow", "A 通道结果"))
        self.valueChannelAResult.setText(_translate("MainWindow", "-"))
        self.labelChannelBResult.setText(_translate("MainWindow", "B 通道结果"))
        self.valueChannelBResult.setText(_translate("MainWindow", "-"))
        self.labelTimeDifference.setText(_translate("MainWindow", "波头时间差 (us)"))
        self.valueTimeDifference.setText(_translate("MainWindow", "-"))
        self.labelDistanceToA.setText(_translate("MainWindow", "故障离 A 端距离 (m)"))
        self.valueDistanceToA.setText(_translate("MainWindow", "-"))
        self.labelDistanceToB.setText(_translate("MainWindow", "故障离 B 端距离 (m)"))
        self.valueDistanceToB.setText(_translate("MainWindow", "-"))
        self.labelDetectionStatus.setText(_translate("MainWindow", "检测状态"))
        self.valueDetectionStatus.setText(_translate("MainWindow", "待运行"))

        self.btnSaveResult.setText(_translate("MainWindow", "保存结果"))
        self.btnSaveResult.setProperty("actionKey", _translate("MainWindow", "save_result"))
        self.btnClearResult.setText(_translate("MainWindow", "清除记录"))
        self.btnClearResult.setProperty("actionKey", _translate("MainWindow", "clear_result"))

        self.labelAlgorithmSectionTitle.setText(_translate("MainWindow", "算法选择与关键参数"))
        self.comboAlgorithm.setProperty("fieldKey", _translate("MainWindow", "algorithm_id"))
        self.comboAlgorithm.setToolTip(_translate("MainWindow", "Select algorithm and switch the corresponding key-parameter page"))
        self.comboAlgorithm.setItemText(0, _translate("MainWindow", "RDP + Local AIC"))
        self.comboAlgorithm.setItemText(1, _translate("MainWindow", "RDP + Global ICEEMDAN-TEO"))

        self.stackedAlgorithmKeyParams.setToolTip(_translate("MainWindow", "Key parameter pages for each algorithm"))

        self.pageRdpLocalAicKeyParams.setProperty("algorithmId", _translate("MainWindow", "rdp_local_aic"))
        self.labelRdpPreN.setText(_translate("MainWindow", "Pre-noise Samples"))
        self.editRdpLocalAic_pre_n.setProperty("algorithmId", _translate("MainWindow", "rdp_local_aic"))
        self.editRdpLocalAic_pre_n.setProperty("paramKey", _translate("MainWindow", "pre_n"))
        self.editRdpLocalAic_pre_n.setProperty("paramType", _translate("MainWindow", "int"))

        self.labelRdpRoughK.setText(_translate("MainWindow", "Rough Threshold K"))
        self.editRdpLocalAic_rough_k.setProperty("algorithmId", _translate("MainWindow", "rdp_local_aic"))
        self.editRdpLocalAic_rough_k.setProperty("paramKey", _translate("MainWindow", "rough_k"))
        self.editRdpLocalAic_rough_k.setProperty("paramType", _translate("MainWindow", "float"))

        self.labelRdpEpsilon.setText(_translate("MainWindow", "RDP Epsilon"))
        self.editRdpLocalAic_rdp_epsilon.setProperty("algorithmId", _translate("MainWindow", "rdp_local_aic"))
        self.editRdpLocalAic_rdp_epsilon.setProperty("paramKey", _translate("MainWindow", "rdp_epsilon"))
        self.editRdpLocalAic_rdp_epsilon.setProperty("paramType", _translate("MainWindow", "float"))

        self.labelRdpSearchLeftUs.setText(_translate("MainWindow", "Search Left (us)"))
        self.editRdpLocalAic_search_left_us.setProperty("algorithmId", _translate("MainWindow", "rdp_local_aic"))
        self.editRdpLocalAic_search_left_us.setProperty("paramKey", _translate("MainWindow", "search_left_us"))
        self.editRdpLocalAic_search_left_us.setProperty("paramType", _translate("MainWindow", "float"))

        self.labelRdpSearchRightUs.setText(_translate("MainWindow", "Search Right (us)"))
        self.editRdpLocalAic_search_right_us.setProperty("algorithmId", _translate("MainWindow", "rdp_local_aic"))
        self.editRdpLocalAic_search_right_us.setProperty("paramKey", _translate("MainWindow", "search_right_us"))
        self.editRdpLocalAic_search_right_us.setProperty("paramType", _translate("MainWindow", "float"))

        self.labelRdpAmpK.setText(_translate("MainWindow", "Amplitude Threshold K"))
        self.editRdpLocalAic_amp_k.setProperty("algorithmId", _translate("MainWindow", "rdp_local_aic"))
        self.editRdpLocalAic_amp_k.setProperty("paramKey", _translate("MainWindow", "amp_k"))
        self.editRdpLocalAic_amp_k.setProperty("paramType", _translate("MainWindow", "float"))

        self.btnOpenRdpLocalAicParamDialog.setText(_translate("MainWindow", "打开 RDP + Local AIC 完整参数表"))
        self.btnOpenRdpLocalAicParamDialog.setProperty("algorithmId", _translate("MainWindow", "rdp_local_aic"))

        self.pageIceemdanTeoKeyParams.setProperty("algorithmId", _translate("MainWindow", "rdp_global_iceemdan_teo"))
        self.labelIceRdpEpsilon.setText(_translate("MainWindow", "RDP Epsilon"))
        self.editIceemdanTeo_rdp_epsilon.setProperty("algorithmId", _translate("MainWindow", "rdp_global_iceemdan_teo"))
        self.editIceemdanTeo_rdp_epsilon.setProperty("paramKey", _translate("MainWindow", "rdp_epsilon"))
        self.editIceemdanTeo_rdp_epsilon.setProperty("paramType", _translate("MainWindow", "float"))

        self.labelIceEnsembleSize.setText(_translate("MainWindow", "ICEEMDAN Ensemble Size"))
        self.editIceemdanTeo_ensemble_size.setProperty("algorithmId", _translate("MainWindow", "rdp_global_iceemdan_teo"))
        self.editIceemdanTeo_ensemble_size.setProperty("paramKey", _translate("MainWindow", "ensemble_size"))
        self.editIceemdanTeo_ensemble_size.setProperty("paramType", _translate("MainWindow", "int"))

        self.labelIceNoiseStrength.setText(_translate("MainWindow", "ICEEMDAN Noise Strength"))
        self.editIceemdanTeo_noise_strength.setProperty("algorithmId", _translate("MainWindow", "rdp_global_iceemdan_teo"))
        self.editIceemdanTeo_noise_strength.setProperty("paramKey", _translate("MainWindow", "noise_strength"))
        self.editIceemdanTeo_noise_strength.setProperty("paramType", _translate("MainWindow", "float"))

        self.labelIceSigmaK.setText(_translate("MainWindow", "Threshold Sigma K"))
        self.editIceemdanTeo_sigma_k.setProperty("algorithmId", _translate("MainWindow", "rdp_global_iceemdan_teo"))
        self.editIceemdanTeo_sigma_k.setProperty("paramKey", _translate("MainWindow", "sigma_k"))
        self.editIceemdanTeo_sigma_k.setProperty("paramType", _translate("MainWindow", "float"))

        self.labelIcePickMode.setText(_translate("MainWindow", "Head Pick Mode"))
        self.comboIceemdanTeo_pick_mode.setProperty("algorithmId", _translate("MainWindow", "rdp_global_iceemdan_teo"))
        self.comboIceemdanTeo_pick_mode.setProperty("paramKey", _translate("MainWindow", "pick_mode"))
        self.comboIceemdanTeo_pick_mode.setProperty("paramType", _translate("MainWindow", "choice"))
        self.comboIceemdanTeo_pick_mode.setItemText(0, _translate("MainWindow", "first_cross"))
        self.comboIceemdanTeo_pick_mode.setItemText(1, _translate("MainWindow", "max"))
        self.comboIceemdanTeo_pick_mode.setItemText(2, _translate("MainWindow", "first_sig_slope"))

        self.labelIceCrossConsecutive.setText(_translate("MainWindow", "Cross Consecutive"))
        self.editIceemdanTeo_cross_consecutive.setProperty("algorithmId", _translate("MainWindow", "rdp_global_iceemdan_teo"))
        self.editIceemdanTeo_cross_consecutive.setProperty("paramKey", _translate("MainWindow", "cross_consecutive"))
        self.editIceemdanTeo_cross_consecutive.setProperty("paramType", _translate("MainWindow", "int"))

        self.btnOpenIceemdanTeoParamDialog.setText(_translate("MainWindow", "打开 RDP + Global ICEEMDAN-TEO 完整参数表"))
        self.btnOpenIceemdanTeoParamDialog.setProperty("algorithmId", _translate("MainWindow", "rdp_global_iceemdan_teo"))

        self.btnRunDetection.setText(_translate("MainWindow", "开始检测"))
        self.btnRunDetection.setProperty("actionKey", _translate("MainWindow", "run_detection"))

        self.menu.setTitle(_translate("MainWindow", "波头识别"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
