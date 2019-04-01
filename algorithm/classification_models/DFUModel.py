class DFUModel(object):
    def __init__(self, denseNet=None, ann=None, featureScalar=None, svdScalar=None, inputWidth=224, inputHeight=224):
        self.__denseNetModel = denseNet
        self.__annModel = ann
        self.__inputWidth = inputWidth
        self.__inputHeight = inputHeight
        self.__featureScalar = featureScalar
        self.__svdScalar = svdScalar

    def setDenseNetModel(self, denseNet):
        self.__denseNetModel = denseNet

    def setANNModel(self,ann):
        self.__annModel = ann

    def setInputWidth(self, inputWidth):
        self.__inputWidth = inputWidth

    def setInputHeight(self, inputHeight):
        self.__inputHeight = inputHeight

    def setFeatureScalar(self, featureScalar):
        self.__featureScalar = featureScalar

    def setSVDScalar(self, svdScalar):
        self.__svdScalar = svdScalar

    def getDenseNetModel(self):
        return self.__denseNetModel

    def getANNModel(self):
        return self.__annModel

    def getInputWidth(self):
        return self.__inputWidth

    def getInputHeight(self):
        return self.__inputHeight

    def getFeatureScalar(self):
        return self.__featureScalar

    def getSVDScalar(self):
        return self.__svdScalar
