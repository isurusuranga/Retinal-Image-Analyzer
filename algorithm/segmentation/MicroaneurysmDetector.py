class MicroaneurysmDetector(object):
    def __init__(self, greenComponent):
        self.__greenComponent = greenComponent

    def recognizeMA(self):
        eyeGreen = self.__greenComponent * 2.5
        eyeGray = eyeGreen.grayscale()

