# mode 0 --> wound boundary
# mode 1 --> wound region

class DFUImage(object):
    def __init__(self, base64Format, mode):
        self.base64Format = base64Format
        self.mode = mode
