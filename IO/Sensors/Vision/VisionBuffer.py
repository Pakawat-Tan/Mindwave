class VisionBuffer:
    def __init__(self):
        self.buffer = []

    def add_frame(self, frame):
        self.buffer.append(frame)

    def get_frames(self):
        return self.buffer