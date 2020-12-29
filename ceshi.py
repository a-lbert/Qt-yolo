class LoadStreams:  # multiple IP or RTSP cameras
    # sources='0'
    def __init__(self, sources='streams.txt', img_size=640):
        self.i=0
    def update(self):
        self.i+=1

    def __iter__(self):

        return self

    def __next__(self):

        self.i+=1

        return self.i

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

a = LoadStreams()
print(next(a))
print(next(a))
print(next(a))
print(next(a))

