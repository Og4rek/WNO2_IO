import cv2
import pyaudio
import wave
from multiprocessing import Process, Value
import numpy as np
import os

FORMAT = pyaudio.paInt16
CHANNELS = 1
WAVE_OUTPUT_FILENAME = "audio.wav"
FPS = 10
audio = pyaudio.PyAudio()

# Create an object to read from camera
video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(video.get(3))
frame_height = int(video.get(4))

size = (frame_width, frame_height)

kernel = np.ones((5, 5), np.uint8)


def video_record_dylacja(RATE, CHUNK, RECORD_SECONDS):
    result = cv2.VideoWriter('dylacja.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             FPS, size)

    for i in range(0, int(RATE.value / CHUNK.value * RECORD_SECONDS.value)):
        ret, frame = video.read()
        if ret:
            frame = cv2.dilate(frame, kernel)
            result.write(frame)
            cv2.imshow('Dylacja', frame)
        else:
            break
    result.release()

    # Closes all the frames
    cv2.destroyAllWindows()


def video_record_erozja(RATE, CHUNK, RECORD_SECONDS):
    result = cv2.VideoWriter('erozja.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             FPS, size)

    for i in range(0, int(RATE.value / CHUNK.value * RECORD_SECONDS.value)):
        ret, frame = video.read()
        if ret:
            frame = cv2.erode(frame, kernel)
            result.write(frame)
            cv2.imshow('Erozja', frame)
        else:
            break
    result.release()

    cv2.destroyAllWindows()


def video_record_sobel(RATE, CHUNK, RECORD_SECONDS):
    result = cv2.VideoWriter('sobel.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             FPS, size)

    for i in range(0, int(RATE.value / CHUNK.value * RECORD_SECONDS.value)):
        ret, frame = video.read()
        if ret:
            sobel = cv2.Sobel(frame, cv2.CV_8U, 1, 1, 5)
            result.write(sobel)
            cv2.imshow('Filtr sobela', sobel)
        else:
            break
    result.release()

    cv2.destroyAllWindows()


def video_record(RATE, CHUNK, RECORD_SECONDS):
    result = cv2.VideoWriter('video.avi',
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             FPS, size)

    for i in range(0, int(RATE.value / CHUNK.value * RECORD_SECONDS.value)):
        ret, frame = video.read()

        if ret:
            result.write(frame)
            cv2.imshow('video_record', frame)

        else:
            break

    video.release()
    result.release()

    cv2.destroyAllWindows()


def audio_record(RATE, CHUNK, RECORD_SECONDS):
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE.value, input=True,
                        frames_per_buffer=CHUNK.value)

    frames = []

    for i in range(0, int(RATE.value / CHUNK.value * RECORD_SECONDS.value)):
        data = stream.read(CHUNK.value)
        frames.append(data)

    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()

    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE.value)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()


if __name__ == "__main__":
    # initial balance (in shared memory)
    RATE = Value('i', 10240)
    CHUNK = Value('i', 1024)
    RECORD_SECONDS = Value('i', 10)

    p1 = Process(target=audio_record, args=(RATE, CHUNK, RECORD_SECONDS,))
    p2 = Process(target=video_record, args=(RATE, CHUNK, RECORD_SECONDS,))
    p3 = Process(target=video_record_sobel, args=(RATE, CHUNK, RECORD_SECONDS,))
    p4 = Process(target=video_record_erozja, args=(RATE, CHUNK, RECORD_SECONDS,))
    p5 = Process(target=video_record_dylacja, args=(RATE, CHUNK, RECORD_SECONDS,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()

    command = "ffmpeg -i video.avi -i audio.wav -c copy output.avi"
    os.system(command)
