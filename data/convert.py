from pyffmpeg import FFmpeg

SOURCE = "voice.mp3"
TEMP = "temp.wav"

ff = FFmpeg()
ff.convert(SOURCE, TEMP)