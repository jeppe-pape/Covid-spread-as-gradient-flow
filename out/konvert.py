import os
import ffmpeg


folder = os.listdir()

for i, file in enumerate(folder):
	if file[-4:] == ".png":
		os.rename(file,f"{str(i).zfill(4)}.png")


o = os.system('ffmpeg -y -framerate 2 -i "%4d.png" out.mp4')