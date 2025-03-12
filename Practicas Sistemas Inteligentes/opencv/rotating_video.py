import cv2
import ffmpeg

image = cv2.imread('../resources/triangulo.png')
height, width = image.shape[:2]
center = (width/2, height/2)
frame_size = (height, width)

# Initialize video writer object
output = cv2.VideoWriter('rotacion.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)

for i in range(360,0,-1):
	# using cv2.getRotationMatrix2D() to get the rotation matrix
	rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=i, scale=1)
	# rotate the image using cv2.warpAffine
	rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
	frame = rotated_image
	output.write(frame)

for i in range(0,360):
	rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=i, scale=1)
	# rotate the image using cv2.warpAffine
	rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))
	frame = rotated_image
	output.write(frame)

output.release()

audio = ffmpeg.input('resources/no_mana.mp4')
audio.output('audio.mp3', acodec='mp3').run()

video = ffmpeg.input('rotacion.avi')
audio = ffmpeg.input('audio.mp3')
ffmpeg.concat(video, audio, v=1, a=1).output('finished_video.mp4').run()
