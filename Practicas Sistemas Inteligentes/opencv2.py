import cv2 
 
# Create a video capture object, in this case we are reading the video from a file
vid_capture = cv2.VideoCapture(0)

# Obtain frame size information using get() method
frame_width = int(vid_capture.get(3))
frame_height = int(vid_capture.get(4))
frame_size = (frame_width,frame_height)
fps = 20

# Initialize video writer object
output = cv2.VideoWriter('Prueba1.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)

if (vid_capture.isOpened() == False):
 print("Error opening the video file")
# Read fps and frame count
else:
 # Get frame rate information
 # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
 fps = vid_capture.get(5)
 print('Frames per second : ', fps,'FPS')
 
 # Get frame count
 # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
 frame_count = vid_capture.get(7)
 print('Frame count : ', frame_count)
 
while(vid_capture.isOpened()):
 # vid_capture.read() methods returns a tuple, first element is a bool 
 # and the second is frame
 ret, frame = vid_capture.read()
 if ret == True:
  output.write(frame)
  cv2.imshow('Frame',frame)
  # 20 is in milliseconds, try to increase the value, say 50 and observe
  key = cv2.waitKey(20)
   
  if key == ord('q'):
    cv2.imwrite('ultimoframe.jpg', frame)
    break
 else:
  break
 
# Release the video capture object
vid_capture.release()
output.release()
cv2.destroyAllWindows()
