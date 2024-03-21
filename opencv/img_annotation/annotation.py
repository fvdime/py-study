import cv2
import matplotlib.pyplot as plt

# DRAWING A LINE
image = cv2.imread("Apollo_11_Launch.jpg")
imgLine = image.copy()

cv2.line(imgLine, (200, 100), (400, 100), (0, 255, 255), thickness=5, lineType=cv2.LINE_AA);

plt.imshow(imgLine[:,:,::-1])
plt.show()

# DRAWING A CIRCLE
imgCircle = image.copy()
cv2.circle(imgCircle, (900, 500), 100, (0, 0, 255), thickness=5, lineType=cv2.LINE_AA)

plt.imshow(imgCircle[:,:,::-1])
plt.show()

# DRAWING A RECTANGLE
imgRectangle = image.copy()

cv2.rectangle(imgRectangle, (500, 100), (700, 600), (255, 0, 255), lineType=cv2.LINE_AA)

plt.imshow(imgRectangle[:,:,::-1])
plt.show()

# ADDING TEXT
imgText = image.copy()
text="Apollo 11 Saturn V Launch"
fontScale = 2.3
fontFace = cv2.FONT_HERSHEY_COMPLEX
fontColor = (255, 0, 255)
fontThickness = 2

cv2.putText(imgText, text, (200, 700), fontFace, fontScale, fontColor, fontThickness, cv2.LINE_AA);

plt.imshow(imgText[:, :, ::-1])
plt.show()