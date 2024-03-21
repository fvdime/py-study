import cv2
import matplotlib.pyplot as plt

cb_img = cv2.imread("checkerboard_color.png")
coke_img = cv2.imread("coca-cola-logo.png")

cb_img2 = cv2.imread("checkerboard_18x18.png", 0)

# Printing the image data (pixel values), element of a 2D numpy array. Each pixel value is 8-bits [0,255]
print(cb_img2)

# Display Image Attributes

# print size of the img
print("Image size (H, W) is:", cb_img.shape)

# print data type of the img
print("Data type of image is:", cb_img.dtype)

# Display Image using Matplotlib
plt.imshow(cb_img2)
# Matplotlib uses different color maps. 

#setting color map to gray scale got proper rendering
plt.imshow(cb_img2, cmap="gray")

plt.imshow(coke_img)
plt.show()

#SPLITTING AND MERGING COLOR CHANNELS
img_NZ_lake = cv2.imread("New_Zealand_Lake.jpg", cv2.IMREAD_COLOR)
# splitting the img into the B, G, R components
b, g, r = cv2.split(img_NZ_lake)

#showing the channels 
plt.figure(figsize=[20, 5])

plt.subplot(141);
plt.imshow(r, cmap="gray");
plt.title("Red Channel")
plt.show()

plt.subplot(142);
plt.imshow(g, cmap="gray");
plt.title("Green Channel")
plt.show()

plt.subplot(143);
plt.imshow(b, cmap="gray");
plt.title("Blue Channel")
plt.show()

#merge the individual channels into BGR image
mergedImage = cv2.merge((b, g, r))

plt.subplot(144)
plt.imshow(mergedImage[:, :, ::-1])
plt.title("Merged Output")
plt.show()


#CHANGING FROM BGR TO RGB
img_NZ_lake = cv2.imread("New_Zealand_Lake.jpg", cv2.IMREAD_COLOR)
img_NZ_rgb = cv2.cvtColor(img_NZ_lake, cv2.COLOR_BGR2RGB)
plt.imshow(img_NZ_rgb)
plt.show()

#CHANGING TO HSV COLOR SPACE
img_NZ_lake = cv2.imread("New_Zealand_Lake.jpg", cv2.IMREAD_COLOR)
img_NZ_rgb = cv2.cvtColor(img_NZ_lake, cv2.COLOR_BGR2RGB)

img_hsv = cv2.cvtColor(img_NZ_lake, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img_hsv)

plt.figure(figsize=[20,5])
plt.subplot(141)
plt.imshow(h, cmap="gray")
plt.title("H Channel")
plt.show()

plt.subplot(141)
plt.imshow(s, cmap="gray")
plt.title("S Channel")
plt.show()

plt.subplot(141)
plt.imshow(v, cmap="gray")
plt.title("V Channel")
plt.show()

plt.subplot(141)
plt.imshow(img_NZ_rgb)
plt.title("ORIGINAL")
plt.show()

# Matplotlib expects the image in rgb format whereas OpenCV stores images in BGR format
coke_img_channels_reversed = coke_img[:, :, ::-1]
plt.imshow(coke_img_channels_reversed)
plt.show()

plt.imshow(cb_img)
plt.title("matplotlib imshow")
plt.show()

# SAVING IMAGES 
cv2.imwrite("New_Zealand_Lake_Saved.png", img_NZ_lake)

Image(filename='New_Zealand_Lake_SAVED.png') 

img_NZ_bgr = cv2.imread("New_Zealand_Lake_SAVED.png", cv2.IMREAD_COLOR)
print("img_NZ_bgr shape (H, W, C) is:", img_NZ_bgr.shape)

# read the image as Grayscaled
img_NZ_gry = cv2.imread("New_Zealand_Lake_SAVED.png", cv2.IMREAD_GRAYSCALE)
print("img_NZ_gry shape (H, W) is:", img_NZ_gry.shape)
#display for 8 sec
window1 = cv2.namedWindow("w1")
cv2.imshow(window1, cb_img)
cv2.waitKey(8000)
cv2.destroyWindow(window1)

window2 = cv2.namedWindow("w2")
cv2.imshow(window2, coke_img)
cv2.waitKey(8000)
cv2.destroyWindow(window2)

# display until any key is pressed
window3 = cv2.namedWindow("w3")
cv2.imshow(window3, cb_img)
cv2.waitKey(0)
cv2.destroyWindow(window3)

window4 = cv2.namedWindow("w4")

Alive = True
while Alive:
    # Use OpenCV imshow(), display until 'q' key is pressed
    cv2.imshow(window4, coke_img)
    keypress = cv2.waitKey(1)
    if keypress == ord('q'):
        Alive = False
cv2.destroyWindow(window4)

cv2.destroyAllWindows()
stop = 1