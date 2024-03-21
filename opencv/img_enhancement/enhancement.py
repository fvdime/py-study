import cv2
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import Image

img_bgr = cv2.imread("New_Zealand_Coast.jpg", cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

Image(filename="New_Zealand_Coast.jpg")

# ADDITION OR BRIGHTNESS
matrix = np.ones(img_rgb.shape, dtype="uint8") * 50

img_rgb_brighter = cv2.add(img_rgb, matrix)
img_rgb_darker = cv2.subtract(img_rgb, matrix)

plt.figure(figsize=[18, 5])

plt.subplot(131)
plt.imshow(img_rgb_darker)
plt.title("Darker")
plt.show()

plt.subplot(132)
plt.imshow(img_rgb)
plt.title("Original")
plt.show()

plt.subplot(133)
plt.imshow(img_rgb_brighter)
plt.title("Brighter")
plt.show()

# MULTIPLICATION OR CONTRAST
matrix1 = np.ones(img_rgb.shape) * 0.8
matrix2 = np.ones(img_rgb.shape) * 1.2

img_rgb_darker = np.uint8(cv2.multiply(np.float64(img_rgb), matrix1))
img_rgb_higher = np.uint8(cv2.multiply(np.float64(img_rgb), matrix2))

plt.figure(figsize=[18, 5])

plt.subplot(141)
plt.imshow(img_rgb_darker)
plt.title("Lower Contrast")
plt.show()

plt.subplot(142)
plt.imshow(img_rgb)
plt.title("Original")
plt.show()

plt.subplot(143)
plt.imshow(img_rgb_higher)
plt.title("Higher Contrast")
plt.show()

# the values are already high, they becoming greater than 255, thus overflow issue. Handling overflow issue:
img_rgb_higher_wo = np.uint8(np.clip(cv2.multiply(np.float64(img_rgb), matrix2), 0, 255))

plt.subplot(144)
plt.imshow(img_rgb_higher_wo)
plt.title("Higher Contrast wo Issue")
plt.show()

# IMAGE THRESHOLDING
img_read = cv2.imread("building-windows.jpg", cv2.IMREAD_GRAYSCALE)
retval, img_thresh = cv2.threshold(img_read, 100, 255, cv2.THRESH_BINARY)


plt.figure(figsize=[18, 5])

plt.subplot(121)
plt.imshow(img_read, cmap="gray")
plt.title("Original")
plt.show()


plt.subplot(122)
plt.imshow(img_thresh, cmap="gray")
plt.title("Threshold")
plt.show()

print(img_thresh.shape)

# APPLICATION: SHEET MUSIC READER
img_read = cv2.imread("Piano_Sheet_music.png", cv2.IMREAD_GRAYSCALE)

retval, img_thresh_gbl_1 = cv2.threshold(img_read, 50, 255, cv2.THRESH_BINARY)
retval, img_thresh_gbl_2 = cv2.threshold(img_read, 130, 255, cv2.THRESH_BINARY)
img_thresh_adaptive = cv2.adaptiveThreshold(img_read, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 7)

plt.figure(figsize=[18, 5])

plt.subplot(221)
plt.imshow(img_read, cmap="gray")
plt.title("Original")
plt.show()


plt.subplot(222)
plt.imshow(img_thresh_gbl_1, cmap="gray")
plt.title("Threshold global: 50")
plt.show()

plt.subplot(223)
plt.imshow(img_thresh_gbl_2, cmap="gray")
plt.title("Threshold global: 130")
plt.show()


plt.subplot(224)
plt.imshow(img_thresh_adaptive, cmap="gray")
plt.title("Adaptive")
plt.show()

# BITWISE OPERATIONS
img_rec = cv2.imread("rectangle.jpg", cv2.IMREAD_GRAYSCALE)
img_cir = cv2.imread("circle.jpg", cv2.IMREAD_GRAYSCALE)

# AND OPERATOR
result_and = cv2.bitwise_and(img_rec, img_cir, mask=None)
plt.imshow(result_and, cmap="gray")
plt.show()

# OR OPERATOR
result_or = cv2.bitwise_or(img_rec, img_cir, mask=None)
plt.imshow(result_or, cmap="gray")
plt.show()

# XOR OPERATOR
result_xor = cv2.bitwise_xor(img_rec, img_cir, mask=None)
plt.imshow(result_xor, cmap="gray")
plt.show()


# APPLICATION: LOGO MANIPULATION
Image(filename="Logo_Manipulation.png")

img_bgr = cv2.imread("coca-cola-logo.png")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()

print(img_rgb.shape)

logo_w = img_rgb.shape[0]
logo_h = img_rgb.shape[1]

img_bg_bgr = cv2.imread("checkerboard_color.png")
img_bg_rgb = cv2.cvtColor(img_bg_bgr, cv2.COLOR_BGR2RGB)

aspect_ratio = logo_w / img_bg_rgb.shape[1]
dim = (logo_w, int(img_bg_rgb.shape[0] * aspect_ratio))

img_bg_rgb = cv2.resize(img_bg_rgb, dim, interpolation=cv2.INTER_AREA)

plt.imshow(img_bg_rgb)
plt.show()
print(img_bg_rgb.shape)

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

retval, img_mask = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

plt.imshow(img_mask, cmap="gray")
plt.show()
print(img_mask.shape)

img_mask_inverted = cv2.bitwise_not(img_mask)
plt.imshow(img_mask_inverted, cmap="gray")
plt.show()


img_bg = cv2.bitwise_and(img_bg_rgb, img_bg_rgb, mask=img_mask)

plt.imshow(img_bg)
plt.show()

img_foreground = cv2.bitwise_and(img_rgb, img_rgb, mask=img_mask_inverted)
plt.imshow(img_foreground)
plt.show()


result = cv2.add(img_bg, img_foreground)
plt.imshow(result)
plt.show()

cv2.imwrite("logo_final.png", result[:, :, ::-1])