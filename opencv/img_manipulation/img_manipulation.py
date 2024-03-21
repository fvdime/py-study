import cv2
import matplotlib.pyplot as plt

# ORIGINAL CHECKERBOARD IMAGE
cb_img = cv2.imread("checkerboard_18x18.png", 0)

plt.imshow(cb_img, cmap="gray")
print(cb_img)

# ACCESSING INDIVIDUAL PIXELS
print(cb_img, [0, 0])
print(cb_img,[0, 6])

# MODIFYING IMAGE PIXELS
cb_img_copy = cb_img.copy()
cb_img_copy[2, 2] = 200
cb_img_copy[2, 3] = 200
cb_img_copy[3, 2] = 200
cb_img_copy[3, 3] = 200

plt.imshow(cb_img_copy, cmap='gray')
print(cb_img_copy)
plt.show()

# CROPPING IMAGES
img_bgr = cv2.imread("New_Zealand_Boat.jpg", cv2.IMREAD_COLOR)
img_rgb = img_bgr[:, :, ::-1]

cropped_region = img_rgb[200:400, 300:600]
plt.imshow(cropped_region)
plt.show()

# RESIZING IMAGES
resized_cropped_region_2x = cv2.resize(cropped_region, None, fx=2, fy=2)
plt.imshow(resized_cropped_region_2x)
plt.show()

width = 100
height = 200
dim = (width, height)

resized_cropped_region = cv2.resize(cropped_region, dsize=dim, interpolation=cv2.INTER_AREA)
plt.imshow(resized_cropped_region)
plt.show()

width = 100
aspect_ratio = width/cropped_region.shape[1]

height = int(cropped_region.shape[0] * aspect_ratio)
dim = (width, height)

resized_cropped_region = cv2.resize(cropped_region, dsize=dim, interpolation=cv2.INTER_AREA)
plt.imshow(resized_cropped_region)
plt.show()

resized_cropped_region_2x = resized_cropped_region_2x[:, :, ::-1]

cv2.imwrite("resized_cropped_region_2x.png", resized_cropped_region_2x)

plt.imshow(resized_cropped_region_2x)
plt.show()

# FLIPPING IMAGES

img_rgb_flipped_horizontally = cv2.flip(img_rgb, 1)
img_rgb_flipped_vertically = cv2.flip(img_rgb, 0)
img_rgb_flipped_both = cv2.flip(img_rgb, -1)

plt.figure(figsize=(18, 5))

plt.subplot(141)
plt.imshow(img_rgb_flipped_horizontally)
plt.title("Horizontal Flip")
plt.show()

plt.subplot(142)
plt.imshow(img_rgb_flipped_vertically)
plt.title("Vertical Flip")
plt.show()

plt.subplot(143)
plt.imshow(img_rgb_flipped_both)
plt.title("Both Flipped")
plt.show()

plt.subplot(144)
plt.imshow(img_rgb)
plt.title("Original")
plt.show()