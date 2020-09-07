import cv2
path = '/media/New Volume/DiabeticRetinopathy train/3/'
gamma = 1.44
alpha = 2.5
beta = 40

image = cv2.imread(path+'2767_left.jpeg')

# cv operations for changing alpha beta and gamma channels for intensity
# image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# lookUpTable = np.empty((1,256), np.uint8)
# for i in range(256):
#     lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
# image = cv2.LUT(image, lookUpTable)

# applying bioinspired_retina for enhancing image quality for dark images.
retina = cv2.bioinspired_Retina.create((image.shape[1], image.shape[0]))

# -> load retina parameters from a xml file : here we load the default parameters that we just wrote to file
retina.setup('retinaParams.xml')

cv2.imshow('input frame', cv2.resize(image, (1000, 1000)))
# run retina on the input image
retina.run(image)
# grab retina outputs
retinaOut_parvo = cv2.resize(retina.getParvo(), (1000, 1000))
# draw retina outputs
cv2.imshow('retina parvo out', retinaOut_parvo)
# wait a little to let the time for figures to be drawn
cv2.waitKey(0)
