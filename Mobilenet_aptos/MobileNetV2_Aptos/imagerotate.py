import cv2
import os

def main():
    path = "/media/sandeep/New Volume/Angel Billy/Final dataset/4/tmp/"
    for filename in os.listdir(path):
        try:
            if 'png' in str(filename):
                dst = "100_" + str(i) + ".png"
            elif 'JPG' or 'jpg' in str(filename):
                dst = "1_" + str(i) + ".jpg"
            elif 'jpeg' in str(filename):
                dst = "10_" + str(i) + ".jpeg"

            src = path+'_'+dst
            dst = path+dst

            im = cv2.imread(filename)
            cv2.imshow('image',im)
            cv2.waitKey(0)
            flipVertical = cv2.flip(im, 0)
            flipHorizontal = cv2.flip(im, 1)
            cv2.imwrite(src, flipHorizontal)
            cv2.imwrite(dst, flipVertical)

        except: FileNotFoundError



# Driver Code
if __name__ == '__main__':
    # Calling main() function
    main()
