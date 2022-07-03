import cv2 as cv
import imutils as iu
import pytesseract as pt
read_image_format = cv.imread('do.jpg')
read_image_format = iu.resize(read_image_format, width=640 )
cv.imshow("real image", read_image_format)
gray_image = cv.bilateralFilter(read_image_format, 11, 17, 17)
cv.imshow("smoothened image", gray_image)
edged = cv.Canny(gray_image, 30, 200) 
cv.imshow("edged image", edged)
cv.waitKey(0)
cnts,new = cv.findContours(edged.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
read_image_format_copy =read_image_format.copy()
cv.drawContours(read_image_format_copy,cnts,-1,(0,255,0),3)
cv.imshow("contours",read_image_format_copy)
cv.waitKey(0)
cnts = sorted(cnts, key = cv.contourArea, reverse = True) [:30]
screenCnt = None
image2 = read_image_format.copy()
cv.drawContours(image2,cnts,-1,(0,255,0),3)
cv.waitKey(0)
i=7
for c in cnts:
        perimeter = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4: 
                screenCnt = approx
                a,b,c,d = cv.boundingRect(c)
                new_img=read_image_format[b:b+d,a:a+c]
                cv.imwrite('./'+str(i)+'.png',new_img)
                i+=1
                break
        cv.drawContours(read_image_format, [screenCnt], -1, (0, 255, 0), 3)
        cv.imshow("read_image_format with detected license plate", read_image_format)
        cv.waitKey(0)
        Cropped_loc = './ar.png'
cv.imshow("cropped", cv.imread(Cropped_loc))
plate = pt.image_to_string(Cropped_loc, lang='eng')
print("plate is:", plate)
cv.waitKey(0)
cv.destroyAllWindows()