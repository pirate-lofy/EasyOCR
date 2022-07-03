import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image, ImageChops
import arabic_reshaper
from bidi.algorithm import get_display
import pandas as pd

blank = np.zeros((30, 900, 3), dtype='uint8')
blank.fill(255)

emptyId = cv.imread('emptyId.jpg')
print(emptyId.shape)
#cv.imshow("id", emptyId[75:105, 150:278, :])

yDim = ['fName', 'lName', 'sAddress', 'rAddress', 'idNumber']
yDict ={'fName':75, 'lName':105, 'sAddress': 135, 'rAddress': 165, 'idNumber': 225 }



#cv.waitKey(0)



# print(pd.read_csv("names.txt"))
names = pd.read_csv("names.txt")
name_numpy = names.to_numpy()

fontpath = "arialBlack.ttf"
font = ImageFont.truetype(fontpath, 24)
b, g, r, a = 0, 0, 0, 0
count = 0
fileNameList = []
wordList = []

#image_numpy = name_numpy[:, 0]
name_numpy = name_numpy[:, 0]
# print(name_numpy)

idNumbers = '٠١٢٣٤٥٦٧٨٩'

for i in range(1000):
    reshaped_text = arabic_reshaper.reshape(name_numpy[count])
    reshaped_text = get_display(reshaped_text, base_dir='R')

    nametype = random.choice(yDim)
    labelString = ''

    if nametype ==  'fName':
        labelString = random.choice(name_numpy)
        reshaped_text = arabic_reshaper.reshape(labelString)
        reshaped_text = get_display(reshaped_text, base_dir='R')
    elif nametype ==  'lName':
        labelString = random.choice(name_numpy)+ ' ' + random.choice(name_numpy) + ' ' + random.choice(name_numpy)
        reshaped_text = arabic_reshaper.reshape(labelString)
        reshaped_text = get_display(reshaped_text, base_dir='R')
    elif nametype == 'idNumber':
        labelString = ''.join(random.choice(idNumbers) for i in range(14))
        reshaped_text = arabic_reshaper.reshape(labelString )
        reshaped_text = get_display(reshaped_text, base_dir='R')
    else:
        labelString = random.choice(name_numpy)
        reshaped_text = arabic_reshaper.reshape(labelString)
        reshaped_text = get_display(reshaped_text, base_dir='R')

    img_pil = Image.fromarray(blank)
    draw = ImageDraw.Draw(img_pil)
    draw.text((10, 0),  reshaped_text, font=font, fill=(b, g, r, a))
    img_pil = np.array(img_pil)
    img_pil = cv.cvtColor(img_pil, cv.COLOR_BGR2GRAY)
    thresholdd, im_pil = cv.threshold(img_pil, 127, 256, cv.THRESH_BINARY_INV)
    rows, cols = np.nonzero(im_pil)
    rightmost = cols.max()
    imgWidth = rightmost
    img = np.array(img_pil)



    ystart = yDict[nametype]
    totalWidth = imgWidth+10
    im3 = emptyId[ystart:ystart + 30, 278-totalWidth:278, :]#img[:, 0:imgWidth+10]
    img_pil2 = Image.fromarray(im3)
    draw = ImageDraw.Draw(img_pil2)
    draw.text((10, 0),  reshaped_text, font=font, fill=(b, g, r, a))
    im2 = np.array(img_pil2)
    imgname = "NourHelalSample/"+ str(count + 1) + ".png"
    imgNameWithoutPath = str(count + 1) + ".png"
    fileNameList.append(imgNameWithoutPath)
    wordList.append(labelString)#name_numpy[count])
    count = count + 1
    cv.imwrite(imgname, im2)

dict = {'filename': fileNameList, 'words': wordList}#name_numpy.tolist()}
df = pd.DataFrame(dict)
df.to_csv('NourHelalSample/labels.csv', index=False)

# dict = {'filename': fileNameList, 'words': wordList}#name_numpy.tolist()}
# df = pd.DataFrame(dict)
# #saving the dataframe
# df.to_csv('labels.csv', index=False)
# writer = pd.ExcelWriter("labels2" + '.xlsx')
# df.to_excel(writer, sheet_name='Sheet1', index=False, encoding="utf-8-sig")

# # Close the Pandas Excel writer and output the Excel file.
# writer.save()

# Display
# cv.imshow("res", im2);cv.waitKey();cv.destroyAllWindows()
# cv.imshow('blank', blank)
# cv.waitKey(0)
