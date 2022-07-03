from glob import glob
import random
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image, ImageChops
import arabic_reshaper
from bidi.algorithm import get_display
import pandas as pd

sz=(301,295)

emptyId = cv.imread('data_generation/emptyId.jpg')
emptyId=cv.resize(emptyId,sz)

sh_data='data_generation/effects/'
sh_data=glob(sh_data+'*')
#cv.imshow("id", emptyId[75:105, 150:278, :])

yDim = ['fName', 'lName', 'fAddress', 'lAddress', 'idNumber']
yDict ={'fName':75, 'lName':105, 'sAddress': 135, 'rAddress': 165, 'idNumber': 225 }

fname_pos=(210,80)
lname_pos=(110,115)
fadd_pos=(220,155)
ladd_pos=(110,180)
nn_pos=(100,260)
lengths=[(210,80),(110,115),(220,155),(110,180),(100,260)]

SHOW=True

def show(t,img):
    if not SHOW:return
    cv.imshow(t,img)

def skew(img):
    rows, cols, ch = img.shape
    pts1 = np.float32(
        [[cols*.25, rows*.95],
         [cols*.90, rows*.95],
         [cols*.10, 0],
         [cols,     0]]
    )
    pts2 = np.float32(
        [[cols*0.3, rows],
         [cols,     rows],
         [0,        2],
         [cols,     0]]
    )
    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(img, M, (cols, rows))
    return dst

def get_salt(shape):
    row,col,ch = shape
    s_vs_p = 0.5
    amount = 0.004
    out = np.zeros(shape,np.uint8)

    # just adding Salt mode as out is zeros (pepper)
    num_salt = np.ceil(amount * np.product(shape) * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
          for i in shape]
    out[coords] = random.uniform(100,150)

    return out


def get_shadow(shape):
    shadow=cv.imread(random.choice(sh_data))
    shadow=cv.resize(shadow,(shape[1],shape[0]))

    # flipping
    if random.uniform(0,1)<0.5:
        shadow=cv.flip(shadow,0)
    if random.uniform(0,1)<0.5:
        shadow=cv.flip(shadow,1)

    return shadow

def blur(img):
    if random.uniform(0,1)>=0.4:return img
    f=random.choice([3,5,7])
    return cv.GaussianBlur(img,(f,f),0)

def add_img(img,ref):
    beta=random.uniform(0,.5)
    return cv.addWeighted(img,1.,ref,beta,0)

def play_img(img):
    if random.uniform(0,1)<0.5:return img
    shadow=get_shadow(img.shape)
    img=add_img(img,shadow)

    salt=get_salt(img.shape)
    img=add_img(img,salt)

    img=blur(img)
    return img


names = pd.read_csv("data_generation/names.txt")
name_numpy = names.to_numpy()

fontpath = "data_generation/arialBlack.ttf"
font = ImageFont.truetype(fontpath, 24)
b, g, r, a = 0, 0, 0, 0
count = 0
fileNameList = []
wordList = []

#image_numpy = name_numpy[:, 0]
name_numpy = name_numpy[:, 0]
# print(name_numpy)

idNumbers = '٠١٢٣٤٥٦٧٨٩'
reshaped_text = arabic_reshaper.reshape(name_numpy[count])
reshaped_text = get_display(reshaped_text, base_dir='R')

fileNameList=[]
wordList=[]
j=0

for i in range(100):
    labelString = ''
    texts=[]

    img_pil=Image.fromarray(emptyId)
    draw = ImageDraw.Draw(img_pil)
    for nametype in yDim:
        if nametype ==  'fName':
            labelString = random.choice(name_numpy)
            reshaped_text = arabic_reshaper.reshape(labelString)
            reshaped_text = get_display(reshaped_text, base_dir='R')
            draw.text(fname_pos,  reshaped_text,font=font, fill=(b, g, r, a))
            texts.append(labelString)

        elif nametype ==  'lName':
            labelString = (random.choice(name_numpy)+ ' ' + random.choice(name_numpy) + ' ' + random.choice(name_numpy))
            reshaped_text = arabic_reshaper.reshape(labelString)
            reshaped_text = get_display(reshaped_text, base_dir='L')
            draw.text(lname_pos,  reshaped_text, anchor="la",font=font, fill=(b, g, r, a))
            texts.append(labelString)

        elif nametype == 'idNumber':
            labelString = ''.join(random.choice(idNumbers) for i in range(14))
            reshaped_text = arabic_reshaper.reshape(labelString )
            reshaped_text = get_display(reshaped_text, base_dir='L')
            draw.text(nn_pos,  reshaped_text, anchor="ma",font=font, fill=(b, g, r, a))
            texts.append(labelString)

        elif nametype=='fAddress':
            labelString = random.choice(name_numpy)
            reshaped_text = arabic_reshaper.reshape(labelString)
            reshaped_text = get_display(reshaped_text, base_dir='L')
            draw.text(fadd_pos,  reshaped_text, font=font, fill=(b, g, r, a))
            texts.append(labelString)

        elif nametype=='lAddress':
            labelString = random.choice(name_numpy)+ ' ' + random.choice(name_numpy) + ' ' + random.choice(name_numpy)
            reshaped_text = arabic_reshaper.reshape(labelString)
            reshaped_text = get_display(reshaped_text, base_dir='L')
            draw.text(ladd_pos,  reshaped_text, font=font, fill=(b, g, r, a))
            texts.append(labelString)


    img_pil = np.array(img_pil)
    img=skew(img_pil)
    img_pil=play_img(img_pil)

    margin=5
    for i in range(len(texts)):
        print(j,texts[i])
        p=lengths[i]
        p1,p2=p[1],p[0]
        space=font.getsize(texts[i])
        img=img_pil[p1-margin//2:p1+space[1]+margin//2,p2-margin:p2+space[0]+margin]

        img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        cv.imwrite('data/imgs/'+str(j)+'.png',img)
        fileNameList.append(str(j)+'.png')
        wordList.append(texts[i])
        j+=1

dict = {'filename': fileNameList, 'words': wordList}#name_numpy.tolist()}
df = pd.DataFrame(dict)
df.to_csv('data/labels.csv', index=False)
