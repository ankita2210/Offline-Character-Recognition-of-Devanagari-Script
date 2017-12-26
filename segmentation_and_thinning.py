import matplotlib
import matplotlib.pyplot as plt
import skimage.io as io
#from skimage.feature import corner_harris, corner_peaks
import numpy as np
from PIL import Image
import matplotlib.image as mplimg
import scipy.misc as s
import pandas as pd
import os 

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
"load image data"
Img_Original =  rgb2gray(io.imread( 'w1.jpg') )     # Gray image, rgb images need pre-conversion

"Convert gray images to binary images using Otsu's method"
from skimage.filters import threshold_otsu
Otsu_Threshold = threshold_otsu(Img_Original)   
BW_Original = Img_Original > Otsu_Threshold    # must set object region as 1, background region as 0 !  < : white background black image 

def neighbours(x,y,image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],     # P2,P3,P4,P5
                img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]    # P6,P7,P8,P9

def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]      # P2, P3, ... , P8, P9, P2
    return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)

def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1        #  the points to be removed (set as 0)
    while changing1 or changing2:   #  iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    2 <= sum(n) <= 6   and    # Condition 1: 2<= N(P1) <= 6
                    transitions(n) == 1 and    # Condition 2: S(P1)=1  
                    P2 * P4 * P6 == 0  and    # Condition 3   
                    P4 * P6 * P8 == 0):         # Condition 4
                    changing1.append((x,y))
        for x, y in changing1: 
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1   and        # Condition 0
                    2 <= sum(n) <= 6  and       # Condition 1
                    transitions(n) == 1 and      # Condition 2
                    P2 * P4 * P8 == 0 and       # Condition 3
                    P2 * P6 * P8 == 0):            # Condition 4
                    changing2.append((x,y))    
        for x, y in changing2: 
            Image_Thinned[x][y] = 0
    return Image_Thinned
 
def improved_thinning(image):
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = 1        #  the points to be removed (set as 0)
    while changing1:
        changing1=0
        rows, columns = Image_Thinned.shape               # x for rows, y for columns
        for x in range(1, rows - 1):                     # No. of  rows
            for y in range(1, columns - 1):            # No. of columns
                P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1     and    # Condition 0: Point P1 in the object regions 
                    ((P2*P4==1 and P7==0)   or    # Condition 1: 2<= N(P1) <= 6
                    (P2*P8==1 and P5==0)   or   # Condition 2: S(P1)=1  
                    (P6*P4==1 and P9==0)   or   # Condition 3   
                    (P6*P8==1 and P3==0))):
                    Image_Thinned[x][y]=0
                    changing1=1
    return Image_Thinned

def segmentation(image,sr,er):
    Image_Thinned = image.copy()
    rows, columns = Image_Thinned.shape
    '''ff=er=sr=0
    for i in range(1, rows-1):
        for j in range(1, columns-1):
            if(Image_Thinned[i][j]==1):
                sr=i
                ff=1
                break
        if ff==1:
            break
    ff=0
    for i in range(sr, rows-1):
        cnt=0
        for j in range(1, columns-1):
            if Image_Thinned[i][j]==1:
                cnt = cnt+1
        if cnt==0:
            er = i
            break
        if ff==1:
            break'''
    BW_TH=[]
    new=[]
    for j in range(columns):
        new.append(0)
    for i in range(11):
        BW_TH.append(new)
    for i in range(sr, er):
        new=[]
        for j in range(columns):
            new.append(Image_Thinned[i][j])
        BW_TH.append(new)
    new =[]
    for j in range(columns):
        new.append(0)
    for i in range(11):
        BW_TH.append(new)
    #print(BW_TH)
    fig, ax3 = plt.subplots(1, 1)
    ax3.imshow(BW_TH, cmap=plt.cm.gray)
    ax3.axis('off')
    plt.show()
    rows= len(BW_TH)
    columns=len(BW_TH[0])
    print(rows,' ',len(BW_TH[0]),' ',columns)
    c=r=f=0
    for i in range(0, columns):
        for j in range(0, rows):
            if BW_TH[j][i]==1:
                c=i
                f=1
                break
            if f==1:
                break

    l = []
    free=b=k=x=0
    p_c=n_c=c   # previous and next column 
    for i in range(c, columns):
        count=0
        for j in range(0, rows):
            if BW_TH[j][i]==1:
                count = count + 1
        if(b==1 and count>1):
            b=0
            p_c = i
            free=0
            x=0
        if(free==1 and count ==0):
            b=1
            if x==0:
                a=[0,1]
                l.append(a)
                x=1
        if(free==1 and count > 1):
            p_c = i
            free=0
            k=1
        if(free==0 and (count ==1 or (k==1 and count==0))):
            n_c = i
            a =[p_c, n_c]
            l.append(a)
            p_c = n_c
            free = 1
            k=0

    n = len(l)
    print(n)
    print(l)
    for i in range(0,n):
        col = l[i][1]-l[i][0]
        if col==0:
            continue
        if col==1:
            empty_csv()
            continue
        BW_New =[]
        for k in range(rows):
            new =[]
            for j in range(10):
                new.append(0)
            for j in range(l[i][0],l[i][1]):
                new.append(BW_TH[k][j])
            for j in range(10):
                new.append(0)
            BW_New.append(new)
            
        f=w=pr=pc=yes=0
        for t in range(len(BW_New[0])):
            for y in range(len(BW_New)):
                if(BW_New[y][t]==0 and w==1):
                    pr = y-1
                    break
                if(BW_New[y][t]==1 and f==1):
                    w=1
                if(BW_New[y][t]==1 and f==0):
                    f=1
            if(pr!=0):
                pc=t
                break
        f=w=g=0

        for t in range(pc+1,len(BW_New[0])):
            cnt=0
            for y in range(0,len(BW_New)):
                if(BW_New[y][t]==0 and g==1):
                    pr = y-1
                    break
                if(BW_New[y][t]==1 and f==1):
                    cnt = cnt+1
                    g=1
                if(BW_New[y][t]==1 and f==1 and y<pr):
                    cnt = cnt+1
                    w=1
                    break
                if(BW_New[y][t]==1 and f==0):
                    cnt = cnt+1
                    f=1
            if w==1:
                yes=1
                break
            
        '''f=w=pr=pc=0
        for t in range(len(BW_New[0])-1,0,-1):
            for y in range(len(BW_New)):
                if(BW_New[y][t]==0 and w==1):
                    pr = y-1
                    break
                if(BW_New[y][t]==1 and f==1):
                    w=1
                if(BW_New[y][t]==1 and f==0):
                    f=1
            if(pr!=0):
                pc=t
                break
        f=w=g=0
        print(pr,' ',BW_New[pr][pc])
        for t in range(pc-1,0,-1):
            cnt=0
            for y in range(0,len(BW_New)):
                if(BW_New[y][t]==0 and g==1):
                    pr = y-1
                    break
                if(BW_New[y][t]==1 and f==1):
                    cnt = cnt+1
                    g=1
                if(BW_New[y][t]==1 and f==1 and y<pr):
                    cnt = cnt+1
                    w=1
                    break
                if(BW_New[y][t]==1 and f==0):
                    cnt = cnt+1
                    f=1
            if w==1:
                yes=1
                break'''

        '''if yes==0:
            BW = []
            for k in range(rows):
                new=[]
                for j in range(10):
                    new.append(0)
                for j in range(l[i-1][0],l[i][1]):
                    new.append(BW_TH[k][j])
                for j in range(10):
                    new.append(0)
                BW.append(new)
            fig, ax3 = plt.subplots(1, 1)
            ax3.imshow(BW, cmap=plt.cm.gray)
            ax3.axis('off')
            plt.show()'''

        
        #img_ints = np.rint(BW_New)
        '''img2 = Image.new("L", (len(BW_New),len(BW_New[0])))
        img2.putdata(BW_New)
        img2.show()'''
        print_csv(BW_New)
            
def empty_csv():
    a = np.array((8))
    b = np.array((8))
    value = np.hstack((a,b))  #merges 2 array into one
    df = pd.DataFrame(value).T
    with open('seg.csv', 'a') as dataset: 
        df.to_csv(dataset, header=False, index=False) #inbuilt function
        
def print_csv(BW_New):
    df = pd.DataFrame()
    fig, ax3 = plt.subplots(1, 1)
    ax3.imshow(BW_New, cmap=plt.cm.gray)
    ax3.axis('off')
    plt.show()
    plt.imsave('image.png',BW_New)  #save the 2D matrix as an image(rgb)
    im1 = Image.open('image.png').convert('L') #.convert('L') rgb to grayscale 
    im2 = im1.resize((32,32), Image.BILINEAR) 
    im2.save('li.png')
    im = io.imread('li.png') #opens image as array
    Otsu_Threshold = threshold_otsu(im)   
    img = im > Otsu_Threshold
    BW_S = zhangSuen(img)
    BW_A = improved_thinning(BW_S)
    fig, ax3 = plt.subplots(1, 1)
    ax3.imshow(BW_A, cmap=plt.cm.gray)
    ax3.axis('off')
    plt.show()
    #Otsu_Threshold = threshold_otsu(img)   
    #im = img > Otsu_Threshold
    value = BW_A.flatten() #2D to 1D
    a = np.array((32))
    value = np.hstack((a,value))
    size=np.shape(value) #returns single dim
    '''for j in range(1,size[0]):
        if int(value[j])==1:
            value[j]=255'''
    for j in range(1,size[0]):
        value[j]=int(value[j])*255
        #value[j]=int(int(value[j])/70)
        if int(value[j])>255:
                value[j]=255
    df = df.append((pd.DataFrame(value).T),ignore_index = True)
    with open('seg.csv', 'a') as dataset: 
        df.to_csv(dataset, header=False, index=False)
    new =  io.imread( 'li.png')
    fig, ax3 = plt.subplots(1, 1)
    ax3.imshow(new, cmap=plt.cm.gray)
    ax3.axis('off')
    plt.show()
    
def p_segmentation(image):        #segmenting para in lines
    Image_Thinned = image.copy()                               # 0 means background 1 means written image
    rows, columns = Image_Thinned.shape  # returns no of rows and columns
    r=b=0
    while r<rows:
        ff=er=sr=0
        for i in range(r, rows):
            for j in range(1, columns):
                if(Image_Thinned[i][j]==1):
                    sr=i
                    ff=1
                    break
            if ff==1:
                break
            if i==rows-1:    #avoids scanning of whole image twice
                b=1
        if b==1:
            break
        ff=0
        for i in range(sr, rows-1):
            cnt=0
            for j in range(1, columns-1):
                if Image_Thinned[i][j]==1:
                    cnt = cnt+1
            if cnt==0:
                er = i
                break
        segmentation(Image_Thinned,sr,er)
        r =er
    
    
def save():
    with open('test_seg.csv', 'a') as dataset: 
        df.to_csv(dataset, header=False, index=False)
        
"Apply the algorithm on images"
BW_Skeleton = zhangSuen(BW_Original)
BW_Adv = improved_thinning(BW_Skeleton)
BW_Adv1 = p_segmentation(BW_Adv)
#BW_Seg = segmentation(BW_Adv)
#save()
"Display the results"
"""fig, ax3 = plt.subplots(1, 1)
ax3.imshow(BW_Skeleton, cmap=plt.cm.gray)
ax3.set_title('Original binary image')
ax3.axis('off')
plt.show()"""
fig, ax = plt.subplots(1, 3)
ax1, ax2, ax3 = ax.ravel()
ax1.imshow(BW_Original, cmap=plt.cm.gray)
ax1.set_title('Original binary image')
ax1.axis('off')
ax2.imshow(BW_Adv, cmap=plt.cm.gray)
ax2.set_title('Skeleton of the image')
ax2.axis('off')
ax3.imshow(BW_Skeleton, cmap=plt.cm.gray)
ax3.set_title('Original binary image')
ax3.axis('off')
plt.show()
