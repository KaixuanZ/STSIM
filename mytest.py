import numpy as np
import cv2

img = cv2.imread('data/iter05.png', 0)
H,W = img.shape


print(H,W)
k=2
#res = np.zeros([H*k,W*k])
res = np.zeros([H*k + 2,W*k + 2])

for i in range(2*k):
#for i in range(4*k):
    for j in range(2*k):
#    for j in range(4*k):
        ii,jj = int(np.random.randint(H//4+1)-H//8 -1), int(np.random.randint(H//4+1)-H//8 -1)
#        src = img[H*3//8+ii:H*5//8+ii,W*3//8+jj:W*5//8+jj]

        #src = img[H*1//4+ii:H*3//4+ii,W*1//4+jj:W*3//4+jj]
        src = img[H*1//4+ii -1:H*3//4+ii +1, W*1//4+jj -1:W*3//4+jj +1]

        src = cv2.flip(src, np.random.randint(2))
        tmp = np.random.randint(4)
        if tmp == 1:
            src = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE)
        elif tmp == 2:
            src = cv2.rotate(src, cv2.ROTATE_180)
        elif tmp == 3:
            src = cv2.rotate(src, cv2.ROTATE_90_COUNTERCLOCKWISE)
#        res[i*H//4:(i+1)*H//4,j*W//4:(j+1)*W//4] = src
        #res[i*H//2:(i+1)*H//2,j*W//2:(j+1)*W//2] = src
        res[i*H//2:(i+1)*H//2 +2,j*W//2:(j+1)*W//2 +2] = res[i*H//2:(i+1)*H//2 +2,j*W//2:(j+1)*W//2 +2] + src
for i in range(1,2*k):
    res[i*H//2:i*H//2 +2] = res[i*H//2:i*H//2 +2]/2
for j in range(1,2*k):
    res[:,j * W // 2:j * W // 2 + 2] = res[:,j * W // 2:j * W // 2 + 2] / 2

import pdb;pdb.set_trace()

#cv2.imwrite('tmp.png',res)
cv2.imwrite('tmp2.png',res[1:-1,1:-1])