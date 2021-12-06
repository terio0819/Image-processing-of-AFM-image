#!/usr/bin/env python
# coding: utf-8



import numpy as np
import mahotas as mh
import cv2


def branchedPoints(skel):
    xbranch0  = np.array([[1,0,1],[0,1,0],[1,0,1]])
    xbranch1 = np.array([[0,1,0],[1,1,1],[0,1,0]])
    tbranch0 = np.array([[0,0,0],[1,1,1],[0,1,0]])
    tbranch1 = np.flipud(tbranch0)
    tbranch2 = tbranch0.T
    tbranch3 = np.fliplr(tbranch2)
    tbranch4 = np.array([[1,0,1],[0,1,0],[1,0,0]])
    tbranch5 = np.flipud(tbranch4)
    tbranch6 = np.fliplr(tbranch4)
    tbranch7 = np.fliplr(tbranch5)  
    ybranch0 = np.array([[1,0,1],[0,1,0],[2,1,2]])
    ybranch1 = np.flipud(ybranch0)
    ybranch2 = ybranch0.T
    ybranch3 = np.fliplr(ybranch2)
    ybranch4 = np.array([[0,1,2],[1,1,2],[2,2,1]])
    ybranch5 = np.flipud(ybranch4)
    ybranch6 = np.fliplr(ybranch4)
    ybranch7 = np.fliplr(ybranch5)
    xbr1=mh.morph.hitmiss(skel, xbranch0); xbr2=mh.morph.hitmiss(skel, xbranch1)
    tbr1=mh.morph.hitmiss(skel, tbranch0); tbr2=mh.morph.hitmiss(skel, tbranch1)
    tbr3=mh.morph.hitmiss(skel, tbranch2); tbr4=mh.morph.hitmiss(skel, tbranch3)
    tbr5=mh.morph.hitmiss(skel, tbranch4); tbr6=mh.morph.hitmiss(skel, tbranch5)
    tbr7=mh.morph.hitmiss(skel, tbranch6); tbr8=mh.morph.hitmiss(skel, tbranch7)
    ybr1=mh.morph.hitmiss(skel, ybranch0); ybr2=mh.morph.hitmiss(skel, ybranch1)
    ybr3=mh.morph.hitmiss(skel, ybranch2); ybr4=mh.morph.hitmiss(skel, ybranch3)
    ybr5=mh.morph.hitmiss(skel, ybranch4); ybr6=mh.morph.hitmiss(skel, ybranch5)
    ybr7=mh.morph.hitmiss(skel, ybranch6); ybr8=mh.morph.hitmiss(skel, ybranch7)
    return xbr1+xbr2+tbr1+tbr2+tbr3+tbr4+tbr5+tbr6+tbr7+tbr8+ybr1+ybr2+ybr3+ybr4+ybr5+ybr6+ybr7+ybr8
    
    
def endPoints(skel):
    endpoint1=np.array([[0, 0, 0],[0, 1, 0],[2, 1, 2]])
    endpoint2=np.array([[0, 0, 0],[0, 1, 2],[0, 2, 1]])
    endpoint3=np.array([[0, 0, 2],[0, 1, 1],[0, 0, 2]])
    endpoint4=np.array([[0, 2, 1],[0, 1, 2],[0, 0, 0]])
    endpoint5=np.array([[2, 1, 2],[0, 1, 0],[0, 0, 0]])
    endpoint6=np.array([[1, 2, 0],[2, 1, 0],[0, 0, 0]])
    endpoint7=np.array([[2, 0, 0],[1, 1, 0],[2, 0, 0]])
    endpoint8=np.array([[0, 0, 0],[2, 1, 0],[1, 2, 0]])
    ep1=mh.morph.hitmiss(skel,endpoint1); ep2=mh.morph.hitmiss(skel,endpoint2)
    ep3=mh.morph.hitmiss(skel,endpoint3); ep4=mh.morph.hitmiss(skel,endpoint4)
    ep5=mh.morph.hitmiss(skel,endpoint5); ep6=mh.morph.hitmiss(skel,endpoint6)
    ep7=mh.morph.hitmiss(skel,endpoint7); ep8=mh.morph.hitmiss(skel,endpoint8)
    return ep1+ep2+ep3+ep4+ep5+ep6+ep7+ep8

def tracking(img, ep_x, ep_y):
    imgcopy = img.copy()
    imgcopy = thin(imgcopy)
    imgcopy = skeletonize(imgcopy)
    xtrack = [ep_x[0]]
    ytrack = [ep_y[0]]
    x = ep_x[0]; y = ep_y[0]
    for i in range(np.sum(imgcopy)):
        imgcopy[x,y] = 0 #現在の注目画素の値を0に更新
        #移動方向の探索
        window = imgcopy[x-1:x+2,y-1:y+2]
        direction = np.where(window != 0)
        direction = [a-1 for a in direction]
        x += int(direction[0]); y += int(direction[1])
        xtrack.append(x)
        ytrack.append(y)
        if x == ep_x[1] and y == ep_y[1]:
            break
    xtrack = np.asarray(xtrack); ytrack = np.asarray(ytrack)
    return xtrack, ytrack


def remove_bp(img, remove_size=1, min_area = 10):     #10/21 min_areaを追加
    imgcopy = img.copy()
    bp = branchedPoints(imgcopy)
    bp_coor = np.where(bp)
    for bp_x, bp_y in zip(bp_coor[0], bp_coor[1]):  #bpの周囲を除去
        imgcopy[bp_x - remove_size:bp_x + remove_size+1, bp_y - remove_size:bp_y + remove_size+1] = 0
    
    if min_area != 0:   #10/21追加
        tmp_nlabels, tmp_label_image = cv2.connectedComponents(np.uint8(imgcopy))
        for i in range(tmp_nlabels):
            size = np.sum(tmp_label_image==i)
            if size < min_area:
                imgcopy[np.where(tmp_label_image==i)] = 0    
    return imgcopy
 

def tracking2(img):    #端点の座標の入力無しでも行けるバージョン（未テスト）(画面端で見切れてるCNFは不可:これを直す必要あり？)
    imgcopy = img.copy()
    
    ep = endPoints(imgcopy)
    ep_c = np.where(ep)
    
    xtrack = [ep_c[0][0]]
    ytrack = [ep_c[1][0]]

    x = ep_c[0][0]; y = ep_c[1][0]
    for i in range(np.sum(imgcopy)):
        imgcopy[x,y] = 0 #現在の注目画素の値を0に更新
        #移動方向の探索
        window = imgcopy[x-1:x+2,y-1:y+2]
        direction = np.where(window != 0)
        direction = [a-1 for a in direction]
        x += direction[0][0]; y += direction[1][0]
        xtrack.append(x)
        ytrack.append(y)
        if x == ep_c[0][1] and y == ep_c[1][1]:
            break
    xtrack = np.asarray(xtrack); ytrack = np.asarray(ytrack)
    return xtrack, ytrack
    


def get_length(skel_img, pixel_size=2000/1024):
    mask1 = np.eye(2).astype(np.uint8)
    mask2 = np.flip(mask1,axis=0)
    row, column = skel_img.shape
    diag = np.zeros((row-1, column-1)).astype(np.uint8)
    for i in range(row-1):
        for j in range(column-1):
            if  np.allclose(mask1, skel_img[i:i+2, j:j+2]) or np.allclose(mask2, skel_img[i:i+2, j:j+2]):
                diag[i, j] = 1

    length = (np.sum(skel_img) + np.sum(diag)*(np.sqrt(2)-1)) * pixel_size
    return length