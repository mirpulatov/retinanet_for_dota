import cv2
import numpy as np
import glob

i = 0
STRIDE = 900
SIDE = 1000

with open("annotation.txt", "w") as f:
    for ano in glob.glob("labels/*.txt"):
        print(ano)
        fn = "images"+ano[6:-4]+".png"
        img = cv2.imread(fn)
        imgH, imgW, _ = img.shape
        for sx in range(0, imgW, STRIDE):
            for sy in range(0, imgH, STRIDE):
                fncrop = "crops/"+fn[7:-4]+"-"+str(sx)+"-"+str(sy)+".png"
                crop = np.zeros((SIDE, SIDE, 3), np.uint8)
                crop1 = img[sy:sy+SIDE, sx:sx+SIDE]
                crop[0:crop1.shape[0], 0:crop1.shape[1]] = crop1
                cv2.imwrite(fncrop, crop)
                for line in open(ano):
                    vals = line.split(' ')
                    if len(vals) == 10:
                        coords = list(map(int, vals[:-2]))
                        x1 = min(coords[0], coords[2], coords[4], coords[6])
                        x2 = max(coords[0], coords[2], coords[4], coords[6])
                        y1 = min(coords[1], coords[3], coords[5], coords[7])
                        y2 = max(coords[1], coords[3], coords[5], coords[7])
                        if x1 > sx and x2 < sx + SIDE and y1 > sy and y2 < sy + SIDE:
                            w = x2 - x1
                            h = y2 - y1
                            if w >= 10 and h >= 10:
                                i += 1
                                f.write(fncrop + "," + str(x1-sx) + "," + str(y1-sy) + "," + str(x2-sx) + "," +
                                        str(y2-sy) + "," + vals[8] + "\n")
                            else:
                                print(w, h, vals)
