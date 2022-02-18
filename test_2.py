import cv2
import numpy as np

cap = cv2.VideoCapture(0)
img_train = cv2.imread("imagesTrain/sniper_elite.jpg", 0)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
width, height = 640, 480

objpoints = []
imgpoints = []

objp = np.zeros((1, width * height, 3), np.float32)
objp[0, :, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher()

kp_train, des_train = orb.detectAndCompute(img_train, None)

imgpoints = []

while True:
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kp_query, des_query = orb.detectAndCompute(img, None)
    imgKp_cam = cv2.drawKeypoints(img, kp_query, None)

    matches = bf.knnMatch(des_train, des_query, k=2)

    good = []

    for m, n in matches:
            if m.distance < 0.6 * n.distance:  # only in this condition we consider the matching good
                    good.append([m])

    img_matches = cv2.drawMatchesKnn(img_train, kp_train, img, kp_query, good, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("img_kp", imgKp_cam)
    cv2.imshow("matches", img_matches)

    coordinates = []
    if len(good) != 0:
        for g in good:
            dictionary = [int(kp_query[g[0].queryIdx].pt[0]), int(kp_query[g[0].queryIdx].pt[1])]
            coordinates.append([dictionary])
            #print(coordinates)

    #print("___________________________________________________________________________________________________________")
    coordinates = np.array(coordinates, dtype=np.float32)
    if len(coordinates) != 0:

        print("len coordinates --> ", len(coordinates))

        objp = np.zeros((1, len(coordinates)*len(coordinates), 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:len(coordinates), 0:len(coordinates)].T.reshape(-1, 2)

        objpoints.append(objp)

        print("len objp --> ", len(objp[0]))
        #print("objp", objp)



        coordinates = [coordinates for _ in range(len(coordinates))]#np.vstack(coordinates*len(coordinates))
        #print("coord", coordinates)
        imgpoints.append(coordinates)

    if cv2.waitKey(1) == ord('q'):
            break

    cv2.waitKey(1)

print(imgpoints[0])
print("img point shape: ", len(np.asarray(imgpoints)))
print("obj point shape: ", len(objpoints))
print(objpoints)

while True:
    _, img = cap.read()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints[0], img.shape[::-1], None, None)
    print(ret)

