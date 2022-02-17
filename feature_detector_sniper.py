import cv2

img_train = cv2.imread("imagesTrain/sniper_elite.jpg", 0)
cap = cv2.VideoCapture(0)

orb = cv2.ORB_create(nfeatures=500)
bf = cv2.BFMatcher()

kp_train, des_train = orb.detectAndCompute(img_train, None)
imgKp_train = cv2.drawKeypoints(img_train, kp_train, None)

while True:
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_cam, des_cam = orb.detectAndCompute(frame, None)
    imgKp_cam = cv2.drawKeypoints(frame, kp_cam, None)

    matches = bf.knnMatch(des_train, des_cam, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.6 * n.distance:  # only in this condition we consider the matching good
            good.append([m])

    distance = []
    for i in range(len(good)):
        distance.append(good[i][0].distance)

    minMatch = None
    if len(distance) != 0:
        minMatch = [m for m, _ in matches if m.distance == min(distance)]

    if minMatch is not None:
        # print(minMatch[0].queryIdx)
        # print(kp_cam[minMatch[0].queryIdx].pt[0])
        # print("shape %d", frame.shape)
        cv2.circle(frame, (int(kp_cam[minMatch[0].queryIdx].pt[0]), int(kp_cam[minMatch[0].queryIdx].pt[1])),
                   5, (255, 0, 255), 5)

    img_matches = cv2.drawMatchesKnn(img_train, kp_train, frame, kp_cam, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)#, flags=cv2.drawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("img_train", img_train)
    cv2.imshow("img_cam", frame)
    cv2.imshow("matches", img_matches)

    cv2.waitKey(1)
