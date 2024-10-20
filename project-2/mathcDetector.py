import cv2
import csv
import numpy as np
import math


video = cv2.VideoCapture('C:\\Users\\matte\\OneDrive\\Desktop\\project-2\\video.mp4')

with open('statistics.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    field = ["average distance in pixels", "standard distance in pixels", "average distance in radiants", "standard distance in radiants", "homography"]
    writer.writerow(field)  

sift = cv2.SIFT_create()

#creiamo la finestra dove vengono mostrati i frame a confronto con una dimensione ridotta
cv2.namedWindow('Frame comparison', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Frame comparison', 800, 600)


#leggiamo il primo frame
ret, frame1 = video.read()
if not ret:
    print("Failed to read the first frame.")
    video.release()
    cv2.destroyAllWindows()
    exit()

#ridimensioniamo il primo frame per ridurre il computing time
frame1 = cv2.resize(frame1, (800, 600))

#convertiamo l'immagine a una scala di grigi
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

#applichiamo l'algoritmo sift per trovare i descrittori che poi verranno confrontati con i descrittori del frame successivo
keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)

#creiamo un oggetto che confronta la distanza tra un descrittore del primo frame e tutti gli altri del frame successivo
bf = cv2.BFMatcher()

while video.isOpened():
    ret, frame2 = video.read()
    if not ret:
        break

    #ridimensioniamo anche il secondo frame e convertiamo di nuovo a una scala di grigi
    frame2 = cv2.resize(frame2, (800, 600))
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    #troviamo i descrittori del frame corrente e li confrontiamo con quelli del frame precedente usando l'oggetto bf e selezionando i due descrittori più vicini
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    #applichiamo il rapporto tra il nearest neighbour e il second nearest neighbour e selezioniamo solo i valori sotto una certa soglia fissata
    good_matches = []
    for m, n in matches:
        ratio = m.distance / n.distance
        if ratio < 0.75:
            good_matches.append(m)

    #evidenziamo i match
    frame1_matches = cv2.drawMatches(frame1, keypoints1, frame2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    #PRODUCIAMO LE STATISTICHE
    #calcoliamo le distanze di ogni buon match e con queste troviamo la media e std in pixel
    distances = [m.distance for m in good_matches]
    avg_distance_pxl = np.mean(distances)
    std_distance_pxl = np.std(distances)

    #facciamo la stessa cosa ma per trovare valori in radianti
    angles = []
    pts_src = []
    pts_dst = []
    for match in good_matches:
        kp1 = keypoints1[match.queryIdx]
        kp2 = keypoints2[match.trainIdx]
        dx = kp2.pt[0] - kp1.pt[0]
        dy = kp2.pt[1] - kp1.pt[1]
        angle_rad = math.atan2(dy, dx)  
        angles.append(angle_rad)


        #salviamo le coordinate dei punti corrispondenti per l'omografia
        pts_src.append(kp1.pt)
        pts_dst.append(kp2.pt)

    #calcoliamo la media e la  deviazione standard degli angoli
    avg_angle_rad = np.mean(angles) if angles else 0
    std_angle_rad = np.std(angles) if angles else 0


    #calcoliamo la matrice di omografia tra i due frame consecutivi
    if len(pts_src) >= 4 and len(pts_dst) >= 4:
        pts_src_np = np.float32(pts_src).reshape(-1, 1, 2)
        pts_dst_np = np.float32(pts_dst).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(pts_src_np, pts_dst_np, cv2.RANSAC)
    else:
        H = None

    #convertiamo la matrice in stringa per fittarla nel csv
    if H is not None:
        homography_str = np.array2string(H, precision=2, separator=',') 
    else:
        "N/A"

    with open('statistics.csv', 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([avg_distance_pxl, std_distance_pxl, avg_angle_rad, std_angle_rad, homography_str])

    
    cv2.imshow('Frame matches', frame1_matches)

    #copiamo il frame corrente e settiamolo come frame 1, facciamo la stessa cosa per i keypoints e i descrittori
    frame1 = frame2.copy()  
    keypoints1, descriptors1 = keypoints2, descriptors2  


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

