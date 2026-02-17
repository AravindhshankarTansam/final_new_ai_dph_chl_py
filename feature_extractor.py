import cv2
import numpy as np

# ===============================
# BLUR CHECK
# ===============================
def is_blurry(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Real world mobile thresholds
    if blur_score < 15:
        return "very_blurry", blur_score
    elif blur_score < 35:
        return "slightly_blurry", blur_score
    else:
        return "sharp", blur_score


# ===============================
# WHITE BALANCE
# ===============================
def white_balance(img):
    result = img.astype(np.float32)
    avg_b, avg_g, avg_r = np.mean(result[:,:,0]),np.mean(result[:,:,1]),np.mean(result[:,:,2])
    avg = (avg_b+avg_g+avg_r)/3
    result[:,:,0] *= avg/avg_b
    result[:,:,1] *= avg/avg_g
    result[:,:,2] *= avg/avg_r
    return np.clip(result,0,255).astype(np.uint8)

# ===============================
# LIGHT NORMALIZATION
# ===============================
def normalize_lighting(img):
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(3.0,(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l,a,b))
    return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)

# ===============================
# SKIN REMOVAL
# ===============================
def remove_skin(img):
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    lower = np.array([0,30,60])
    upper = np.array([20,150,255])
    mask = cv2.inRange(hsv,lower,upper)
    img[mask>0] = 0
    return img

# ===============================
# LIQUID DETECTION
# ===============================
def extract_liquid(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(hsv,(10,40,40),(45,255,255))
    red1 = cv2.inRange(hsv,(0,70,50),(10,255,255))
    red2 = cv2.inRange(hsv,(160,70,50),(180,255,255))

    mask = yellow | red1 | red2

    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    h_img,w_img = img.shape[:2]

    # 🧠 fallback center crop if no contour
    if len(contours)==0:
        return img[int(h_img*0.3):int(h_img*0.7),
                   int(w_img*0.3):int(w_img*0.7)]

    cnt=max(contours,key=cv2.contourArea)

    if cv2.contourArea(cnt) < 500:
        return img[int(h_img*0.3):int(h_img*0.7),
                   int(w_img*0.3):int(w_img*0.7)]

    x,y,w,h=cv2.boundingRect(cnt)

    liquid=img[y:y+h,x:x+w]

    if liquid.size == 0:
        return img[int(h_img*0.3):int(h_img*0.7),
                   int(w_img*0.3):int(w_img*0.7)]

    ch,cw=liquid.shape[:2]

    liquid=liquid[int(ch*0.2):int(ch*0.8),
                  int(cw*0.2):int(cw*0.8)]

    if liquid.size == 0:
        return img[int(h_img*0.3):int(h_img*0.7),
                   int(w_img*0.3):int(w_img*0.7)]

    return liquid


# ===============================
# FEATURE EXTRACTION
# ===============================
def extract_features(path):

    img = cv2.imread(path)

    if img is None:
        print("⚠️ cannot load image:", path)
        return None

    # preprocessing
    img = white_balance(img)
    img = normalize_lighting(img)
    img = remove_skin(img)

    liquid = extract_liquid(img)

    # ✅ CRITICAL SAFETY CHECK
    if liquid is None or liquid.size == 0:
        print("⚠️ empty liquid region:", path)
        return None

    # another safety resize (prevents tiny crops)
    if liquid.shape[0] < 10 or liquid.shape[1] < 10:
        print("⚠️ too small liquid:", path)
        return None

    lab = cv2.cvtColor(liquid, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(liquid, cv2.COLOR_BGR2HSV)

    lab_mean = np.mean(lab.reshape(-1,3), axis=0)
    hsv_mean = np.mean(hsv.reshape(-1,3), axis=0)

    std = np.std(liquid)

    area_ratio = (liquid.shape[0]*liquid.shape[1])/(img.shape[0]*img.shape[1])

    return [
        float(lab_mean[0]),
        float(lab_mean[1]),
        float(lab_mean[2]),
        float(hsv_mean[0]),
        float(hsv_mean[1]),
        float(std),
        float(area_ratio)
    ]
