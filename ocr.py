import numpy as np
import cv2
import requests


def _show(image, title=''):
    cv2.imshow(f'{title}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


class OCR:
    def __init__(self, file_name):
        with np.load(file_name) as data:
            self.train = data['train']
            self.train_labels = data['train_labels']
        self.knn = cv2.ml.KNearest_create()
        self.knn.train(self.train, cv2.ml.ROW_SAMPLE, self.train_labels)

    def check_digit(self, img):
        ret, result, neighbours, dist = self.knn.findNearest(img.reshape(-1, 900).astype(np.float32), k=5)
        return int(result[0][0])

    def check(self, images):
        numbers = ""
        for img in images:
            num = self.check_digit(img)
            if num < 10:
                numbers += str(num)
            else:
                pass
        return int(numbers) if numbers else None


def img_normalize(img: np.ndarray):
    """원본 이미지를 HD 규격(1280x720)의 이미지로 변환
    Args:
        img(np.ndarray): cv2.imread 등으로 읽어온 원본 이미지 파일
    Return:
        img(np.ndarray): 1280x720 으로 필요 없는 부분이 잘리고, 크기가 조정된 이미지
    """
    # 가로가 더 긴 경우
    if (img.shape[1] / img.shape[0]) > (16 / 9):
        # 세로 높이를 고정
        height = img.shape[0]
        width = int(height / 9 * 16)
        w_blk = int((img.shape[1] - width) / 2)
        sliced = img[:, w_blk:-w_blk] if w_blk != 0 else img
    # 세로가 더 긴 경우
    elif (img.shape[1] / img.shape[0]) < (16 / 9):
        # 가로 길이를 고정
        width = img.shape[1]
        height = int(width / 16 * 9)
        h_blk = int(img.shape[0] - height)
        sliced = img[h_blk:]
    else:
        sliced = img
    return cv2.resize(sliced, (1280, 720))


def get_boxes(img: np.ndarray):
    image, contours, hierachy = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = [cv2.boundingRect(n) for n in contours]
    rects = sorted([n for n in rects if n[3] > 20 and n[2] < 16])
    ret = []
    for rect in rects:
        box_img = padding(img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]])
        ret.append(box_img)
    return ret


def get_per_box(img: np.ndarray):
    img = img[690:, 936:1006]
    img = cv2.equalizeHist(img, 2)
    img = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)[1]
    return get_boxes(img)


def get_score_box(img: np.ndarray):
    img = ~img[640:680, 1120:1240]
    img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)[1]
    return get_boxes(img)


def padding(img):
    if img.shape[0] % 2 == 0:
        t = b = int((30 - img.shape[0]) / 2)
    else:
        t = int((30 - img.shape[0]) / 2)
        b = t + 1
    if img.shape[1] % 2 == 0:
        l = r = int((30 - img.shape[1]) / 2)
    else:
        l = int((30 - img.shape[1]) / 2)
        r = l + 1
    img = cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT)
    return img


def get_image_from_url(url):
    raw = requests.get(url)
    if raw.status_code == 200:
        img = cv2.imdecode(np.frombuffer(raw.content, np.uint8), 0)
        return img
    else:
        return


ocr = OCR("digits.npz")
