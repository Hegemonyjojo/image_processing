"""
data:2022/10/31
owner:lrChang
lrcgnn@163.com
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 读取图像
img = cv2.imread("demo-image.jpg")
# plt.imshow(img[:,:,::-1])
# plt.show()


# 1.机械变换
# 1.1 平移
def translation():
    h, w = img.shape[:2]
    # 平移矩阵,y方向向下平移50，x方向向右平移100
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    res = cv2.warpAffine(img, M, (w, h))
    return res

# 1.2 放缩
def resize():
    res = cv2.resize(img, (800, 800))
    return res

# 1.3 旋转
def rotate(angle):
    (h, w) = img.shape[:2]
    (cX, cY) = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    return cv2.warpAffine(img, M, (nW, nH))

# 1.4 转置
def transpose():
    trans_img = cv2.transpose(img)
    res = cv2.flip(trans_img, 0)
    return res

# 1.5 仿射变换
def affine():
    h, w = img.shape[:2]
    pts1 = np.float32([[50, 100], [200, 50], [50, 200]])
    pts2 = np.float32([[100, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    res = cv2.warpAffine(img, M, (w, h))
    return res

# 1.6 镜像
def mirror():
    res = cv2.flip(img, 1)
    return res

# 2.风格变化
# 2.1 玻璃化
def glass():
    res = np.zeros_like(img)
    rows, cols = img.shape[:2]
    # 定义偏移量
    offsets = 50
    # 毛玻璃效果: 像素点邻域内随机像素点的颜色替代当前像素点的颜色
    for y in range(rows - offsets):
        for x in range(cols - offsets):
            random_num = np.random.randint(0, offsets)
            res[y, x] = img[y + random_num, x + random_num]
    return res

# 2.2 浮雕
def embossment():
    height, width = img.shape[:2]
    # 图像灰度处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 创建目标图像
    dstImg = np.zeros((height, width, 1), np.uint8)
    # 浮雕特效算法
    for i in range(0, height):
        for j in range(0, width - 1):
            grayCurrentPixel = int(gray[i, j])
            grayNextPixel = int(gray[i, j + 1])
            newPixel = grayCurrentPixel - grayNextPixel + 150
            if newPixel > 255:
                newPixel = 255
            if newPixel < 0:
                newPixel = 0
            dstImg[i, j] = newPixel
    return dstImg

# 2.3 素描化
def sketch():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波降噪
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny算子
    canny = cv2.Canny(gaussian, 50, 150)
    # 阈值化处理
    _, res = cv2.threshold(canny, 100, 255, cv2.THRESH_BINARY_INV)
    res = np.expand_dims(gray, axis=2)
    return res

# 5.图像扰动
# 5.1 高斯噪声扰动
def clamp(pv):
    if pv > 255:
        return 255
    elif pv < 0:
        return 0
    else:
        return pv

def gaussian_noise(image):
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            image[row, col, 0] = clamp(b+s[0])
            image[row, col, 1] = clamp(g+s[1])
            image[row, col, 2] = clamp(r+s[2])

    return image


# 5.2 椒噪声扰动
def PepperNoise(image):
    # 指定信噪比
    SNR = 0.9
    # 获取总共像素个数
    size = image.size
    # 因为信噪比是 SNR ，所以噪声占据百分之10，所以需要对这百分之10加噪声
    noiseSize = int(size * (1 - SNR))
    # 对这些点加噪声
    for k in range(0, noiseSize):
        # 随机获取 某个点
        xi = int(np.random.uniform(0, image.shape[1]))
        xj = int(np.random.uniform(0, image.shape[0]))
        # 增加噪声
        if image.ndim == 2:
            image[xj, xi] = 0
    return image


# 5.3 盐噪声扰动
def SaltNoise(image):
    # 指定信噪比
    SNR = 0.9
    # 获取总共像素个数
    size = image.size
    # 因为信噪比是 SNR ，所以噪声占据百分之10，所以需要对这百分之10加噪声
    noiseSize = int(size * (1 - SNR))
    # 对这些点加噪声
    for k in range(0, noiseSize):
        # 随机获取 某个点
        xi = int(np.random.uniform(0, image.shape[1]))
        xj = int(np.random.uniform(0, image.shape[0]))
        # 增加噪声
        if image.ndim == 3:
            image[xj, xi] = 255
    return image


# 5.4 泊松噪声
def PoissonNoise(image):
    noise_type = np.random.poisson(lam=0.03, size=image.shape).astype(dtype='uint8')  # lam>=0 值越小，噪声频率就越少，size为图像尺寸
    noise_image = noise_type + image  # 将原图与噪声叠加
    return noise_image


# 5.5 伽马噪声
def GammaNoise(image):
    a, b = 10.0, 2.5
    noiseGamma = np.random.gamma(shape=b, scale=a, size=image.shape)
    imgGammaNoise = image + noiseGamma
    imgGammaNoise = np.uint8(cv2.normalize(imgGammaNoise, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
    return imgGammaNoise


# 5.6 指数噪声
def ExponentNoise(image):
    a = 10.0
    noiseExponent = np.random.exponential(scale=a, size=image.shape)
    imgExponentNoise = image + noiseExponent
    imgExponentNoise = np.uint8(cv2.normalize(imgExponentNoise, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
    return imgExponentNoise


# 5.7 均匀噪声
def UniformNoise(image):
    mean, sigma = 10, 100
    a = 2 * mean - np.sqrt(12 * sigma)  # a = -14.64
    b = 2 * mean + np.sqrt(12 * sigma)  # b = 54.64
    noiseUniform = np.random.uniform(a, b, image.shape)
    imgUniformNoise = image + noiseUniform
    imgUniformNoise = np.uint8(cv2.normalize(imgUniformNoise, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
    return imgUniformNoise


# 5.8 瑞利噪声
def RayleighNoise(image):
    a = 60.0
    noiseRayleigh = np.random.rayleigh(a, size=image.shape)
    imgRayleighNoise = image + noiseRayleigh
    imgRayleighNoise = np.uint8(cv2.normalize(imgRayleighNoise, None, 0, 255, cv2.NORM_MINMAX))  # 归一化为 [0,255]
    return imgRayleighNoise


# 6.图像去噪
# 6.1 非局部均值去噪
def NL_meansNoise(image):
    dst = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return dst


# 6.2 中值滤波
def MedianBlur(image):
    imgMedianBlur = cv2.medianBlur(image, 3)
    return imgMedianBlur


# 6.3 均值滤波
def Blur(image):
    img_blur = cv2.blur(image, (5, 5))
    return img_blur


# 6.4 高斯滤波
def GaussBlur(image):
    imgGaussBlur = cv2.GaussianBlur(image, (5,5), sigmaX=10)
    return imgGaussBlur


# 6.5 联合双边滤波
def JointBiFilter(image):
    imgBiFilter = cv2.bilateralFilter(image, d=5, sigmaColor=100, sigmaSpace=10)
    imgGauss = cv2.GaussianBlur(image, (5, 5), sigmaX=1)  # 高斯滤波用作导向图像
    imgJointBiFilter = cv2.ximgproc.jointBilateralFilter(imgGauss, image, d=5, sigmaColor=10, sigmaSpace=5)
    return imgJointBiFilter



# res = translation()
# res = resize()
# res = rotate(30)
# res = transpose()
# res = affine()
# res = mirror()

# res = glass()
# res = embossment()
# res = sketch()
# res = gaussian_noise(img)
# res = PoissonNoise(img)
res = JointBiFilter(img)

plt.imshow(res[:,:,::-1])
plt.show()