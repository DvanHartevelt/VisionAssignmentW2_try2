import numpy as np
import cv2
import math
from util import ProgressBar as pb


def gray_convolute3x3(img, kernel):
    """
    Convoluting a kernel across an image.

    :param img: grayscaled image
    :param kernel: np.array kernel
    :return: new grayscaled image
    """
    # exception testing
    if len(img.shape) != 2:
        print("Image has to be grayscale.")
        return img

    width = img.shape[1]
    height = img.shape[0]

    imgNew = np.zeros_like(img)

    # A sweep over all the pixels in img
    weightsum = kernel.sum()
    if weightsum == 0:
        #determining weight factors
        hi = 0
        lo = 0
        for i in range(3):
            for j in range(3):
                if kernel[i,j] < 0:
                    lo += kernel[i,j]
                else:
                    hi += kernel[i,j]

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # using the dotproduct of our kernel and a slice of img
                newValue = (kernel * img[y - 1: y + 2, x - 1: x + 2]).sum()

                newValue = np.interp(newValue, [lo * 255, hi * 255], [0, 255])

                imgNew[y, x] = newValue

            pb.printProgressBar(y, height - 2, prefix=f'Convoluting...:', length=50)
    else:
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                # using the dotproduct of our kernel and a slice of img
                newValue = (kernel * img[y - 1: y + 2, x - 1: x + 2]).sum()

                newValue = newValue / weightsum

                if newValue < 0:
                    newValue = 0
                if newValue > 255:
                    newValue = 255

                imgNew[y, x] = int(newValue)

            pb.printProgressBar(y, height - 2, prefix=f'Convoluting...:', length=50)

    return imgNew

def main():
    #import original pictures
    imgCat = cv2.imread("Resources/smollCat.jpeg", 0)
    imgCat = cv2.resize(imgCat, (int(imgCat.shape[1]/3), int(imgCat.shape[0]/3)))

    imgMoon = cv2.imread("Resources/Moon.jpeg", 0)
    imgMoon = cv2.resize(imgMoon, (int(imgMoon.shape[1]/2), int(imgMoon.shape[0]/2)))

    imgBuild = cv2.imread("Resources/buildings.png", 0)
    imgBuild = cv2.resize(imgBuild, (int(imgBuild.shape[1]/2), int(imgBuild.shape[0]/2)))

    imgPony = cv2.imread("Resources/pony.jpeg", 0)
    imgPony = cv2.resize(imgPony, (int(imgPony.shape[1]/4), int(imgPony.shape[0]/4)))


    """
    #Blurring
    """
    kernelBoxBlur =         np.array([[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]])
    kernelGaussianLowPass = np.array([[1, 2, 1],
                                      [2, 4, 2],
                                      [1, 2, 2]])

    imgCatBoxBlurred = gray_convolute3x3(imgCat, kernelBoxBlur)
    imgCatGausianLowPass = gray_convolute3x3(imgCat, kernelGaussianLowPass)

    cv2.imwrite("Output/CatBoxBlurred.png", imgCatBoxBlurred)
    cv2.imwrite("Output/CatGausianLowPass.png", imgCatGausianLowPass)

    cv2.imshow('Original', imgCat)
    cv2.imshow('Boxblurred', imgCatBoxBlurred)
    cv2.imshow('Gausian Low Pass', imgCatGausianLowPass)

    cv2.waitKey(0)

    

    """
    #Laplacian sharpening
    """

    kernelLaplacian4 = np.array([[ 0, -1,  0],
                                 [-1,  4, -1],
                                 [ 0, -1,  0]])

    kernelCompositeLaplacian = np.array([[ 0, -1,  0],
                                         [-1,  5, -1],
                                         [ 0, -1,  0]])

    imgMoonLap = gray_convolute3x3(imgMoon, kernelLaplacian4)
    imgMoonSharp = gray_convolute3x3(imgMoon, kernelCompositeLaplacian)

    cv2.imwrite("Output/MoonLaplacian.png", imgMoonLap)
    cv2.imwrite("Output/MoonSharpened.png", imgMoonSharp)

    cv2.imshow('Original', imgMoon)
    cv2.imshow('Laplacian', imgMoonLap)
    cv2.imshow('Laplacianly sharpened', imgMoonSharp)

    cv2.waitKey(0)
    
    

    """
    #Directional edge detection filters
    """

    kernelNorth = np.array([[ 1,  1,  1],
                            [ 1, -2,  1],
                            [-1, -1, -1]])

    kernelEast = np.array([[-1,  1,  1],
                           [-1, -2,  1],
                           [-1,  1,  1]])

    kernelSouth = np.array([[-1, -1, -1],
                            [ 1, -2,  1],
                            [ 1,  1,  1]])

    imgBuildTop = gray_convolute3x3(imgBuild, kernelNorth)
    imgBuildSide = gray_convolute3x3(imgBuild, kernelEast)
    imgBuildBottom = gray_convolute3x3(imgBuild, kernelSouth)

    cv2.imwrite("Output/BuildingNorth.png", imgBuildTop)
    cv2.imwrite("Output/BuildingEast.png", imgBuildSide)
    cv2.imwrite("Output/BuildingSouth.png", imgBuildBottom)

    cv2.imshow('Original', imgBuild)
    cv2.imshow('North edge detection', imgBuildTop)
    cv2.imshow('East edge detection', imgBuildSide)
    cv2.imshow('South edge detection', imgBuildBottom)

    cv2.waitKey(0)
    
    

    """
    #Prewitt en Sobel
    """
    def Prewitt(img):
        PWVert = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]])

        PWHori = np.array([[1,  1,   1],
                           [0,  0,   0],
                           [-1, -1, -1]])

        imgNew = np.zeros_like(img)

        width = img.shape[1]
        height = img.shape[0]

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                vert = (PWVert * img[y - 1: y + 2, x - 1: x + 2]).sum()
                hori = (PWHori * img[y - 1: y + 2, x - 1: x + 2]).sum()

                newValue = math.sqrt(vert**2 + hori**2)

                if newValue > 255:
                    newValue = 255

                imgNew[y, x] = newValue

            pb.printProgressBar(y, height - 2, prefix=f'Applying Prewitt filter...:', length=50)

        return imgNew

    def Sobel(img):
        SBVert = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

        SBHori = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])

        imgNew = np.zeros_like(img)

        width = img.shape[1]
        height = img.shape[0]

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                vert = (SBVert * img[y - 1: y + 2, x - 1: x + 2]).sum()
                hori = (SBHori * img[y - 1: y + 2, x - 1: x + 2]).sum()

                newValue = math.sqrt(vert ** 2 + hori ** 2)

                if newValue > 255:
                    newValue = 255

                imgNew[y, x] = newValue

            pb.printProgressBar(y, height - 2, prefix=f'Applying Sobel filter...:', length=50)

        return imgNew


    imgCatPrewitt = Prewitt(imgCat)
    imgCatSobel = Sobel(imgCat)

    cv2.imwrite("Output/CatPrewitt.png", imgCatPrewitt)
    cv2.imwrite("Output/CatSobel.png", imgCatSobel)

    cv2.imshow('Original', imgCat)
    cv2.imshow('Prewitt', imgCatPrewitt)
    cv2.imshow('Sobel', imgCatSobel)
    
    
    imgPonyPrewitt = Prewitt(imgPony)
    imgPonySobel = Sobel(imgPony)

    cv2.imwrite("Output/PonyPrewitt.png", imgPonyPrewitt)
    cv2.imwrite("Output/PonySobel.png", imgPonySobel)

    cv2.imshow('Original', imgPony)
    cv2.imshow('Prewitt', imgPonyPrewitt)
    cv2.imshow('Sobel', imgPonySobel)
    
    cv2.waitKey(0)

    
    """
    #minmax operator
    """
    def MaxMin(img):
        imgNew = np.zeros_like(img)

        width = img.shape[1]
        height = img.shape[0]

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                max = img[y - 1: y + 2, x - 1: x + 2].max()
                min = img[y - 1: y + 2, x - 1: x + 2].min()

                newValue = max - min

                if newValue < 0:
                    newValue = 0
                if newValue > 255:
                    newValue = 255

                imgNew[y, x] = newValue

            pb.printProgressBar(y, height - 2, prefix=f'Applying MaxMin filter...:', length=50)

        return imgNew

    imgPonyMaxMin = MaxMin(imgPony)

    cv2.imwrite("Output/PonyMaxMin.png", imgPonyMaxMin)

    cv2.imshow('Original', imgPony)
    cv2.imshow('MaxMin', imgPonyMaxMin)

    cv2.waitKey(0)




    """
    #Matched filter
    """

    def MatchedFilter(img, filter):
        imgNew = np.zeros_like(img)

        width = img.shape[1]
        height = img.shape[0]

        fwidth = filter.shape[1]
        fheight = filter.shape[0]

        # A sweep over all the pixels in img
        weightsum = filter.sum()
        if weightsum == 0:
            # determining weight factors
            hi = 0
            lo = 0
            for i in range(fheight):
                for j in range(fwidth):
                    if filter[i, j] < 0:
                        lo += filter[i, j]
                    else:
                        hi += filter[i, j]

            for y in range(height - fheight):
                for x in range(width - fwidth):
                    # using the dotproduct of our filter and a slice of img
                    newValue = (filter * img[y: y + fheight, x : x + fwidth]).sum()

                    newValue = np.interp(newValue, [lo * 255, hi * 255], [0, 255])

                    imgNew[y, x] = newValue

                pb.printProgressBar(y, height - 2, prefix=f'Applying matched filter...:', length=50)
        else:
            for y in range(height - fheight):
                for x in range(width - fwidth):
                    # using the dotproduct of our kernel and a slice of img
                    newValue = (filter * img[y : y + fheight, x : x + fwidth]).sum()

                    newValue = newValue / weightsum

                    if newValue < 0:
                        newValue = 0
                    if newValue > 255:
                        newValue = 255

                    imgNew[y, x] = int(newValue)

                pb.printProgressBar(y, height - fheight -1, prefix=f'Applying matched filter...:', length=50)

        return imgNew

    specialFilter = np.array([[ 0,  0,  0,  0,  0,  0,  0,  0],
                              [ 1,  1,  1,  1,  1,  1,  1,  1],
                              [-1, -1, -1, -1, -1, -1, -1, -1]])

    CompSpeFilter = np.array([[ 1,  0,  0,  0,  0,  0,  0,  0],
                              [ 1,  1,  1,  1,  1,  1,  1,  1],
                              [-1, -1, -1, -1, -1, -1, -1, -1]])

    imgMatched = MatchedFilter(imgBuild, specialFilter)
    imgCompSpe = MatchedFilter(imgBuild, CompSpeFilter)

    cv2.imwrite("Output/BuildingMatched.png", imgMatched)
    cv2.imwrite("Output/BuildingMatchedComposite.png", imgCompSpe)

    cv2.imshow('Original', imgBuild)
    cv2.imshow('Matched filter', imgMatched)
    cv2.imshow('Composite Matched filter', imgCompSpe)

    cv2.waitKey(0)

    print("Thank you for running all filters.")

if __name__ == '__main__':
    main()
