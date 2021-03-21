import numpy as np
from PIL import Image, ImageDraw
import matplotlib
import matplotlib.pyplot as plt
import math
from typing import Tuple, List
import sys


def linear_ker(img: np.ndarray, ker: np.ndarray) -> np.ndarray:
	itImg = np.nditer(img, flags=["multi_index"])
	itKer = np.nditer(ker, flags=["multi_index"])
	border = int(len(ker) / 2)
	imgCpy = np.zeros((len(img),len(img[0])))
	moyKer = 0
	while not itImg.finished:
		i, j = itImg.multi_index
		while not itKer.finished:
			k,l = itKer.multi_index
			if i - border + k >= 0 and j - border + l >= 0 and i - border + k < len(img) and j - border + l < len(img[0]):				
				moyKer += ker[k][l] * img[i - border + k][j - border + l]
			itKer.iternext()
		imgCpy[i][j] = moyKer
		moyKer = 0
		itKer = np.nditer(ker, flags=["multi_index"])
		itImg.iternext()
	return imgCpy


def kernel_average(img: np.ndarray, k: int) -> np.ndarray:
	if not k % 2:
		print("k is not odd")
		return img;
	ker = np.full((k,k), 1/(k*k))
	return linear_ker(img, ker) 

def sobel(img: np.ndarray) -> np.ndarray:
	kerX = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
	kerY = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
	sobelX = linear_ker(img,kerX)
	sobelY = linear_ker(img,kerY)
	itImg = np.nditer(img, flags=["multi_index"])
	while not itImg.finished:
		i, j = itImg.multi_index
		img[i][j] = math.sqrt(sobelX[i][j]**2 + sobelY[i][j]**2)
		itImg.iternext()
	return img

def normalize(img: np.ndarray) -> np.ndarray:
	itImg = np.nditer(img, flags=["multi_index"])
	while not itImg.finished:
		i, j = itImg.multi_index
		if img[i][j] > 150:#valeur de pixel pour rendre noir ou blanc, pas optimal
			img[i][j] = 255
		else:
			img[i][j] = 0
		itImg.iternext()
	return img

def hough_lines(img: np.ndarray, angle_step: int = 1) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
	thetas = np.deg2rad(np.arange(0.0, 360.0, 1))
	width, height = img.shape
	diag_len = int(round(math.sqrt(width * width + height * height)))
	rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
	cos_t = np.cos(thetas)
	sin_t = np.sin(thetas)
	num_thetas = len(thetas)
	accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
	are_edges = img > 0
	y_idxs, x_idxs = np.nonzero(are_edges)
	for i in range(len(x_idxs)):
		x = x_idxs[i]
		y = y_idxs[i]
		for t_idx in range(num_thetas):
			rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
			accumulator[rho, t_idx] += 1
	itAccu = np.nditer(accumulator, flags=["multi_index"])
	listRho = []
	listTheta = []
	while not itAccu.finished:
		i, j = itAccu.multi_index
		if accumulator[i][j] > 200:#valeur a modifier pour detecter plus ou moins de lignes, pas optimal mais pas le temps
			listRho.append(rhos[i])
			listTheta.append(thetas[j])
		itAccu.iternext()
	return (accumulator), (listTheta, listRho)


def show_hough(img: np.ndarray, img_processed: np.ndarray,accumulator: np.ndarray, lines_eq: List[Tuple[int, int]]) -> None:
	fig = plt.figure()
	fig.add_subplot(2,2,1)
	plt.imshow(img)
	fig.add_subplot(2,2,2)
	plt.imshow(img_processed, cmap="gray")
	fig.add_subplot(2,2,3)
	plt.imshow(accumulator,cmap="gray")
	fig.add_subplot(2,2,4)
	thetas = lines_eq[0]
	rhos = lines_eq[1]
	allX = np.arange(len(img_processed[0]))
	img_processed = Image.fromarray(img_processed, 'L')
	draw = ImageDraw.Draw(img_processed)
	for i in range(len(thetas)):
		if(math.sin(thetas[i]) != 0):
			plt.plot(allX,(-(math.cos(thetas[i]) / math.sin(thetas[i]))) * allX + (rhos[i] / math.sin(thetas[i])), "r-")
	plt.imshow(img, cmap = plt.cm.gray)
	plt.show()
	return

def main():
	im_fn = sys.argv[1]	
	img = np.array(Image.open(im_fn).convert("L"))
	img = kernel_average(img, 3)
	img = sobel(img)
	img = normalize(img)
	accum, listTandR = hough_lines(img)
	show_hough(Image.open(sys.argv[1]),img 	, accum, listTandR)

main()
