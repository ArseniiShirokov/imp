import os
import numpy as np
from skimage.io import imread, imsave
import sys

def rotate(img: np.array, type_: str, angle: int) -> np.array:
    if type_ == 'cw':
        return np.rot90(img, k=angle//90, axes=(1, 0))
    elif type_ == 'ccw':
        return np.rot90(img, k=angle//90, axes=(0, 1))
    else:
        sys.exit(f"Rotate dir {type_} no implement")

def mirror(img: np.array, type_: str) -> np.array:
    if type_ == 'h':
        return np.flip(img, axis=0)
    elif type_ == 'v':
        return np.flip(img, axis=1)
    elif type_ == 'd':
        return mirror(rotate(img, 'ccw', 90), 'h')
    elif type_ == 'cd':
        return mirror(rotate(img, 'cw', 90), 'h') 
    else:
        sys.exit(f"Mirror with param {type_}: No implemented.")
        
def extract(img: np.array, left_x: int, top_y: int, width: int, height: int) -> np.array:
    border_width = max(-left_x, -top_y, 
                       top_y + height - img.shape[0], 
                       left_x + width - img.shape[1], 1)
    padding_img = np.zeros([img.shape[0] + 2 * border_width, 
                            img.shape[1] + 2 * border_width, 3], dtype=img.dtype)
    padding_img[border_width:-border_width, border_width:-border_width] = img
    top_y += border_width
    left_x += border_width
    return padding_img[top_y:top_y+height, left_x:left_x+width]

def get_score(img: np.array) -> float:
    return (0.2989 * img[:,:,0] + 0.5870 * img[:,:,1] + 0.1140 * img[:,:,2]).sum()
    
def autorotate(img: np.array) -> np.array:
    max_score = None
    opt_k = 0
    for k in range(0, 360, 90):
        candidate = rotate(img, 'cw', k)
        height, width, _ = candidate.shape
        top = extract(candidate, 0, 0, width, height // 2)
        bottom = extract(candidate, 0, height // 2, width, height // 2)
        score = get_score(top) - get_score(bottom)
        if not max_score or score > max_score:
            opt_k = k
            max_score = score
    return rotate(img, 'cw', opt_k)

if __name__ == '__main__':
    op = sys.argv[1]
    input_file = sys.argv[-2]
    output_file = sys.argv[-1]  
    input_img = imread(input_file)
    if op == 'autorotate':
        imsave(output_file, autorotate(input_img))
    elif op == 'mirror':
        param = sys.argv[2]
        imsave(output_file, mirror(input_img, param))
    elif op == 'rotate':
        direction, angle = sys.argv[2:-2]
        imsave(output_file, rotate(input_img, direction, int(angle)))
    elif op == 'extract':
        l, t, w, h = list(map(int, sys.argv[2:-2]))
        imsave(output_file, extract(input_img, l, t, w, h))
    else:
        sys.exit("No implement")
