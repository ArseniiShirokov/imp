## Methods of image processing
### Task 1. Basics of working with images
The photographer, taking landscapes, took a lot of pictures. However, some of the images were rotated by an angle multiple of 90 degrees. Help the photographer solve the problem automatically.

The program must support running from the command line with a strictly defined command format:

```python basic.py (command) (parameters...) (input_file) (output_file)```

List of commands:
* ```mirror {h|v|d|cd}```                               Reflection relative to the horizontal axis (h), vertical axis (v), main diagonal (d), side diagonal (cd)
* ```extract (left_x) (top_y) (width) (height)```       Extract an image fragment with parameters: left indent (left_x, integer), top indent (top_y, integer), fragment width (width, positive), fragment height (height, positive)
* ```rotate {cw|ccw} (angle)```                         Clockwise (cw) or counterclockwise (ccw) by a given number of degrees, for example: rotate cw 90
* ```autorotate```                                      Automatic rotation by an angle of 0, 90, 180 or 270 degrees according to the proposed algorithm

### Task 2. Filtering and metrics
You should implement the basic image filtering algorithms and metrics:
* Median image filtering
* Gaussian Filter
* Bilateral filtering
* MSE Metric
* PSNR Metric
* SSIM Metric

The program must support running from the command line with a strictly defined command format:

```python filters.py (command) (parameters...)```

List of commands:
* ```mse (input_file_1) (input_file_2)```
Calculate the value of the MSE metric and output it to the console
* ```psnr (input_file_1) (input_file_2)```
Calculate the value of the PSNR metric and output it to the
* ```ssim console (input_file_1) (input_file_2)```
Calculate the value of the SSIM metric and output it to the
* ```median console (read) (input_file) (output_file)```
Median filtering with a window size (2rad+1) × (2rad+1)
* ```gauss (sigma_d) (input_file) (output_file)```
 	Gaussian filter with the parameter σd
* ```bilateral (sigma_d) (sigma_r) (input_file) (output_file)```
Bilateral filtering with parameters σd and σr
