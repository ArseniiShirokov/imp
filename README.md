## Course on modern methods of image processing
### Task 1. Basics of working with images
The photographer, taking landscapes, took a lot of pictures. However, some of the images were rotated by an angle multiple of 90 degrees. Help the photographer solve the problem automatically.

The program must support running from the command line with a strictly defined command format:

```python basic.py (command) (parameters...) (input_file) (output_file)```

List of commands:
* ```mirror {h|v|d|cd}```                               Reflection relative to the horizontal axis (h), vertical axis (v), main diagonal (d), side diagonal (cd)
* ```extract (left_x) (top_y) (width) (height)```       Extract an image fragment with parameters: left indent (left_x, integer), top indent (top_y, integer), fragment width (width, positive), fragment height (height, positive)
* ```rotate {cw|ccw} (angle)```                         Clockwise (cw) or counterclockwise (ccw) by a given number of degrees, for example: rotate cw 90
* ```autorotate```                                      Automatic rotation by an angle of 0, 90, 180 or 270 degrees according to the proposed algorithm

