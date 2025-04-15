# Chessboard AR Visualizer

This project uses OpenCV to detect a chessboard pattern in video frames and renders a virtual cube on top of **a clicked square**, aligned using **camera pose estimation**.

The cube is drawn in perspective based on the camera's intrinsic and extrinsic parameters. Only **the top** and **two closest side faces** are rendered in white and gray to simulate 3D depth.

## Democration Video
[Youtube](https://youtu.be/OBXHJy_vRdY)

## Features

- Camera calibration from video using chessboard pattern
- **`Click-based selection of a chessboard square`**
- Pose estimation using **`solvePnP`**
- Perspective projection of a virtual cube aligned with the board
- **`Face culling based on camera distance`** (renders top + 2 closest side faces)
- Interactive visualization via OpenCV

## File Structure

```
chessboard_ar_visualizer/
│
├── pose_estimation.py                    # Main script for calibration + AR visualization
├── data/
│   └── chessboard.MOV         # Input video containing chessboard pattern
└── README.md
```

## Problems faced

### 1. The order of the drawing faces

if...
```
cv.fillConvexPoly(display, top_face, colors["top"])
cv.fillConvexPoly(display, side_face_1, colors["side_1"])
cv.fillConvexPoly(display, side_face_2, colors["side_2"])
cv.fillConvexPoly(display, side_face_3, colors["side_3"])
cv.fillConvexPoly(display, side_face_4, colors["side_4"])
cv.fillConvexPoly(display, bottom_face, colors["bottom"])
```
-> The top face is hidden by the faces drawn later, and the subsequent faces are also hidden sequentially.

This is how I solved it.
```
R, _ = cv.Rodrigues(rvec)
transformed = (R @ (box_lower * board_cellsize).T).T + tvec.reshape(1, 3)

distances = np.sum(transformed**2, axis = 1)
sorted_indices = np.argsort(distances)

lower_pts = lower[sorted_indices[:3]]
upper_pts = upper[sorted_indices[:3]]
second_face = np.array([upper_pts[1], upper_pts[0], lower_pts[0], lower_pts[1]])
third_face = np.array([upper_pts[0], upper_pts[2], lower_pts[2], lower_pts[0]])
```
1. Convert 'box_lower' to camera coordinates
2. Calculate distances from the camera, then sort in ascending order
3. Get projected 2d points sorted by closest order
4. Get two close faces

### 2. Light and shade expression using differential colors according to the absolute direction of the surface

The above method can find the two faces closest to the camera, but this is only a relative position and does not give the absolute direction of the east, west, south, and north sides of the chessboard.

How to draw a cube that was bright on the east side and dark on the west side, based on the direction of the light? Must choose colors based on absolute positions into account.

```
The bottom face of the cube are indexed clockwise as 0, 1, 2, 3, starting from the upper left corner.
0--------1
|        |
|        |
|        |
3--------2
```
For example, the situation is (3, 2, 0, 1), if the distance from the camera is calculated and sorted in closest order.

Line segment (3, 2) is the lower segment of the closest side face, and (3, 0) is the lower segment of the second closest side face.

```
color_map = {
    frozenset({0, 1}): (128, 128, 128),
    frozenset({1, 2}): (200, 200, 200),
    frozenset({2, 3}): (128, 128, 128),
    frozenset({3, 0}): (50, 50, 50)
}

second_color = color_map[frozenset({sorted_indices[0], sorted_indices[1]})]
third_color = color_map[frozenset({sorted_indices[0], sorted_indices[2]})]
```
1. Mapping line segments and colors using 'set' concepts
2. Get color based on absolute direction using color_map

### Lastly...

```
cv.fillConvexPoly(display, third_face, third_color)
cv.fillConvexPoly(display, second_face, second_color)
cv.fillConvexPoly(display, upper, (255, 255, 255))
```
-> Draw the three sides in the opposite order.
