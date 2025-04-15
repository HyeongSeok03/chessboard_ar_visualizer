# Chessboard AR Visualizer

This project uses OpenCV to detect a chessboard pattern in video frames and renders a virtual cube on top of **a clicked square**, aligned using **camera pose estimation**.

The cube is drawn in perspective based on the camera's intrinsic and extrinsic parameters. Only **the top** and **two closest side faces** are rendered in white and gray to simulate 3D depth.

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

## Code Structure


```
```
