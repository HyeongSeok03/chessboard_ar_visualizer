import numpy as np
import cv2 as cv

# Play video_file, pause one frame with 'space', select image with 'enter', end selection with 'esc'
def select_img_from_video(video_file, board_pattern, wait_msec=10, wnd_name='Camera Calibration'):
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    img_select = []
    while True:
        valid, img = video.read()
        if not valid:
            break

        display = img.copy()
        cv.putText(display, f'NSelect: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
        cv.imshow(wnd_name, display)

        key = cv.waitKey(wait_msec)
        if key == ord(' '):
            complete, pts = cv.findChessboardCorners(img, board_pattern)
            if complete:
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow(wnd_name, display)
                key = cv.waitKey()
                if key == ord('\r'):
                    img_select.append(img)
        if key == 27:
            break
    
    cv.destroyAllWindows()
    return img_select

def calib_camera_from_chessboard(images, board_pattern, board_cellsize):
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0

    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points)

    rms, K, dist_coeff, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return rms, K, dist_coeff

def get_square_form_point(click_point, points, board_pattern):
    cols, rows = board_pattern
    width, height = rows - 1, cols - 1
    pts = np.array(points).reshape((rows, cols, 2))
    box_lower, box_upper = None, None
    for x in range(width):
        for y in range(height):
            square = np.array([
                pts[x][y],
                pts[x][y+1],
                pts[x+1][y+1],
                pts[x+1][y]
            ], dtype=np.float32)

            if cv.pointPolygonTest(square, click_point, False) >= 0:
                box_lower = np.array([
                    [y, x, 0],
                    [y + 1, x, 0],
                    [y + 1, x + 1, 0],
                    [y, x + 1, 0]
                ])
                box_upper = box_lower - [0, 0, 1]

    return box_lower, box_upper

def mouse_event_handler(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONUP:
        param['xy_e'] = (x, y)

if __name__ == '__main__':
    video_file = '../data/chessboard.MOV'
    board_pattern = (10, 7)
    board_cellsize = 0.025
    board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

    img_select = select_img_from_video(video_file, board_pattern)
    assert len(img_select) > 0, 'There is no selected images!'

    rms, K, dist_coeff = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)
    print(K)
    print(dist_coeff)

    mouse_state = {'xy_e': (0, 0)}
    cv.namedWindow('Pose Estimation')
    cv.setMouseCallback('Pose Estimation', mouse_event_handler, mouse_state)

    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    display = None
    click_point = (0, 0)
    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])], dtype=np.float32)
    cols, rows = board_pattern

    box_lower, box_upper = None, None

    while True:
        valid, display = video.read()
        if not valid:
            break

        success, img_points = cv.findChessboardCorners(display, board_pattern)
        
        if success:
            if mouse_state['xy_e'][0] > 0 and mouse_state['xy_e'][1] > 0:
                click_point = (mouse_state['xy_e'][0], mouse_state['xy_e'][1])
                box_lower, box_upper = get_square_form_point(click_point, img_points, board_pattern)
                mouse_state['xy_e'] = (0, 0)

            if (box_lower is not None):
                ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

                lower, _ = cv.projectPoints(box_lower * board_cellsize, rvec, tvec, K, dist_coeff)
                upper, _ = cv.projectPoints(box_upper * board_cellsize, rvec, tvec, K, dist_coeff)
                lower = np.int32(lower.reshape(-1, 2))
                upper = np.int32(upper.reshape(-1, 2))

                # Convert 'box_lower' to camera coordinates
                R, _ = cv.Rodrigues(rvec)
                transformed = (R @ (box_lower * board_cellsize).T).T + tvec.reshape(1, 3)
                
                # Calculate distances from the camera, then sort in ascending order
                distances = np.sum(transformed**2, axis = 1)
                sorted_indices = np.argsort(distances)

                # Get projected 2d points sorted by closest order
                lower_pts_closest_ord = lower[sorted_indices[:3]]
                upper_pts_closest_ord = upper[sorted_indices[:3]]

                # Get two close faces
                second_face = np.array([upper_pts_closest_ord[1], upper_pts_closest_ord[0], lower_pts_closest_ord[0], lower_pts_closest_ord[1]])
                third_face = np.array([upper_pts_closest_ord[0], upper_pts_closest_ord[2], lower_pts_closest_ord[2], lower_pts_closest_ord[0]])

                # Mapping line segments and colors using 'set' concepts
                color_map = {
                    frozenset({0, 1}): (128, 128, 128),
                    frozenset({1, 2}): (200, 200, 200),
                    frozenset({2, 3}): (128, 128, 128),
                    frozenset({3, 0}): (50, 50, 50)
                }

                # Get color based on absolute direction using color_map
                second_color = color_map[frozenset({sorted_indices[0], sorted_indices[1]})]
                third_color = color_map[frozenset({sorted_indices[0], sorted_indices[2]})]

                # Draw cube
                cv.fillConvexPoly(display, third_face, third_color)
                cv.fillConvexPoly(display, second_face, second_color)
                cv.fillConvexPoly(display, upper, (255, 255, 255))


        cv.imshow('Pose Estimation', display)
        
        key = cv.waitKey(10)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break