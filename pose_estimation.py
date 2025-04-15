import numpy as np
import cv2 as cv

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

    mouse_state = {'xy_e': (0, 0)}
    cv.namedWindow('Pose Estimation')
    cv.setMouseCallback('Pose Estimation', mouse_event_handler, mouse_state)

    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    display = None
    click_points = (0, 0)
    obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])
    cols, rows = board_pattern

    box_lower, box_upper = None, None

    while True:
        valid, display = video.read()
        if not valid:
            break

        success, img_points = cv.findChessboardCorners(display, board_pattern, board_criteria)

        if mouse_state['xy_e'][0] > 0 and mouse_state['xy_e'][1] > 0:
            click_points = (mouse_state['xy_e'][0], mouse_state['xy_e'][1])
            pts = np.array(img_points).reshape((rows, cols, 2))
            for x in range(rows - 1):
                for y in range(cols - 1):
                    # 한 네모칸을 구성하는 4개 점 좌표 (좌상단부터 시계 방향)
                    square = np.array([
                        pts[x][y],
                        pts[x][y+1],
                        pts[x+1][y+1],
                        pts[x+1][y]
                    ], dtype=np.float32)

                    # 클릭한 점이 해당 네모 내부에 있는지 확인
                    if cv.pointPolygonTest(square, click_points, False) >= 0:
                        box_lower = np.array([
                            [y, x, 0],
                            [y + 1, x, 0],
                            [y + 1, x + 1, 0],
                            [y, x + 1, 0]
                        ])
                        box_upper = box_lower - [0, 0, 1]
            mouse_state['xy_e'] = (0, 0)

        if (box_lower is not None):
            ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

            lower, _ = cv.projectPoints(box_lower * board_cellsize, rvec, tvec, K, dist_coeff)
            upper, _ = cv.projectPoints(box_upper * board_cellsize, rvec, tvec, K, dist_coeff)
            lower = np.int32(lower.reshape(-1, 2))
            upper = np.int32(upper.reshape(-1, 2))

            R, _ = cv.Rodrigues(rvec)
            transformed = (R @ (box_lower * board_cellsize).T).T + tvec.reshape(1, 3)
            
            distances = np.sum(transformed**2, axis = 1)

            sorted_indices = np.argsort(distances)

            lower_pts = lower[sorted_indices[:3]]
            upper_pts = upper[sorted_indices[:3]]

            second_face = np.array([upper_pts[1], upper_pts[0], lower_pts[0], lower_pts[1]])
            third_face = np.array([upper_pts[0], upper_pts[2], lower_pts[2], lower_pts[0]])

            cv.fillConvexPoly(display, third_face, (128, 128, 128))
            cv.fillConvexPoly(display, second_face, (128, 128, 128))
            cv.fillConvexPoly(display, upper, (255, 255, 255))


        cv.imshow('Pose Estimation', display)
        
        key = cv.waitKey(10)
        if key == ord(' '):
            key = cv.waitKey()
        if key == 27: # ESC
            break