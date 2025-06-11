import cv2
import numpy as np
import os
import time

# 相机参数
mat_inter = np.array([[1.45425805e+03, 0.00000000e+00, 2.01856871e+03],
       [0.00000000e+00, 1.45786958e+03, 1.47997968e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
coff_dis = np.array([[-0.08081855, -0.02421897,  0.00031707,  0.00018403,  0.01068942]])
# 全局变量
g_window_name = "img"  # 窗口名
g_window_wh = [1920, 1080]  # 窗口宽高

g_location_win = [0, 0]  # 相对于大图，窗口在图片中的位置
location_win = [0, 0]  # 鼠标左键点击时，暂存g_location_win
g_location_click, g_location_release = [0, 0], [0, 0]  # 相对于窗口，鼠标左键点击和释放的位置

g_zoom, g_step = 1, 0.1  # 图片缩放比例和缩放系数

# 矫正窗口在图片中的位置
def check_location(img_wh, win_wh, win_xy):
    for i in range(2):
        if win_xy[i] < 0:
            win_xy[i] = 0
        elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] > win_wh[i]:
            win_xy[i] = img_wh[i] - win_wh[i]
        elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] < win_wh[i]:
            win_xy[i] = 0

# 计算缩放倍数
def count_zoom(flag, step, zoom):
    if flag > 0:  # 滚轮上移
        zoom += step
        if zoom > 1 + step * 20:  # 最多只能放大到3倍
            zoom = 1 + step * 20
    else:  # 滚轮下移
        zoom -= step
        if zoom < step:  # 最多只能缩小到0.1倍
            zoom = step
    zoom = round(zoom, 2)  # 取2位有效数字
    return zoom

class Image_Possion:
    def __init__(self, img_folder_path, img_num):
        self.folder_path = img_folder_path
        self.img_num = img_num
        self.img_id = None

        self.index = None
        self.load_list = []
        self.filepath_list = []
        
        self.possion_list = []

        self.end_signal = False
        
    def dedistortion(self, w, h):
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mat_inter, coff_dis, (w, h), 0, (w, h))  # 自由比例参数
        self.img = cv2.undistort(self.img, mat_inter, coff_dis, None, newcameramtx)
        
    def get_sort_filenames(self):
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.JPG'):
                file_path = os.path.join(self.folder_path, filename)
                split_filename = filename.split('.')
                split_filename = split_filename[0] +'.' + split_filename[1]
                
                parts = split_filename.split('_')
                self.index = int(float(parts[0].replace('index', '')))
                load_ = parts[-1].split('.')
                load = float(load_[0].replace('load', ''))
                self.load_list.append(load)
                self.filepath_list.append(file_path)

    def mouse(self, event, x, y, flags, param):
        global g_location_click, g_location_release, g_image_zoom, g_location_win, location_win, g_zoom

        if event == cv2.EVENT_LBUTTONDOWN:
            '''
            鼠标左键点击后，保留缩放后的坐标信息到coordinates中
            '''
            # 转换坐标到原始图像坐标系
            x_orig = int((x + g_location_win[0]) / g_zoom)
            y_orig = int((y + g_location_win[1]) / g_zoom)

            font = cv2.FONT_HERSHEY_SIMPLEX
            txt = 'Coord: (' + str(x_orig) + ', ' + str(y_orig) + ')'
            cv2.putText(self.img, txt, (x_orig, y_orig), font, 0.3, (255, 0, 0), 2)
            cv2.circle(self.img, (x_orig, y_orig), 2, (255, 0, 0), -1)  # 2是圆的半径，(255, 0, 0)是蓝色
            self.coordinates.append((x_orig, y_orig))
            
        elif event == cv2.EVENT_RBUTTONDOWN:
            '''
            鼠标右键点击后，代表选点工作已结束，判断是否为正确的流程
            '''
            input_key = input('T代表这四个点标注正确,数据将会被保存    F代表标注不正确, 将自动重新开始标注')
            if input_key.lower() == 't':
                print(self.coordinates)
                self.save_coordinates(self.img_id)
                self.end_signal = True
                
            elif input_key.lower() == 'f':
                print('请重新标记关键点')
                pass
            self.coordinates = []  
            self.img = self.original_img.copy()
        elif event == cv2.EVENT_MBUTTONDOWN:  # 中键点击
            g_location_click = [x, y]  # 点击时，鼠标相对于窗口的坐标
            location_win = [g_location_win[0], g_location_win[1]]  # 窗口相对于图片的坐标
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_MBUTTON):  # 按住中键拖曳
            g_location_release = [x, y]  # 左键拖曳时，鼠标相对于窗口的坐标
            h1, w1 = g_image_zoom.shape[0:2]  # 缩放图片的宽高
            w2, h2 = g_window_wh  # 窗口的宽高
            show_wh = [0, 0]  # 实际显示图片的宽高
            if w1 < w2 and h1 < h2:  # 图片的宽高小于窗口宽高，无法移动
                show_wh = [w1, h1]
                g_location_win = [0, 0]
            elif w1 >= w2 and h1 < h2:  # 图片的宽度大于窗口的宽度，可左右移动
                show_wh = [w2, h1]
                g_location_win[0] = location_win[0] + g_location_click[0] - g_location_release[0]
            elif w1 < w2 and h1 >= h2:  # 图片的高度大于窗口的高度，可上下移动
                show_wh = [w1, h2]
                g_location_win[1] = location_win[1] + g_location_click[1] - g_location_release[1]
            else:  # 图片的宽高大于窗口宽高，可左右上下移动
                show_wh = [w2, h2]
                g_location_win[0] = location_win[0] + g_location_click[0] - g_location_release[0]
                g_location_win[1] = location_win[1] + g_location_click[1] - g_location_release[1]
            check_location([w1, h1], [w2, h2], g_location_win)  # 矫正窗口在图片中的位置
            self.g_image_show = g_image_zoom[g_location_win[1]:g_location_win[1] + show_wh[1], g_location_win[0]:g_location_win[0] + show_wh[0]]  # 实际显示的图片
        elif event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
            z = g_zoom  # 缩放前的缩放倍数，用于计算缩放后窗口在图片中的位置
            g_zoom = count_zoom(flags, g_step, g_zoom)  # 计算缩放倍数
            w1, h1 = [int(self.img.shape[1] * g_zoom), int(self.img.shape[0] * g_zoom)]  # 缩放图片的宽高
            w2, h2 = g_window_wh  # 窗口的宽高
            g_image_zoom = cv2.resize(self.img, (w1, h1), interpolation=cv2.INTER_AREA)  # 图片缩放
            show_wh = [0, 0]  # 实际显示图片的宽高
            if w1 < w2 and h1 < h2:  # 缩放后，图片宽高小于窗口宽高
                show_wh = [w1, h1]
                cv2.resizeWindow(g_window_name, w1, h1)
            elif w1 >= w2 and h1 < h2:  # 缩放后，图片高度小于窗口高度
                show_wh = [w2, h1]
                cv2.resizeWindow(g_window_name, w2, h1)
            elif w1 < w2 and h1 >= h2:  # 缩放后，图片宽度小于窗口宽度
                show_wh = [w1, h2]
                cv2.resizeWindow(g_window_name, w1, h2)
            else:  # 缩放后，图片宽高大于窗口宽高
                show_wh = [w2, h2]
                cv2.resizeWindow(g_window_name, w2, h2)
            g_location_win = [int((g_location_win[0] + x) * g_zoom / z - x), int((g_location_win[1] + y) * g_zoom / z - y)]  # 缩放后，窗口在图片的位置
            check_location([w1, h1], [w2, h2], g_location_win)  # 矫正窗口在图片中的位置
            self.g_image_show = g_image_zoom[g_location_win[1]:g_location_win[1] + show_wh[1], g_location_win[0]:g_location_win[0] + show_wh[0]]  # 实际的显示图片
        cv2.imshow(g_window_name, self.g_image_show)
            
    def img_init(self, img_path):
        self.img = cv2.imread(img_path)
        self.original_img = self.img.copy()
        self.g_image_show = self.img[g_location_win[1]:g_location_win[1] + g_window_wh[1], g_location_win[0]:g_location_win[0] + g_window_wh[0]]  # 实际显示的图片
        cv2.namedWindow(g_window_name)
        cv2.resizeWindow(g_window_name, g_window_wh[0], g_window_wh[1])
        cv2.moveWindow(g_window_name, 700, 100)  # 设置窗口在电脑屏幕中的位置
        cv2.setMouseCallback(g_window_name, self.mouse)

    def img_operation(self):
        while True:
            cv2.imshow(g_window_name, self.g_image_show)
            k = cv2.waitKey(1)
            if self.end_signal:
                self.end_signal = False
                break
            if k == ord('q'):
                break
        cv2.destroyAllWindows()
    
    def save_coordinates(self, img_id):
        coordinates = np.array(self.coordinates)
        max_x = max(coordinates[:, 0])
        min_x = min(coordinates[:, 0])
        max_y = max(coordinates[:, 1])
        min_y = min(coordinates[:, 1])
        temp_rec = (max_x, min_x, max_y, min_y)
        self.min_reclist.append(temp_rec)
        print(self.min_reclist)

    def calculate_possion(self):
        origin_length = self.min_reclist[0][0] - self.min_reclist[0][1]
        origin_height = self.min_reclist[0][2] - self.min_reclist[0][3]
        for i in range(len(self.filepath_list)-1):
            current_length = self.min_reclist[i+1][0] - self.min_reclist[i+1][1]
            current_height = self.min_reclist[i+1][2] - self.min_reclist[i+1][3]
            delta_length = current_length - origin_length
            delta_height = current_height - origin_height
            strain_length = delta_length / origin_length
            strain_height = delta_height / origin_height
            possion = - strain_length / strain_height
            self.possion_list.append(possion)
        
    def main(self):
        self.get_sort_filenames()
        if self.img_num == len(self.filepath_list):
            self.min_reclist = []
            for img_id in range(len(self.filepath_list)):
                img_path = self.filepath_list[img_id]
                load = self.load_list[img_id]
                self.img_id = img_id
                self.coordinates = []  # 初始化坐标列表
                self.img_init(img_path)
                self.img_operation()
            self.calculate_possion()
        else:
            print('当前文件夹下的图像数目与输入的图像数目不符，请重新检查数据')
        
if __name__ == "__main__":
    folder_path = 'E:/01_Graduate_projects/Cellular_structures/Multi-functional_design/exp/exps/index11_load10.16'
    img_num = 6
    image_process = Image_Possion(folder_path, img_num)
    image_process.main()
    for i in range(img_num-1):
        print('应变={}%时的泊松比：{}'.format(image_process.load_list[i] * 20, image_process.possion_list[i]))
