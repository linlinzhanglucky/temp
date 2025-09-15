import os
import numpy as np
import cv2
import argparse
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot
from evogym.utils import get_full_connectivity
def find_bg_img(img, black_thresh=40):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    black_rows = np.all(gray < black_thresh, axis=1)
    H, W, _ = img.shape

    vis = np.zeros((H, W, 3), dtype=np.uint8)  # black by default

    # Set rows to white in all 3 channels
    vis[black_rows] = [255, 255, 255]

    return vis

def show_side_by_side(bg_img, img):
    # 确保所有图像都是 uint8 类型，且尺寸一致
    assert bg_img.shape == img.shape, "Image sizes must match"
    assert bg_img.dtype == np.uint8 and img.dtype == np.uint8, "Images must be uint8"

    # 防止加法溢出，先转换为更高类型再 clip 回 0~255，再转回 uint8
    added_img = np.clip(bg_img.astype(np.int16) + img.astype(np.int16), 0, 255).astype(np.uint8)

    # 拼接三张图像：横向拼接
    combined = np.hstack([bg_img, img, added_img])

    # 显示图像
    cv2.imshow("bg_img | img | bg_img + img", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

file_path = os.path.abspath(__file__)

file_folder_name = "evo_gym_dataset_0717"
folder_path = os.path.dirname(file_path)
input_folder = os.path.join(folder_path, file_folder_name, "input")
label_folder = os.path.join(folder_path, file_folder_name, "label")
raw_folder = os.path.join(folder_path, file_folder_name, "raw")

os.makedirs(input_folder, exist_ok=True)
os.makedirs(label_folder, exist_ok=True)
os.makedirs(raw_folder, exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--render-option', 
    choices=['to-debug-screen', 'to-numpy-array', 'special-options', 'very-fast'],
    help='Select a rendering option from: to-debug-screen, to-numpy-array, special-options, very-fast',
    default= 'to-numpy-array'
)
args = parser.parse_args()

num_samples = 10000
num_frames = 40
steps_per_frame = 5
total_steps = num_frames * steps_per_frame
white_thresh = 240
start_epoch = 0

sin_actuater = True

### CREATE A SIMPLE ENVIRONMENT ###
# create world
for i in range(start_epoch, num_samples):
    # data folder
    video_folder = os.path.join(input_folder, f"V{i:06d}")
    os.makedirs(video_folder, exist_ok=True)

    img_folder = os.path.join(label_folder, f"V{i:06d}")
    os.makedirs(img_folder, exist_ok=True)

    color_video_folder = os.path.join(raw_folder, f"V{i:06d}")
    os.makedirs(color_video_folder, exist_ok=True)
    # add world
    world = EvoWorld.from_json(os.path.join('world_data', 'simple_environment.json'))
    # add robot
    robot_structure = np.random.randint(1, 5, size=(3, 4))
    robot_structure[1:4, 1:3] = 0
    contains_3_or_4 = np.any((robot_structure == 3) | (robot_structure == 4))
    if not contains_3_or_4:
        i -= 1
        continue
    
    print(robot_structure)
    robot_connections = get_full_connectivity(robot_structure)
    world.add_from_array(
        name='robot',
        structure=robot_structure, 
        x=10,
        y=1, 
        connections=robot_connections)

    # create simulation 
    sim = EvoSim(world)
    sim.reset()

    # set up viewer
    viewer = EvoViewer(sim)
    viewer.track_objects('robot')
    viewer.set_tracking_settings(
    lock_x=viewer.pos[0], 
    lock_y=viewer.pos[1], 
    lock_width=viewer.view_size[0],
    lock_height=viewer.view_size[1]
    )



    ### SELECT A RENDERING OPTION ###


    print(f'\nUsing rendering option {args.render_option}...\n')

    # if the 'very-fast' option is chosen, set the rendering speed to be unlimited
    if args.render_option == 'very-fast':
        viewer.set_target_rps(None)
    print(i)

    for j in range(num_frames):
        img_name = f"f{j:06d}.png"
        
        if args.render_option == 'to-numpy-array':
            img = viewer.render('img')
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            bg_img = find_bg_img(img)
            # show_side_by_side(bg_img, img)
            white_mask = np.all(img >= white_thresh, axis=2)
            bw_img = np.zeros_like(img, dtype=np.uint8)
            bw_img[white_mask] = 255
            file_path = os.path.join(video_folder, img_name)
            color_file_path = os.path.join(color_video_folder, img_name)

            cv2.imwrite(file_path, np.clip(bg_img.astype(np.int16) + bw_img.astype(np.int16), 0, 255).astype(np.uint8))
            cv2.imwrite(color_file_path, np.clip(bg_img.astype(np.int16) + img.astype(np.int16), 0, 255).astype(np.uint8))
            if j == 0:
                file_path = os.path.join(img_folder, img_name)
                cv2.imwrite(file_path, np.clip(bg_img.astype(np.int16) + img.astype(np.int16), 0, 255).astype(np.uint8))
        for substep in range(steps_per_frame):
            # sim.set_action(
            #     'robot', 
            #     np.random.uniform(
            #         low = 0.4,
            #         high = 1.6,
            #         size=(sim.get_dim_action_space('robot'),))
            #     )
            sim.set_action(
                'robot', 
                np.full((sim.get_dim_action_space('robot'),),
                        1 + 0.6 * np.sin(2 * np.pi * 4 * (j * steps_per_frame + substep)/total_steps))
            )
            # sim.set_action(
            #     'robot', 
            #     np.full((sim.get_dim_action_space('robot'),),
            #             1.6)
            # )
                # np.random.uniform(
                #     low = 0.4,
                #     high = 1.6,
                #     size=(sim.get_dim_action_space('robot'),))
            sim.step()
        

        # step and render to a debug screen
        if args.render_option == 'to-debug-screen':
            img = viewer.render('img')
            viewer.render('screen')

        # step and render to a numpy array
        # use open cv to visualize output


            cv2.waitKey(1)
            # cv2.imshow("Open CV Window", img)

        # rendering with more options
        if args.render_option == 'special-options':   
            img = viewer.render(
                'screen', 
                verbose = True,
                hide_background = False,
                hide_grid = True,
                hide_edges = False,
                hide_voxels = False)

        # rendering as fast as possible
        if args.render_option == 'very-fast':
            viewer.render('screen', verbose=True)

    cv2.destroyAllWindows()
    viewer.close()