import yaml
import argparse
import torch
import os
import io
import shutil
from PIL import Image
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from utils import *
from dataset import get_dataset, get_loader
from models import get_model, get_loss
import lpips
import tqdm
from tqdm import tqdm
import open3d as o3d
from scipy.signal import windows

try:
    from skimage.measure import compare_ssim
except:
    from skimage.metrics import structural_similarity

    def compare_ssim(gt, img, win_size, channel_axis=2):
        return structural_similarity(
            gt, img, win_size=win_size, channel_axis=channel_axis, data_range=1.0
        )


def parse_args():
    parser = argparse.ArgumentParser(description="PAPR")
    parser.add_argument("--opt", type=str, default="", help="Option file path")
    parser.add_argument("--resume", type=int, default=250000, help="Resume step")
    return parser.parse_args()


def get_depth(pc, rayo):
    od = -rayo
    dists = np.abs(np.sum((pc - rayo) * od, axis=-1)) / np.linalg.norm(od)
    return dists


def get_batch(dataloader, index):
    for i, batch in enumerate(dataloader):
        if i == index:
            return batch
    raise IndexError("Index out of range")


def plot_pc(pc, colors, feat=False, focal=0.17):
    elev, azim, roll = -90, -90, 0
    pltscale = 0.03
    size = 12.8
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(projection="3d")
    ax.set_proj_type("persp", focal_length=focal)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_axis_off()

    if feat is True:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], alpha=1.0, s=50, c=colors)
    else:
        cname = "plasma"
        cmap = matplotlib.colormaps.get_cmap(cname)
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], alpha=1.0, s=80, c=cmap(-colors + 1))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    lim = pltscale * 1.0
    xlim = lim * 1.0
    ylim = lim * 1.0
    zlim = lim * 1.0
    ax.set_xlim3d(-xlim, xlim)
    ax.set_ylim3d(-ylim, ylim)
    ax.set_zlim3d(-zlim, zlim)

    plt.tight_layout()
    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)

    plt.close()

    return img


def get_all_weights_with_prefix(model_state_dict, prefix):
    weights = {}
    for key, value in model_state_dict.items():
        if prefix in key:
            weights[key] = value.detach().cpu()
    return weights


def smooth_point_cloud(point_clouds, window_size):
    # Create an empty list to store the smoothed point clouds
    smoothed_point_clouds = []
    # Pad the list of point clouds with copies of the first and last point clouds
    padded_point_clouds = ([point_clouds[0]] * (window_size // 2)) + point_clouds + ([point_clouds[-1]] * (window_size // 2))
    # Apply the moving average filter
    for i in range(window_size // 2, len(padded_point_clouds) - window_size // 2):
        # Get the window of point clouds for this iteration
        window = padded_point_clouds[i - window_size // 2 : i + window_size // 2 + 1]
        # Calculate the average point cloud for this window
        average_point_cloud = np.mean(window, axis=0)
        # Add the average point cloud to the list of smoothed point clouds
        smoothed_point_clouds.append(average_point_cloud)
    # Set the first and last point clouds to be the same as the original ones
    smoothed_point_clouds[0] = point_clouds[0]
    smoothed_point_clouds[-1] = point_clouds[-1]

    return smoothed_point_clouds


def plot_all(frame, testloader, pc_dir, args, model, device):
    batch = get_batch(testloader, frame)
    rgb_save_dir = os.path.join(args.save_dir, args.index, f"rgb_interpolation_{frame}")
    os.makedirs(rgb_save_dir, exist_ok=True)
    pc_save_dir = os.path.join(args.save_dir, args.index, f"pc_interpolation_{frame}")
    os.makedirs(pc_save_dir, exist_ok=True)
    idx, _, img, rayd, rayo = batch
    c2w = testloader.dataset.get_c2w(idx.squeeze())
    c2w_np = c2w.detach().cpu().numpy()

    # load the model weights after the second stage (appearance finetuning)
    end_model_state_dict = torch.load(
        os.path.join(args.save_dir, args.stage_two_path, "model.pth")
    )
    for step, state_dict in end_model_state_dict.items():
        end_model = state_dict
    end_points = end_model["points"].data.cpu().numpy()
    end_pc_feats = end_model["pc_feats"].data.cpu().numpy()
    end_attn_weights = get_all_weights_with_prefix(end_model, "proximity_attn")
    init_attn_weights = get_all_weights_with_prefix(
        model.state_dict(), "proximity_attn"
    )

    result = {}

    N, H, W, _ = rayd.shape
    num_pts, _ = model.points.shape

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    img = img.to(device)
    c2w = c2w.to(device)

    topk = min([num_pts, model.select_k])
    bkg_seq_len_attn = 0
    feat_dim = args.models.attn.embed.value.d_ff_out
    if model.bkg_feats is not None:
        bkg_seq_len_attn = model.bkg_feats.shape[0]

    # the number of iterations with both regularization enabled, the change in geometry is
    # more significant in this stage, so keep more samples from here than later stage
    fast_iters = (
        20000 if "fast_iters" not in args.test.keys() else int(args.test.fast_iters)
    )
    # the iteration number when LDAS is disabled and with only rigid loss.
    slow_iters = (
        20000 if "slow_iters" not in args.test.keys() else int(args.test.slow_iters)
    )
    skip_num = 4 if "skip_num" not in args.test.keys() else int(args.test.skip_num)
    focal = 0.17 if "focal" not in args.test.keys() else float(args.test.focal)
    dist_type = 1 if "dist_type" not in args.test.keys() else int(args.test.dist_type)
    # find all point cloud files and sort them by name, the name convention is points_{step}.npy
    # fist find all the point clouds for the fast iterations
    pc_files = [
        os.path.join(pc_dir, f)
        for f in os.listdir(pc_dir)
        if os.path.isfile(os.path.join(pc_dir, f))
        and f.startswith("points_")
        and f.endswith(".npy")
        and int(os.path.splitext(os.path.basename(f))[0][7:]) <= fast_iters
    ]
    pc_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0][7:]))
    # select every skip_num files in pc_files
    pc_files = pc_files[::skip_num]

    # add the point clouds for the slow iterations
    slow_iter_pcs = [
        os.path.join(pc_dir, f)
        for f in os.listdir(pc_dir)
        if os.path.isfile(os.path.join(pc_dir, f))
        and f.startswith("points_")
        and f.endswith(".npy")
        and int(os.path.splitext(os.path.basename(f))[0][7:]) > slow_iters
    ]
    slow_iter_pcs.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0][7:]))
    # select every 4*skip_num files in pc_files (fewer samples are needed for the slow iterations)
    slow_iter_pcs = slow_iter_pcs[::(skip_num * 4)]
    # append the slow iterations to the fast iterations
    pc_files.extend(slow_iter_pcs)

    rgb_results = []
    pc_results = []

    pc_files.append(end_points)

    # load all the pc files
    pcs = [np.load(pc_file) if isinstance(pc_file, str) else pc_file for pc_file in pc_files]
    smoothing_window_size = 7 if "smoothing_window_size" not in args.test.keys() else int(args.test.smoothing_window_size)
    pcs = smooth_point_cloud(pcs, smoothing_window_size)

    init_points = pcs[0]
    init_pc_feats = model.pc_feats.data.cpu().numpy()
    max_diff = np.sum(np.linalg.norm(init_points - end_points, axis=1))

    counter = 0
    for pc_file in tqdm(pc_files):
        pc = pcs[counter]
        model.points = torch.nn.Parameter(torch.from_numpy(pc).to(device))

        cur_max_pt_diffs = np.sum(np.linalg.norm(init_points - pc, axis=1))
        cur_ratio = min(cur_max_pt_diffs / max_diff, 1.0)

        # interpolate the point feature
        cur_pc_feats = init_pc_feats * (1.0 - cur_ratio) + end_pc_feats * cur_ratio
        model.pc_feats = torch.nn.Parameter(torch.from_numpy(cur_pc_feats).to(device))
        # interpolate attention weights
        for key, value in init_attn_weights.items():
            cur_value = (
                value * (1.0 - cur_ratio) + end_attn_weights[key] * cur_ratio
            )
            model.state_dict()[key].copy_(cur_value.to(device))

        feature_map = torch.zeros(N, H, W, 1, feat_dim).to(device)
        attn = torch.zeros(N, H, W, topk + bkg_seq_len_attn, 1).to(device)

        with torch.no_grad():
            cur_gamma, cur_beta, code_mean = None, None, 0
            for height_start in range(0, H, args.test.max_height):
                for width_start in range(0, W, args.test.max_width):
                    height_end = min(height_start + args.test.max_height, H)
                    width_end = min(width_start + args.test.max_width, W)
                    feature_map[:, height_start:height_end, width_start:width_end, :, :], attn[:, height_start:height_end, width_start:width_end, :, :] = model.evaluate(rayo, rayd[:, height_start:height_end, width_start:width_end], c2w, step=resume_step)

            if args.models.use_renderer:
                foreground_rgb = model.renderer(feature_map.squeeze(-2).permute(0, 3, 1, 2), gamma=cur_gamma, beta=cur_beta).permute(0, 2, 3, 1).unsqueeze(-2)   # (N, H, W, 1, 3)
            else:
                foreground_rgb = feature_map

            if model.bkg_feats is not None:
                bkg_attn = attn[..., topk:, :]
                bkg_mask = (model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn).squeeze()
                if args.models.normalize_topk_attn:
                    rgb = foreground_rgb * (1 - bkg_attn) + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                else:
                    rgb = foreground_rgb + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                rgb = rgb.squeeze(-2)
            else:
                rgb = foreground_rgb.squeeze(-2)
                bkg_mask = torch.zeros(N, H, W, 1).to(device)

            rgb = model.last_act(rgb)
            rgb = torch.clamp(rgb, 0, 1)

        rgb = rgb.squeeze().detach().cpu().numpy().astype(np.float32)
        rgb = (rgb * 255).astype(np.uint8)
        rgb_results.append(rgb)
        rgb = Image.fromarray(rgb)
        if isinstance(pc_file, str):
            iter_num = int(os.path.splitext(os.path.basename(pc_file))[0][7:])
            save_img_name = f"{iter_num}.jpg"
        else:
            save_img_name = "final.jpg"
        rgb.save(os.path.join(rgb_save_dir, save_img_name))

        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        pc.transform(np.linalg.inv(c2w_np @ blender2opencv))
        cur_points = np.asarray(pc.points)
        if dist_type == 0:
            dists = np.sqrt((cur_points**2).sum(axis=1))
        else:
            dists = get_depth(cur_points, c2w_np[:3, -1])
        colors = (dists - dists.min()) / (dists.max() - dists.min())
        img = plot_pc(cur_points, colors, focal=focal)
        # convert img from RGBA to RGB with white background
        img.convert("RGB").save(os.path.join(pc_save_dir, save_img_name))
        pc_results.append(img)
        counter += 1

    result["rgb"] = rgb_results
    result["pc"] = pc_results
    return result


def test(model, device, dataset, save_name, args, resume_step):
    pc_dir = os.path.join(args.save_dir, args.stage_one_path, "test", "point_clouds")

    testloader = get_loader(dataset, args.dataset, mode="test")
    print("testloader:", testloader)

    frame = args.test.frame_idx

    frames = plot_all(
        frame,
        testloader,
        pc_dir,
        args,
        model,
        device,
    )

    if frames:
        for key, value in frames.items():
            name = f"{args.index}-{frame}-{key}.mp4"
            f = os.path.join(args.save_dir, args.index, name)
            imageio.mimwrite(f, value, fps=30, quality=10)
            name = f"{args.index}-{frame}-{key}-loop.mp4"
            f = os.path.join(args.save_dir, args.index, name)
            # revese value and append to itself
            new_value = value[::-1]
            for _ in range(4):
                value.append(value[-1])
            value.extend(new_value)
            imageio.mimwrite(f, value, fps=30, quality=10)


def main(args, save_name, mode, resume_step=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, device)
    dataset = get_dataset(args.dataset, mode=mode)

    model_state_dict = torch.load(
        os.path.join(args.save_dir, args.stage_one_path, "model.pth")
    )
    for step, state_dict in model_state_dict.items():
        resume_step = int(step)
        model.load_my_state_dict(state_dict)
    print(
        "!!!!! Loaded model from %s at step %s" % (args.stage_one_path, resume_step)
    )

    model = model.to(device)

    test(model, device, dataset, save_name, args, resume_step)


if __name__ == "__main__":

    with open("configs/default.yml", 'r') as f:
        default_config = yaml.safe_load(f)

    args = parse_args()
    with open(args.opt, "r") as f:
        config = yaml.safe_load(f)

    test_config = copy.deepcopy(default_config)
    update_dict(test_config, config)

    resume_step = args.resume

    log_dir = os.path.join(test_config["save_dir"], test_config["index"])
    os.makedirs(log_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(log_dir, 'test.log'), sys.stdout)
    sys.stderr = Logger(os.path.join(log_dir, 'test_error.log'), sys.stderr)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    shutil.copyfile(args.opt, os.path.join(log_dir, os.path.basename(args.opt)))

    setup_seed(test_config['seed'])

    for i, dataset in enumerate(test_config['test']['datasets']):
        name = dataset['name']
        mode = dataset['mode']
        print(name, dataset)
        test_config['dataset'].update(dataset)
        test_config = DictAsMember(test_config)
        main(test_config, name, mode, resume_step)
