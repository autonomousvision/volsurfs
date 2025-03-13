from rich import print
import os
import csv
from PIL import Image
import torch
from tqdm import tqdm

from piq import psnr, ssim, LPIPS

from volsurfs_py.utils.rendering import render_cameras_and_save_from_cameras
from mvdatasets.utils.images import image_to_tensor


# stores the results for a certain scene
class PerSceneEvaluator:
    def __init__(self, render_mode):
        self.render_mode = render_mode
        self.imgs_results = {}

    def update(self, img_name, psnr, ssim, lpips):
        # check if img_name already in results
        if img_name in self.imgs_results:
            print(
                f"[bold yellow]WARNING[/bold yellow]: {img_name} already evaluated, overwriting"
            )

        self.imgs_results[img_name] = {"psnr": psnr, "ssim": ssim, "lpips": lpips}

    def results_averaged(self):
        psnr = self.psnr_avg()
        ssim = self.ssim_avg()
        lpips = self.lpips_avg()

        return {"psnr": psnr, "ssim": ssim, "lpips": lpips}

    def psnr_avg(self):
        total_psnr = 0
        for img_name, img_res in self.imgs_results.items():
            total_psnr += img_res["psnr"]
        psnr = total_psnr / len(self.imgs_results)
        return psnr

    def ssim_avg(self):
        total_ssim = 0
        for img_name, img_res in self.imgs_results.items():
            total_ssim += img_res["ssim"]
        ssim = total_ssim / len(self.imgs_results)
        return ssim

    def lpips_avg(self):
        total_lpips = 0
        for img_name, img_res in self.imgs_results.items():
            total_lpips += img_res["lpips"]
        lpips = total_lpips / len(self.imgs_results)
        return lpips

    def save_to_csv(self, save_path, override_filename=None):
        all_rows = []

        print(f"results saved in {save_path}")
        if override_filename is not None:
            file_path = os.path.join(save_path, f"{override_filename}.csv")
        else:
            file_path = os.path.join(save_path, f"{self.render_mode}.csv")

        # save to csv file
        with open(file_path, "w") as csv_file:
            writer = csv.writer(csv_file)

            row = ["img_name", "psnr", "ssim", "lpips"]

            for img_name, img_res in self.imgs_results.items():
                row = [img_name, img_res["psnr"], img_res["ssim"], img_res["lpips"]]
                writer.writerow(row)
                all_rows.append(row)

            res_avg = self.results_averaged()
            row = ["avg", res_avg["psnr"], res_avg["ssim"], res_avg["lpips"]]
            writer.writerow(row)
            all_rows.append(row)

        return all_rows


# evaluates the model
@torch.no_grad()
def eval_rendered_imgs(renders_path, scene_name):
    """

    out:
        list of results (PerSceneEvaluator), one for each render mode
    """
    # iterate over folders in renders_path
    # each folder contains a different render mode
    # (e.g. "volumetric", "sphere_traced", ...)

    # check if path exists
    if not os.path.exists(renders_path):
        print(
            f"[bold red]ERROR[/bold red]: renders path {renders_path} for evaluation does not exist"
        )
        exit(1)

    # list all folders in renders_path
    render_modes = []
    for name in os.listdir(renders_path):
        if os.path.isdir(os.path.join(renders_path, name)):
            render_modes.append(name)
    print(f"found renders for rendering modalities: {render_modes}")

    render_modes_paths = [os.path.join(renders_path, folder) for folder in render_modes]

    results = []

    # unmasked
    for render_mode_path, render_mode in zip(render_modes_paths, render_modes):
        # print(f"evaluating render mode {render_mode}")

        # check if "gt" and "rgb" folders exists
        if os.path.exists(os.path.join(render_mode_path, "gt")) and os.path.exists(
            os.path.join(render_mode_path, "rgb")
        ):
            #

            # get all images filenames in gt
            # "000.png", "001.png", ... "999.png"
            img_filenames = os.listdir(os.path.join(render_mode_path, "gt"))
            # sort by name
            img_filenames.sort()

            # list all images in gt
            gt_path = os.path.join(render_mode_path, "gt")
            gt_imgs_paths = sorted(
                [os.path.join(gt_path, img_filename) for img_filename in img_filenames]
            )

            # load corresponding images in "rgb"
            rgb_path = os.path.join(render_mode_path, "rgb")
            pred_imgs_paths = sorted(
                [os.path.join(rgb_path, img_filename) for img_filename in img_filenames]
            )

            # load images and compute psnr, ssim, lpips

            scene_evaluator = PerSceneEvaluator(render_mode)
            scene_masked_evaluator = PerSceneEvaluator(render_mode + "_masked")

            print(f"[bold black]evaluating {render_mode}[/bold black]")
            print("[bold black]img_name, psnr, ssim, lpips[/bold black]")
            for img_filename, gt_img_path, pred_img_path in zip(
                img_filenames, gt_imgs_paths, pred_imgs_paths
            ):

                img_name = img_filename.split(".")[0]

                gt_img_pil = Image.open(gt_img_path)
                pred_img_pil = Image.open(pred_img_path)

                gt_rgb = image_to_tensor(gt_img_pil).cuda()
                pred_rgb = image_to_tensor(pred_img_pil).cuda()

                gt_rgb_tensor = gt_rgb.cuda()
                pred_rgb_tensor = pred_rgb.cuda()
                gt_rgb_tensor = gt_rgb_tensor.permute(2, 0, 1).unsqueeze(0)
                pred_rgb_tensor = pred_rgb_tensor.permute(2, 0, 1).unsqueeze(0)

                psnr_val = psnr(pred_rgb_tensor, gt_rgb_tensor, data_range=1.0).item()
                ssim_val = ssim(pred_rgb_tensor, gt_rgb_tensor, data_range=1.0).item()
                lpips_val = LPIPS()(pred_rgb_tensor, gt_rgb_tensor).item()

                print(
                    f"[bold black]{img_name}[/bold black]",
                    psnr_val,
                    ssim_val,
                    lpips_val,
                )

                scene_evaluator.update(img_name, psnr_val, ssim_val, lpips_val)

                # # check if "mask" folder exists
                # if (
                #     os.path.exists(os.path.join(render_mode_path, "mask"))
                # ):

                #     # load masks, mask gt and pred to compute psnr, ssim, lpips

            results.append(scene_evaluator)

    # # masked
    # for render_mode_path, render_mode in zip(render_modes_paths, render_modes):
    #     # print(f"evaluating render mode {render_mode}")

    #         # get all images filenames in gt
    #         # "000.png", "001.png", ... "999.png"
    #         img_filenames = os.listdir(os.path.join(render_mode_path, "masked_gt"))
    #         # sort by name
    #         img_filenames.sort()

    #         # list all images in gt
    #         gt_path = os.path.join(render_mode_path, "masked_gt")
    #         gt_imgs_paths = sorted(
    #             [os.path.join(gt_path, img_filename) for img_filename in img_filenames]
    #         )

    #         # load corresponding images in "masked_rgb"
    #         rgb_path = os.path.join(render_mode_path, "masked_rgb")
    #         pred_imgs_paths = sorted(
    #             [os.path.join(rgb_path, img_filename) for img_filename in img_filenames]
    #         )

    #         test_results = PerSceneEvaluator(render_mode + "_masked")
    #         print(f"[bold black]evaluating {render_mode}[/bold black]")
    #         print("[bold black]img_name, psnr, ssim, lpips[/bold black]")
    #         for img_filename, gt_img_path, pred_img_path in zip(img_filenames, gt_imgs_paths, pred_imgs_paths):

    #             img_name = img_filename.split(".")[0]

    #             gt_img_pil = Image.open(gt_img_path)
    #             pred_img_pil = Image.open(pred_img_path)

    #             gt_rgb = image_to_tensor(gt_img_pil).cuda()
    #             pred_rgb = image_to_tensor(pred_img_pil).cuda()

    #             gt_rgb_tensor = gt_rgb.cuda()
    #             pred_rgb_tensor = pred_rgb.cuda()
    #             gt_rgb_tensor = gt_rgb_tensor.permute(2, 0, 1).unsqueeze(0)
    #             pred_rgb_tensor = pred_rgb_tensor.permute(2, 0, 1).unsqueeze(0)

    #             psnr_val = psnr(pred_rgb_tensor, gt_rgb_tensor, data_range=1.0).item()
    #             ssim_val = ssim(pred_rgb_tensor, gt_rgb_tensor, data_range=1.0).item()
    #             # lpips_val = LPIPS()(pred_rgb_tensor, gt_rgb_tensor).item()
    #             lpips_val = 0.0

    #             print(f"[bold black]{img_name}[/bold black]", psnr_val, ssim_val, lpips_val)

    #             test_results.update(img_name, psnr_val, ssim_val, lpips_val)

    #         results.append(test_results)

    return results


@torch.no_grad()
def render_and_eval(method, mv_data, data_splits, iter_nr=None):
    """
    runs evaluation during training

    args:
        method: method object
        mv_data: MVDataset object
        data_splits: list of str, evaluation modes (e.g. "train", "test")
        iter_nr: int, iteration number
    """

    # render images
    renders_path = os.path.join(
        method.save_checkpoints_path, format(iter_nr, "07d"), "renders"
    )
    results_path = os.path.join(method.save_checkpoints_path, "results")

    # run evaluation for each eval mode
    for data_split in data_splits:
        print(f"\nrunning rendering on {data_split} set")

        # check if renders folder exists
        if not os.path.exists(os.path.join(renders_path, data_split)):

            print(
                f"[bold blue]INFO[/bold blue]: {data_split} renders folder not found, rendering images to {renders_path}"
            )
            os.makedirs(os.path.join(renders_path, data_split), exist_ok=True)

            # only render if there are no renders
            render_cameras_and_save_from_cameras(
                cameras=mv_data[data_split],
                method=method,
                renders_path=os.path.join(renders_path, data_split),
                iter_nr=iter_nr,
            )

        else:

            print(
                f"[bold yellow]WARNING[/bold yellow]: {data_split} renders already exist in {renders_path}, skipping rendering"
            )

    # evaluate

    # prepare output dict
    eval_res = {}
    for data_split in data_splits:
        eval_res[data_split] = dict()

    # run evaluation for each split
    for data_split, eval_dict in eval_res.items():
        print(f"\nrunning rendering on {data_split} set")

        # check if results_path/data_split.csv exists
        if os.path.exists(os.path.join(results_path, f"{data_split}.csv")):

            print(
                f"[bold yellow]WARNING[/bold yellow]: {data_split} results exists in {method.save_checkpoints_path}, skipping evaluation"
            )

        else:

            print(
                f"[bold blue]INFO[/bold blue]: evaluating {data_split} renders in {renders_path}"
            )

            if os.path.exists(os.path.join(renders_path, data_split)):
                print(f"[bold blue]INFO[/bold blue]: found renders for {data_split}")
            else:
                print(
                    f"[bold red]ERROR[/bold red]: renders for {data_split} not found in {renders_path}"
                )
                exit(1)

            render_modes_eval_res = eval_rendered_imgs(
                os.path.join(renders_path, data_split), scene_name=mv_data.scene_name
            )

            for res in render_modes_eval_res:
                res_avg = res.results_averaged()
                eval_dict.update(res_avg)
                # print results
                print(f"render mode: {res.render_mode}")
                for key, value in res_avg.items():
                    print(f"{key}: {value}")
                # store results to csv
                res.save_to_csv(os.path.join(renders_path, data_split))

            # create results dir if not exists
            os.makedirs(results_path, exist_ok=True)

            # save results to csv
            # TODO: assumption: only one render mode is evaluated
            for res in render_modes_eval_res:
                res_avg = res.results_averaged()
                eval_dict.update(res_avg)
                # print results
                print(f"render mode: {res.render_mode}")
                for key, value in res_avg.items():
                    print(f"{key}: {value}")
                # store results to csv
                res.save_to_csv(results_path, override_filename=f"{data_split}")

    return eval_res
