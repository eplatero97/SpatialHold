# Copyright (c) OpenMMLab. All rights reserved.
from jsonargparse import ArgumentParser, ActionConfigFile
import cv2
import torch
from mmdet.apis import inference_detector, init_detector
import matplotlib
import numpy as np
from mmcv import imshow_det_bboxes, imshow
import requests
# matplotlib.use("TkAgg")


def compare(n_employees: int, n_customers: int, past_results: dict):
    n_employees_diff = past_results["n_employees"] - n_employees
    n_customers_diff = past_results["n_customers"] - n_customers

    
    # if n_employees_diff != 0:
    #     url = f"https://customeralert.herokuapp.com/partnerCount/{n_employees}"
    #     requests.put(url)

    # if n_customers_diff != 0:
    #     url = f"https://customeralert.herokuapp.com/customerCount/{n_customers}"
    #     requests.put(url)

    if n_customers_diff != 0:
        url = f"https://customeralert.herokuapp.com/partnerCount/{n_customers}"
        requests.put(url)

    if n_employees_diff != 0:
        url = f"https://customeralert.herokuapp.com/customerCount/{n_employees}"
        requests.put(url)


def compute_centers(bboxes: np.ndarray) -> tuple:
    """
    :obj: compute center of bounding boxes
    :param bboxes: (n,4) or (n,5) shape with format: (tl, br) 
    """
    bbox_int = bboxes.astype(np.int32) # truncates confidence to 0
    tl_y, tl_x = (bbox_int[:,0], bbox_int[:,1])
    br_y, br_x = (bbox_int[:,2], bbox_int[:,3])
    mid_x = (tl_x + br_x) // 2
    mid_y = (tl_y + br_y) // 2
    return mid_x, mid_y

def compute_img_center(img: np.ndarray) -> tuple:
    _,h,w = img.shape
    img_mid_y = h // 2
    img_mid_x = w // 2
    return img_mid_x, img_mid_y


def draw_vert_line(img: np.ndarray) -> np.ndarray:
    _,h,w = img.shape
    _, img_mid_y = compute_img_center(img)
    start_pt = (img_mid_y, 0)
    end_pt = (img_mid_y, h)
    img = cv2.line(img, start_pt, end_pt, color = (0,255,0), thickness=1)
    return img

def parse_args():
    parser = ArgumentParser(description='MMDetection webcam demo')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--camera-id', type=int, default=0, help='camera device id')
    parser.add_argument(
        '--score-thr', type=float, default=0.5, help='bbox score threshold')
    parser.add_argument("--draw_vert_line", type=bool,default=False, help="whether to draw vertical line at center of img")
    parser.add_argument("--send_request", type=bool,default=False, help="whether to send request")
    parser.add_argument("--annt_img", type=bool,default=True, help="whether to annotate image or not")
    parser.add_argument("--mconfig", action=ActionConfigFile)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    device = torch.device(args.device)
    checkpoint = args.checkpoint if args.checkpoint != "None" else None
    model = init_detector(args.config, checkpoint, device=device)
    if checkpoint is None:
        from mmdet.core import get_classes
        model.CLASSES = get_classes("coco")

    camera = cv2.VideoCapture(args.camera_id)

    print('Press "Esc", "q" or "Q" to exit.')
    past_results = dict(n_employees = 0, n_customers=0)
    while True:
        ret_val, img = camera.read()
        result = inference_detector(model, img)

        # filter bboxes
        bboxes = result[0] # person bboxes
        is_conf = bboxes[:,4] > args.score_thr
        filt_bboxes = bboxes[is_conf,:]

        
        # compute center of person bboxes
        mid_x, mid_y = compute_centers(filt_bboxes)

        # draw vertical line on img
        if args.draw_vert_line:
            img = draw_vert_line(img)

        
        img_mid_x, img_mid_y = compute_img_center(img)
        labels = np.where(mid_y <= img_mid_y, 0, 1)

        # record metadata
        n_dets = len(labels)
        n_employees = sum(labels)
        n_customers = n_dets - n_employees

        if args.send_request:
            compare(n_employees, n_customers, past_results)
        

        class_names = {1: "customer", 0:"employee"}
        score_thr = args.score_thr
        bbox_color = text_color = {1:"green", 0:"red"}
        if args.annt_img:
            imshow_det_bboxes(img=img, 
                            bboxes=filt_bboxes, 
                            labels=labels,
                            class_names=class_names,
                            score_thr=score_thr,
                            bbox_color=bbox_color,
                            text_color=text_color,
                            wait_time=1)
        else:
            imshow(img, wait_time=1)

        past_results["n_employees"] = n_employees
        past_results["n_customers"] = n_customers









        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord('q') or ch == ord('Q'):
            break

        model.show_result(
            img, result, score_thr=args.score_thr, wait_time=1, show=True)


if __name__ == '__main__':
    main()
