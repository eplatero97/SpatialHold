# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp
import tempfile
from jsonargparse import ArgumentParser, ActionConfigFile
import numpy as np 
from loguru import logger
from pathlib import Path
import mmcv
from typing import List, Optional
from mmdet.apis import inference_detector, init_detector
from mmtrack.apis import inference_mot, init_model

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--det_config', type=str, help='config file')
    parser.add_argument('config', type=str, help='config file')
    parser.add_argument('--input', type=str, help='input video file or folder with images as frames')
    parser.add_argument(
        '--output', type=str, help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', type=str, help='checkpoint file of model weights')
    parser.add_argument('--det_checkpoint', type=str, help='checkpoint file of model weights')
    parser.add_argument(
        '--score_thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', type=str, default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether show the results on the fly')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', help='FPS of the output video')
    parser.add_argument('--pkl_results', type=bool, default=False, help='whether to pickle results')
    parser.add_argument("--mconfig", action=ActionConfigFile)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    input = Path(args.input)

    assert args.output or args.show
    # load images
    """
    There are three-formats supposrted:

    input_dir_with_vids
    |__ vid0.mp4
    |__ vid1.mp4
    |__ vid2.mp4
    . . .

    input_dir_with_frames
    |__ frame_000.jpg
    |__ frame_001.jpg
    |__ frame_002.jpg
    . . .

    input_vid.mp4
    """
    if input.is_dir():
        
        subFiles: List[Path] = list(input.iterdir())
        subFile: str = subFiles[0].name
        vid_exts = (".mov", ".MOV", ".mp4")
        img_exts = (".jpg", ".png", ".jpeg")
        if subFile.endswith(vid_exts):
            # directory is assumed to contain video subfiles
            output = Path(args.output)
            assert output.is_dir(), f"since input path ({args.input}) is a directory with videos, output path ({args.output}) MUST also be a directory"
            imgs_per_vid: list = [mmcv.VideoReader(str(input / vid)) for vid in subFiles]
            IN_VIDEO = True
            # create subdirectories to storey each video
            vidNames: List[str] = [vid.stem for vid in subFiles] # extract names of files w/o extension
            [(output / vidName).mkdir(exist_ok=True, parents=True) for vidName in vidNames]
        elif subFile.endswith(img_exts):
            # directory is assumed to contain frame subfiles
            imgs_per_vid = [sorted(
                filter(lambda x: x.name.endswith(img_exts),
                    subFiles),
                key=lambda x: int(x.split('.')[0]))]
            
            IN_VIDEO = False
        else:
            logger.error(f"children file {subFile} of directory {args.input} is neither an image or video")
            raise 
    else:
        # parse video into its frames
        imgs_per_vid = [mmcv.VideoReader(args.input)]
        IN_VIDEO = True
    # define output
    if args.output is not None:
        output = Path(args.output)
        if output.suffix == ".mp4":
            OUT_VIDEO = True # output annotated video
            out_dir = tempfile.TemporaryDirectory()
            out_path = Path(out_dir.name)
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            OUT_VIDEO = False # do NOT output annotated video (just frames)
            out_path: Path = output
            os.makedirs(out_path, exist_ok=True)

    logger.info(f"base output path: {out_path}")

        
    fps = args.fps
    if args.show or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps: list = [imgs.fps for imgs in imgs_per_vid]
            
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        if isinstance(fps, list):
            if len(fps) == 1:
                fps = int(fps[0])
        else:
            fps = int(fps)

    # define conditions 
    isDirWithMultipleVids = input.is_dir() and IN_VIDEO
    outputIsDefined = args.output is not None
    outputIsDefinedOrIsAVideo = outputIsDefined and OUT_VIDEO # NOTE: `OUT_VIDEO` is ONLY `True` if `args.output` points to a *.mp4 video
    inputIsProcessingVideoOrOutputAVideo = IN_VIDEO or OUT_VIDEO # NOTE: `IN_VIDEO` is ONLY `True` if `args.input` points to a video file OR to a directory with video files

    # iterate through each video
    base_out_path: Path = out_path
    for j,imgs in enumerate(imgs_per_vid):
        
        if isDirWithMultipleVids:
            vid_name = os.listdir(args.input)[j]
            vid_forename = osp.splitext(vid_name)[0]
            out_path: Path = base_out_path / vid_forename
        
        # build the model from a config file and a checkpoint file
        model = init_model(args.config, args.checkpoint, device=args.device)
        # build detection model
        det_model = init_detector(args.det_config, args.det_checkpoint, device=args.device)
        model.detector = det_model
        model.CLASSES = model.detector.CLASSES

        n_imgs = len(imgs)
        prog_bar = mmcv.ProgressBar(n_imgs)
        # test and show/save the images
        results = []
        
        # iterate through each frame in current video
        for i, img in enumerate(imgs):
            if isinstance(img, str):
                img = str(input / img)
            # attain model predictions
            result = inference_mot(model, img, frame_id=i)
            if args.pkl_results:
                # save prediction 
                results.append(result)
            # define output path of annotated image
            out_file: Optional[str] = None
            if outputIsDefined:
                if inputIsProcessingVideoOrOutputAVideo:
                    out_file = str(out_path / "frames" / f'{i:06d}.jpg')
                else:
                    out_file = str(out_path / img.rsplit(os.sep, 1)[-1])
            # visualize results
            model.show_result(
                img,
                result,
                score_thr=args.score_thr,
                show=args.show,
                wait_time=int(1000. / fps) if fps else 0,
                out_file=out_file,
                backend=args.backend)
            prog_bar.update()

        if args.pkl_results:
            # modify path
            out_pkl_file = str(out_path / "results.npy")
            np.save(out_pkl_file, results)

        if outputIsDefinedOrIsAVideo:
            logger.info(f'making the output video at {args.output} with a FPS of {fps}')
            frames_path = str( out_path / "frames" )
            mmcv.frames2video(frames_path, args.output, fps=fps, fourcc='mp4v')
            out_dir.cleanup()
        elif isDirWithMultipleVids:
            output_vid = str( out_path / (vid_forename + "_annt.mp4") )
            logger.info(f'making the output video at {output} with a FPS of {fps}')
            frames_path = str( out_path / "frames" )
            mmcv.frames2video(frames_path, output_vid, fps=fps, fourcc='mp4v')
            


if __name__ == '__main__':
    main()
