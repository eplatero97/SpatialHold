import cv2
from typing import List, Tuple, Union, Optional
import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image, ImageDraw
import pandas as pd 
from random import randint
import sys
import os
from  loguru import logger 

# need to run `python setup.py install_lib` @ root of project for below to import
# from mot_utils import max_trackID

################## mot_utils code ####################
def max_trackID(preds_perframe: List[np.ndarray]) -> int:
    """
    :obj: calculate maximum tracklet ID
    :param preds_perframe: list where each element contains all predictions of a specific frame
    :return max_trackID_: integer containing maximum tracklet ID
    """
    # what is the maximum trackID?
    trackIDs = []
    [trackIDs.extend(pred_perframe[:,1].tolist()) for pred_perframe in preds_perframe]
    max_trackID_ = max(trackIDs)
    return max_trackID_

################## bbox-related code ##################


def bbox_centers(bboxes: Union[List[List], np.ndarray], is_coco = False) -> Union[List[List[int]], np.ndarray]:
    """
    :obj: attain center coordinates for each bounding box in `bboxes`
    :param bboxes: nx4 array-like object  
    :param is_coco: if False, bboxes come in [min_x, min_y, max_x, max_y] format. Else, [min_x, min_y, width, length]
    :return centers: nx2 array-like where features represent center coordinates of bboxes in (mid_x, mid_y) format
    ----------------------------------------------------------------------------
    steps:
    * iterate through each bbox:
        * Unpack the bbox coordinates
        * store coordinates in `centers`
    * return `centers`
    """

    is_list = type(bboxes) == list

    if is_list:
        centers = []
        for bbox in bboxes:
            if is_coco:
                min_x, min_y, width, length = bbox # unpack coordinates
                
                mid_x = min_x + (width // 2) # compute mid xs
                mid_y = min_y + (length // 2) # compute mid ys
            else:
                min_x, min_y, max_x, max_y = bbox # unpack coordinates
                
                # compute mid xs
                width = max_x - min_x
                mid_x = min_x + (width // 2)
                
                # compute mid ys
                lenght = max_y - min_y
                mid_y = min_y + (length // 2)
            
            mid_coord = [mid_x, mid_y]
            centers.append(mid_coord)
    elif type(bboxes) == np.ndarray:
        if is_coco:
            min_xs, min_ys, widths, lengths = np.hsplit(bboxes, 4) # unpack coordinates
            
            # compute mid xs
            half_widths = widths // 2
            mid_xs = (min_xs + half_widths).reshape(-1, 1)
            
            # compute mid ys
            half_lengths = lengths // 2
            mid_ys = (min_ys + half_lengths).reshape(-1, 1)
            
            # calculate centers
            centers = np.concatenate([mid_xs, mid_ys], axis = 1)
            
        
        else:
            min_xs, min_ys, max_xs, max_ys = np.hsplit(bboxes, 4) # unpack coordinates
            
            # compute mid xs
            half_widths = (max_xs - min_xs) // 2
            mid_xs = (min_xs + half_widths).reshape(-1, 1)
            
            # compute mid ys
            half_lengths = (max_ys - min_ys) // 2
            mid_ys = (min_ys + half_lengths).reshape(-1, 1)
            
            # calculate centers
            centers = np.concatenate([mid_xs, mid_ys], axis = 1)

    else:
        logging.debug(f"bboxes argument is supposed to be of type list or np.ndarray. It is actually type: \n{type(bboxes)}")
        raise 

    return centers


def bbox2circle(bboxes: Union[List[List], np.ndarray], is_coco = False, take_max = True) -> Union[Tuple[list, List[List[int]]], Tuple[np.ndarray, np.ndarray]]:
    """
    :obj: transform each bbox in `bboxes` to circles by returning the circle radius and circle centers
    :method: make circumference of circle equal the max(length, width)
    :param bboxes: each element contains 4 integers representing the bbox in format [min_x, min_y, max_x, max_y] if is_coco == False, else [min_x, min_y, height, width]
    :param is_coco: if False, bboxes come in [min_x, min_y, max_x, max_y] format. Else, [min_x, min_y, width, length]
    :param take_max: if False, make circumeference of each bbox equal min(length, width) dimension of bbox. Else, take max
    :return circle_params: each tuple element contains circle parameters in the form of (radius, center_coordinate)
    ----------------------------------------------------------------------------
    steps:
    * iterate through each bbox:
        * Unpack the bbox coordinates
        * calculate radius and center coordinates
        * store circle parameters
    * return `circle_params`
    """

    is_list = type(bboxes) == list
    
    # store center of each circle
    circle_centers: Union[List[List[int, int]], np.ndarray] = bbox_centers(bboxes, is_coco)

    # storage radius of each circle
    if is_list:
        circle_radius = []
        for bbox in bboxes:
            
            if is_coco:
                min_x, min_y, width, length = bbox
            else:
                min_x, min_y, max_x, max_y = bbox
                length = max_x - min_x
                width = max_y - min_y

            if take_max:
                circumference = max(length, width)
            else:
                circumference = min(length, width)
            radius = circumference // 2
            
            circle_radius.append(radius)


    else:
        # compute radius associated with each bounding box
        if is_coco:
            wls = bboxes[:, 2:] # width lengths of each bbox
            
        else:
            min_xs, min_ys, max_xs, max_ys = np.hsplit(bboxes, 4)
            widths = (max_ys - min_ys).reshape(-1, 1)
            lenghts = (max_xs - min_xs).reshape(-1, 1)
            wls = np.concatenate([widths, lenghts], axis = 1)
        
        if take_max:
            circumferences = np.max(wls, axis = 1) # (n,) dim
        else:
            circumferences = np.min(wls, axis = 1) # (n,) dim
        
        circle_radius = circumferences // 2

    return circle_radius, circle_centers
        


def annotate_tracklets(frame: np.ndarray, 
                       bboxes: List[list], 
                       tracklets: list, 
                       colors: list, 
                       continuity: Optional[dict], 
                       font = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, thickness = 5) -> np.ndarray:

    """
    :obj: annotate tracklets associated with each bounding box in given frame
    :param frame: frame to annotate
    :param bboxes: each element list is of shape (n x 4) where `n` represents number of detections in current frame. bbox format: (tlx,tly,width,height)
    :param tracklets: list of shape (n) where `n` represents the number of tracklets associated with each detection in current frame
    :param colors: list whose idx represents a tracklet ID and whose value represents its color 
    :param continuity: tracks the center coordinates of all recurrent tracklets (tracklets that have continuously appeared in previous and current frame) (modified inplace)
    :param font, fontScale, thickness: cv2 parameters relating to annotations of bboxes and tracklets
    :return frame: return annotated frame
    """


    # annotate bboxes and tracklets
    for bbox,tracklet_ID in zip(bboxes, tracklets):

        # extract color associated with tracklet
        color = colors[tracklet_ID]

        # annotate bbox
        min_x, min_y, width, length = bbox
        max_x = min_x + width
        max_y = min_y + length

        start_pt = (min_x, min_y)
        end_pt = (max_x, max_y)
        frame = cv2.rectangle(frame, start_pt, end_pt, color, 3)

        # annotate tracklets
        org = (min_x - 15, min_y - 10)
        frame = cv2.putText(frame, f"{tracklet_ID}", org, font, fontScale, (0,255,0))

        # what is the center coordinate of our bbox?
        curr_center_coord = (min_x + width // 2, min_y + length // 2)

        # did the current tracklet appear in the previous frame?
        if tracklet_ID in continuity:


            # if so, draw a straight line between every adjacent pair
            cont_coords: list = continuity[tracklet_ID] + [curr_center_coord] # continuous coords


            # annotate past coordinates
            for i in range(1, len(cont_coords)):

                curr_pt = cont_coords[i]
                prev_pt = cont_coords[i-1]

                cv2.line(frame, prev_pt, curr_pt, color, thickness)

            # update new coords
            continuity[tracklet_ID] = cont_coords

        else:
            continuity[tracklet_ID] = [curr_center_coord]


    return frame

def are_tracklets_continuous(curr_tracklets: list, continuity: dict) -> None:
    """
    :obj: did any of the current tracklets appear in previous frame?
    :purpose: this is to know whether to draw a line of past coordinates of current tracklets if they have continuity
    :method: the keys in `continuity` represent tracklets that have had continuity up to the previous frame. If any of the past tracklet keys are not in the current tracklets, we delete them as they no longer have continuity
    :param curr_tracklets: list of shape (n) where `n` represents the number of tracklets associated with each detection in current frame
    :param continuity: tracks the center coordinates of all recurrent tracklets (tracklets that have continuously appeared in previous and current frame) (modified inplace if a tracklet does not have continuity)
    :return: None, modifies `continuity` inplace if a tracklet no longer has continuity
    """

    # are any of the prev tracklet IDs in the curr tracklet IDs? 
    prev_tracklets = list(continuity.keys())
    for prev_tracklet in prev_tracklets:
        if prev_tracklet in curr_tracklets:
            # keep key as it does appear in current frame
            continue
        else:
            # delete key as it does not appear in current frame
            del continuity[prev_tracklet]


################## headhunter relted code ##################
def coerce_preds(preds: List[dict]) -> List[dict]: 
    """
    :obj: coerce HeadHunter preds from torch tensors to list objects
    :param preds: single-item list that contains dictionary with predictions as torch objects
    """
    for key,val in preds[0].items():
    
        if key == 'boxes':
            preds[0][key] = val.cpu().int().numpy().tolist()
        else:
            preds[0][key] = val.cpu().numpy().tolist()
    
    return preds


def draw_bboxes(input_img: 'torch img', preds: List[dict]) -> 'PIL img':
    """
    :obj: draw HeadHunter bbox predictions
    :param input_img: torch image that was fed to calculate predictions
    :param preds: single-item list that contains dictionary with predictions as list objects
                  NOTE: `preds` to be in the format presented after `coerce_preds` function
    :return: PIL image with bbox annotations predicted from HeadHunter
    This function depends on:
        * `coerce_preds` function
    """
    
    fed_img = (input_img.cpu() * 255).int().numpy().astype(np.uint8).transpose(1,2,0)
    pil_img = Image.fromarray(fed_img)

    draw = ImageDraw.Draw(pil_img)
    for bbox in preds[0]['boxes']:
        draw.rectangle(xy = bbox, outline = 'red', width = 3)
    return pil_img

#################################### FCFS Data Association Algorithm ################################################
def resolve_conflict(PB_ranks_PA: np.ndarray, zeroth_duplicate_PB: int, conflicted_prevIDs: np.ndarray) -> np.int:
    """
    :obj: if >=2 prevIDs are fithing over a currID, resolve conflict by choosing the pair that minimizes distance w.r.t. currID
    :param PB_ranks_PA: created through `np.argsort(pair_dists, axis = 0)`
    :param zeroth_duplicate_PB: column indice that is getting "fought" over by >=2 prev indices
    :param conflicted_prevIDs: prevIDs that are "fighting" over `zeroth_duplicate_PB`
    :return conf_winner: `conf_winner` contains the row indice of prevID that keeps the currID pair 
    """
    idx_ranks = PB_ranks_PA[:, zeroth_duplicate_PB] # lenght equal to number of prevIDs
    is_conflicted = np.isin(idx_ranks, conflicted_prevIDs) # lenght equal to number of prevIDs
    conf_order = np.nonzero(is_conflicted) # conflicted error (length equal to number of conflicted prevIDs)
    conf_rank = idx_ranks[conf_order] # conflicted rank (length equal to number of conflicted prevIDs)
    conf_winner: np.int = conf_rank[0] # prevID indice that minizes distance w.r.t. `zeroth_duplicate_PB`
    return conf_winner


# association module
def spatial_bias_assignment(pair_dists: np.ndarray) -> np.ndarray:
    """
    :obj: implement minimum weight matching between the pairwise distances of the previous and current bboxes
    :param pair_dists: pairwise weight matrix between previous and current bboxes
    :return PA_min_PB: 1d array where indices represent prevIDs and 
                                             values represent their currID indice pairings
    :terminilogy PA: Partition A (row indices)
    :terminology PB: Partition B (col indices)
    """
    logger.info(f"input matrix: \n{pair_dists}")
    logger.info(f"input matrix shape: \n{pair_dists.shape}")
    
    # calculate min dimension
    n, m = pair_dists.shape
    min_dim = min(n,m)
    
    
    # for every node in PB, "rank" its corresponding node in PA based on closest proximity (e.g., `pair_dists[0][1]==0 # True` indicates closest node of node 1 from PB is node 0 from PA)
    # NOTE: just because prevID is the closes to currID does NOT mean currID is the closes to prevID
    PB_ranks_PA = np.argsort(pair_dists, axis = 0) # nxm vector
    

    # attain "top pick" currID per prevID (the currID closest to prevID)
    # for every node in PA, attain PB indice with minimum distance w.r.t. PA node (e.g., `PA_min_PB[1]==2 # True` means that node 2 of PB minimizes the distance w.r.t. node 1 in PA)
    PA_min_PB = np.argmin(pair_dists, axis = 1) # n vector
    
    unique_PA_min_PB, unique_PA_min_PB_counts = np.unique(PA_min_PB, return_counts= True) # unique_PA_min_PB_counts only needed if a "conflict" arises
    PA_min_PB_picks_are_unique = (len(PA_min_PB) == len(unique_PA_min_PB)) # did each node `i` in PA have a unique node `j` that minimized its distance (`PA_min_PB[i]==j`)?

    n_conflicts = -1
    theres_conflict = not PA_min_PB_picks_are_unique
    while theres_conflict:
        
        n_conflicts += 1
        logger.info(f"conflict ID: {n_conflicts}")
        
        # if we have paired up all possible pairs, then we have reached our limit! We have to stop the draft now
        # this will ALWAYS happen if there are more prevIDs than currIDs (not everyone can receive a pair!)
        if n_conflicts == min_dim:
            logger.info(f"we have paired {min_dim} nodes from partition A to partition B. As such, we have reached our limit of pairs since minimum dim of matrix is {min_dim}")
            theres_conflict = False
            break 
        
        # check which currIDs are getting "fought" over for
        PB_indice_duplicates = unique_PA_min_PB[unique_PA_min_PB_counts > 1]
        
        
        # pick first duplicate node in PB
        zeroth_duplicate_PB: np.int = PB_indice_duplicates[0]
        logger.info(f"PB duplicate indice: {zeroth_duplicate_PB}")

        # find specific PA nodes that are "fighting" over duplicate node in PB
        conflicted_prevIDs = np.flatnonzero(PA_min_PB == zeroth_duplicate_PB) # vals represents currIDs
        logger.info(f"PA nodes with duplicate PB node: {conflicted_prevIDs}")

        # find which prevID the currID "prefers" (the prevID that minimizes the distance with currID)
        zeroth_duplicate_PB_min_PA: np.int = resolve_conflict(PB_ranks_PA, zeroth_duplicate_PB, conflicted_prevIDs) # value represents PA node that min. distance w.r.t. duplicate node PB


        # modify value of `pair_dists[prevID, currID]` to artificially have distance of zero (assumption is that no one else will have this and thus no racing condition)
        #   modify all other to have an artificially high value so that no one is able to choose it 
        pair_dist = pair_dists[zeroth_duplicate_PB_min_PA, zeroth_duplicate_PB]
        pair_dists[zeroth_duplicate_PB_min_PA, :] = np.Inf # make entire row associated with node PA equal inf
        pair_dists[:, zeroth_duplicate_PB] = np.Inf # make entire col associated with node PB equal inf
        pair_dists[zeroth_duplicate_PB_min_PA, zeroth_duplicate_PB] = pair_dist # insert previous value of selected pair
        logger.info(f"Input Matrix after conflict ID {n_conflicts}: \n{pair_dists}")


        '''
        recompute all computations that were dependent on `pair_dists` since we modified the object
        '''

        # sort based on "top picks" per previous ID (row-wise)
        PB_ranks_PA = np.argsort(pair_dists, axis = 0) # nxm matrix

        # attain "top pick" prevID row idx for every currID
        PA_min_PB = np.argmin(pair_dists, axis = 1) # n matrix
        
        
        unique_PA_min_PB, unique_PA_min_PB_counts = np.unique(PA_min_PB, return_counts= True) # unique_PA_min_PB_counts only needed if a "conflict" arises
        PA_min_PB_picks_are_unique = (len(PA_min_PB) == len(unique_PA_min_PB))

        theres_conflict = not PA_min_PB_picks_are_unique
        

    '''no conflicts should occur after this point'''
    
    
    PA_min_PB = np.argmin(pair_dists, axis = 1) # n vector
    
    '''
    each prevID will be paired with a currID (even if its a duplicate)
    Indice of `PA_min_PB` represents prevID while value represents the "top pick" curr ID of each prevID
    '''
    
    return PA_min_PB



# assign tracklets
def assignIDs(prevIDs: 'np 1d array', n_newIDs: int) -> np.ndarray:
    """
    :obj: assign IDs to each unique tracklet
    :assmpt: rows are the "gt" or prevIDs and cols are currIDs of matrix `W`
    :param W:
    """

    # extract maximum ID of `prevIDs`
    maxPrevID: np.int = np.max(prevIDs)

    # curr IDs that need new tracklets
    start_step = maxPrevID + 1
    end_step = start_step + n_newIDs
    newIDs = np.arange(start_step, end_step)

    return newIDs



def skipped_frames(annts: list) -> list:
    """
    :obj: are there any frames that did not contain any annotations?
    :param annts: computed by accesssing `coco["annotations"]` section of coco annotations
    :property: if a frame was skipped, then the `frame_id`s in `annts` will NOT be linear
    :return global_skipped_IDs: returns the `image_id`s that were skipped
                                NOTE: `image_id`s start at idx 1 (NOT zero)
    """
    prevID = 0
    
    global_skipped_IDs = []
    for annt in annts:
        # what is the `image_id` of the corresponding `annt`?
        imgID: int = annt['image_id']
        # what is the hop from previous `image_id` and current?
        hop: int = imgID - prevID
        if hop > 1:

            # previous and current ID do NOT follow a linear pattern (record skipped `image_id`s)
            local_skipped_IDs = list(range(prevID + 1, imgID ))
            global_skipped_IDs.extend(local_skipped_IDs)
            
        prevID = imgID
            
    return global_skipped_IDs



# which are pairs are within the radius of their associated previus ID?
def is_within_radius(pair_dists: np.ndarray, 
                     PA_min_PB: np.ndarray, 
                     prev_params: Union[list, np.ndarray]) -> Tuple[np.ndarray]:
    """
    :obj: if `pair_dists[prevID, currID] < radius_of_prevID`, keep pair; else, remove pair
    :param pair_dists: pairwise distance matrix
    :param PA_min_PB: indice represents prevID while value represents col idx of `pair_dists`. Together, pairing is defined
    :param prev_params: radiuses of each prevID
    :return selected_prevIDs, selected_currIDs: both arrays contain indices of IDs whose distance is < radius_detection_of_prevID
    """
    n = pair_dists.shape[0]
    rows = np.arange(n)

    # extract distance between prevID and its "top pick" currID 
    selected_pair_dists = pair_dists[rows, PA_min_PB].astype(int) # shape: (n,)

    # which pair has a distance less-than that of the radius of the detection of the prevID?
    is_within = (selected_pair_dists < np.asarray(prev_params))
    selected_prevIDs = np.flatnonzero(is_within) # extract all prevID indices whose distance pair is < prev_detection_radius of prevID
    selected_currIDs = PA_min_PB[selected_prevIDs] # extract all currID indices whose distance pair is < prev_detection_radius of prevID

    return selected_prevIDs, selected_currIDs



def resolve_duplicate_currIDs(pair_dists: np.ndarray, prevIDs: np.ndarray, currIDs: np.ndarray, m: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    :obj: remove all `prevIDs` and `currIDs` pairs except for the one that minimizes the distance 
    :purpose: `prevIDs` and `currIDs` need to have a 1:1 mapping
    :reason: each prevID tracklet can only match one currID (a prevID can't associate itself with two different detections since we assume that each bbox contains a unique object)
    :param pair_dists: pairwise distance matrix
    :param prevIDs: array whose values represent the row indice in `pair_dists`. 
    :param currIDs: array whose value represents the col indice in `pair_dists`. 
    :param m: number of detections in current frame
    :return: Modified `prevIDs` and `currIDs` if duplicate currIDs are found
    :predecessor func: function `is_within_radius` was executed before executing this function
    """
    
    # are there any duplicate currIDs (aka a currID that has been chosen by >=2 different prevIDs)?
    unique_currIDs, unique_PA_min_PB_counts = np.unique(currIDs, return_counts = True)
    there_are_duplicate_currIDs = (len(unique_currIDs) != len(currIDs))
    if there_are_duplicate_currIDs:

        # what are the indices of the duplicate currID?
        duplicate_currIDs = unique_currIDs[unique_PA_min_PB_counts > 1]


        for duplicate_currID in duplicate_currIDs:


            # what are the indices in `currIDs` that contain `duplicate_currID`?
            duplicate_currID_indices_wrt_currIDs = np.flatnonzero(currIDs == duplicate_currID)

            # what are the value indices of the "fighting" prevIDs who selected the same `duplicate_currID`?
            fighting_prevIDs = prevIDs[duplicate_currID_indices_wrt_currIDs]

            # what is the distance between each "fighting" currID and `duplicate_currID`?
            duplicate_dists = pair_dists[fighting_prevIDs, duplicate_currID]

            # what is the indice in `duplicate_dists` that contains the minimum distance pair between prevID and `duplicate_currID`?
            retained_prevID = np.argmin(duplicate_dists)
            
            # what is the indice of the prevID that minimized the distance with `duplicate_currID`?
            retained_prevID_val = fighting_prevIDs[retained_prevID]

            # what are the prevIDs that do NOT match `retained_prevID_val`?
            is_prevID_not_retained = (prevIDs != retained_prevID_val)

            # what are the currIDs that match `duplicate_currID`?
            is_currID_duplicate = (currIDs == duplicate_currID)

            # what is the pair that we need to discard (True if we need to discard pair) (only pairs we need to discard meet this condition as True)?
            is_pair_discarded = (is_prevID_not_retained == is_currID_duplicate)

            # what are the pairs we need to keep (True if we keep pair)?
            is_pair_retained = (is_pair_discarded == False)

            # filter kept pairs
            prevIDs = prevIDs[is_pair_retained]
            currIDs = currIDs[is_pair_retained]
            
    
    # safety check: assert a 1:1 mapping between prevIDs and currIDs
    assert_pairs_are_one_to_one(prevIDs, currIDs, m)
            
    return prevIDs, currIDs



def assert_pairs_are_one_to_one(selected_prevIDs: np.ndarray, selected_currIDs: np.ndarray, m: int) -> None:
    """
    :obj: assert that there exist no duplicates currIDs and that there are as many prevIDs as currIDs (come in pairs)
    :param selected_prevIDs: array whose values represent prevIDs in the form of row indices of `pair_dists`
    :param selected_currIDs: array whose values represent currIDs in the form of col indices of `pair_dists`
    :param m: number of detections in current frame
    :predecessor func: `resolve_duplicate_currIDs`
    """

    # sanity check
    there_are_as_many_selected_prevIDs_as_currIDs = (len(selected_prevIDs) == len(selected_currIDs))
    assert there_are_as_many_selected_prevIDs_as_currIDs, f"There are NOT as many selected prevIDs as currIDs: len(selected_prevIDs) != len(selected_currIDs). Numbers are: {len(selected_prevIDs)} != {len(selected_currIDs)}"

    # assert no other duplicates exist
    _, unique_PA_min_PB_counts = np.unique(selected_currIDs, return_counts = True)
    assert np.any(unique_PA_min_PB_counts > 1) == False, f"Duplicates still exist in selected_currIDs: {selected_currIDs}"

    # by this point, we should NOT have more prevIDs than currIDs. We can only have equal or less
    n_prevIDs = len(selected_prevIDs)
    assert n_prevIDs <= m, f"Error! We have {n_prevIDs} > {m} with number of prevIDs and total number of detections in current frame. This should NOT be so"

    
    

# assign tracklets to currIDs without pairs
def assign_new_tracklets_to_currIDs_wo_pairs(n_currIDs: int, 
                                             paired_prevIDs: np.ndarray, 
                                             paired_currIDs: np.ndarray, 
                                             prev_tracklets: np.ndarray, 
                                             global_tracklets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :obj: assign new tracklets to currIDs without a prevID pair
    :purpose: each detection in the current frame must have a tracklet. Thus, if a currID did not match with a prevID in the last detection, we must assing a new and unique tracklet
    :method: find the number of currIDs without pairs and generate new tracklets 
    :param n_currIDs: total number of current IDs (equals total number of detections in current frame)
    :param paired_prevIDs, paired_currIDs: paired IDs
    :param prev_tracklets: contains actual tracklets of previous detections (idx represents tracklets in Weight matrix)
    :param global_tracklets: contains all assigned tracklets throughout the "life" of a video (needed to generate new tracklets). Is modified if new tracklets are generated.
    :dependency assignIDs: function generates new tracklets
    :predecessor func: function `resolve_duplicate_currIDs` was executed before this function 
    --------------------------------------------------------------------------------------------
    :steps:
    * are there current IDs without a prevID pair?
    * how many currIDs need tracklets?
    * generate new tracklets for currIDs without pairs
    * assert there is a unique tracklet for each currID
    """

    # associate each paired prevID with their previous corresponding labels
    reassigned_tracklets = prev_tracklets[paired_prevIDs]

    # are there any currIDs without assigned tracklets?
    n_reassigned_tracklets = len(reassigned_tracklets)
    each_currID_does_NOT_have_a_tracklet_yet = (n_reassigned_tracklets < n_currIDs)

    if each_currID_does_NOT_have_a_tracklet_yet:

        # how many new tracklets do we have to generate?
        n_new_tracklets: int = n_currIDs - n_reassigned_tracklets

        # which currID indices do not have pairs?
        currID_indices = np.arange(n_currIDs).reshape(-1,1).repeat(n_reassigned_tracklets, axis = 1) # shape: (n_currIDs, n_reassigned_tracklets)
        expanded_paired_currIDs = paired_currIDs.reshape(1,-1).repeat(n_currIDs, axis = 0) # shape: (n_currIDs, n_reassigned_tracklets)
        is_currID_paired = currID_indices == expanded_paired_currIDs # shape: (n_currIDs, n_reassigned_tracklets)
        currID_pairs = is_currID_paired.sum(axis = 1) # shape: (n_currIDs)
        is_currID_not_paired = (currID_pairs == 0) # shape: (n_currIDs)
        currID_indices_wo_pairs = np.flatnonzero(is_currID_not_paired) 

        # safety check: assert currID indices without pairs equals number of new tracklets
        assert len(currID_indices_wo_pairs) == n_new_tracklets, f"Error: len(currID_indices_wo_pairs) != n_new_tracklets. Numbers: {len(currID_indices_wo_pairs)} != {n_new_tracklets}"

        # generate new tracklets
        new_tracklets: np.ndarray = assignIDs(global_tracklets, n_new_tracklets)
        # update `global_tracklets` of new tracklets
        global_tracklets = np.concatenate([global_tracklets, new_tracklets])

    else:
        # creeate empty values to keep a consistant output in terms of returned number of arguments
        currID_indices_wo_pairs = np.array([])
        new_tracklets = np.array([])
        n_new_tracklets = 0

    
    # sanity check: assert each currID has an assigned tracklet
    there_are_as_many_tracklets_as_currIDs = (n_reassigned_tracklets + n_new_tracklets == n_currIDs)
    assert there_are_as_many_tracklets_as_currIDs, f"There are NOT as many tracklets as currIDs: n_reassigned_tracklets + n_new_tracklets != m. Numbers are: {n_reassigned_tracklets + n_new_tracklets} != {m}"

    
    return reassigned_tracklets, currID_indices_wo_pairs, global_tracklets, new_tracklets



def make_curr_prev(n_currIDs: int,
                   curr_params: list, 
                   curr_centers: list, 
                   currIDs_with_pairs: np.ndarray, 
                   currIDs_wo_pairs: np.ndarray, 
                   reassigned_tracklets: np.ndarray, 
                   new_tracklets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :obj: make all current parameters previous to get ready to receive the new current parameters from next frame
    :param curr_params: radiuses of currIDs
    :param curr_centers: bbox centers of currIDs
    :param currIDs_with_pairs: currIDs that were re-linked (paired)
    :param currIDs_wo_pairs: currIDs that received new tracklets
    :param reassigned_tracklets: previously existing tracklets from previous frames
    :param new_tracklets: newly created tracklets
    :return prev_tracklets, prev_params, prev_centers: output previous tracklets, circle radiuses, and bbox centers that were previously from currIDs
    """

    # make currIDs prevIDs for next round 
    prev_params, prev_centers = curr_params, curr_centers


    # concatenat currIDs into single array
    all_currIDs = np.concatenate([currIDs_with_pairs, currIDs_wo_pairs])
    
    # argsort all currIDs in an ascending manner
    sorted_currID_indices = np.argsort(all_currIDs)


    # sanity check
    assert len(all_currIDs) == n_currIDs, f"Error: len(all_currIDs) != n_currIDs. They are supposed to be equal! Actual results are: {len(all_currIDs)} == {n_currIDs}"

    # concatenate reassigned and new tracklets into single array
    all_tracklets = np.concatenate([reassigned_tracklets, new_tracklets])
    
    # sort all tracklets in this frame by argsort of `all_currIDs`
    sorted_tracklets_by_currID_indices = all_tracklets[sorted_currID_indices]

    # make all tracklets in this frame become the `prev_tracklets` of the next frame
    prev_tracklets = sorted_tracklets_by_currID_indices

    return prev_tracklets, prev_params, prev_centers 





class spatial_track:
    """
    :obj: generate trackles of each frame given during `forward()` method using spatial data
    """
    def __init__(self):

        self.tracklets = [] # needed to track what tracklets each bbox received
        self.prev_tracklets = None
        self.global_tracklets = None
        self.prev_params = self.prev_centers = None
        

    def forward(self,frame_detections, is_xywh = True) -> None:
        """
        :obj: generate tracklets of `frame_detections`
        :param frame_detections: nx4 array like that contains bbox parameters
        :param is_xywh: do bounding box parameters come in (min_x, min_y, width, height) format?
                        else, they come in (min_x, min_y, max_x, max_y) format
        """
        
        
        
        '''
        Phase I:
        Compute prevID and currID pairs between previous frame and current frame
        '''
        
        
        n_detections = len(frame_detections)

        if (self.prev_tracklets is None) and (n_detections != 0):
            # then initialize the first set of IDs and bbox centers
            self.prev_tracklets = np.arange(n_detections, dtype = np.int32)
            self.global_tracklets = self.prev_tracklets.copy()
            prev_params, prev_centers = bbox2circle(frame_detections, is_coco = is_xywh)
            self.prev_params = prev_params
            self.prev_centers = prev_centers
            self.tracklets.append(self.prev_tracklets.tolist())
            return None


        # transform bboxes to circles
        curr_params, curr_centers = bbox2circle(frame_detections, is_coco = is_xywh)

        # calculate pairwise distances 
        pair_dists = cdist(self.prev_centers, curr_centers)
        n, m = pair_dists.shape

        # calculate association of each prevID
        # Indice of `final_picks_curr` represents prevID while value represents the "top pick" curr ID of each prevID indice
        final_picks_curr: np.ndarray = spatial_bias_assignment(pair_dists) # sorted by prevIDs

        '''
        Phase II:
        make sure pairs are within bounds of prevID radius and afterwards,
        assert there is a 1:1 mapping between prevID and currID pairs
        '''


        # are selected pairs within the radius of the previous IDs?
        selected_prevIDs, selected_currIDs = is_within_radius(pair_dists, final_picks_curr, self.prev_params)

        # are there any duplicate currIDs?
        selected_prevIDs, selected_currIDs = resolve_duplicate_currIDs(pair_dists, selected_prevIDs, selected_currIDs, m)


        # assign new tracklets to currIDs without prevID pairs
        reassigned_tracklets, currID_indices_wo_pairs, global_tracklets, new_tracklets = assign_new_tracklets_to_currIDs_wo_pairs(m, selected_prevIDs, selected_currIDs, self.prev_tracklets, self.global_tracklets)

        '''
        Phase III:
        update all currIDs to be prevIDs for next frame
        '''


        # make all curr parameters previous to setup tracking for next frame
        prev_tracklets, prev_params, prev_centers = make_curr_prev(m, curr_params, curr_centers, selected_currIDs, currID_indices_wo_pairs, reassigned_tracklets, new_tracklets)
        self.prev_tracklets = prev_tracklets
        self.prev_params = prev_params
        self.prev_centers = prev_centers


        # update tracklets with all new `prev_tracklets`
        self.tracklets.append(prev_tracklets.astype(int).tolist())


        assert len(self.prev_tracklets) == m

        return None


    def get_tracklets(self) -> List[list]:
        """
        :obj: once tracklets for each frame of a segment have been generated, 
              return all tracklets calculated during `forward()` method
        """
        return self.tracklets
    
    def get_colors(self) -> List[tuple]:
        """
        :obj: associate a unique and random color to each unique tracklet
        """
        flat_tracklets = []
        [flat_tracklets.extend(tracklet) for tracklet in self.tracklets] # merge List[list] into single list
        n_unique_IDs: int = len(np.unique(flat_tracklets))
        colors = [(randint(0,255), randint(0,255), randint(0,255)) for i in range(n_unique_IDs)] # generate random RGB colors
        return colors

    def reset(self) -> None:
        """
        :obj: restart all parameters needed to generate tracklets
        :purpose: if you have generated tracklets for a segment and want to continue generating tracklets
                  for a different segment, restarting is necessary so that tracking is not influenced by previous segment
        """
        self.tracklets = [] # needed to track what tracklets each bbox received
        self.prev_tracklets = None
        self.global_tracklets = None
        self.prev_params = self.prev_centers = None



def proximity(new_preds: np.ndarray, lost_preds: np.ndarray, is_coco = True, take_max = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    :obj: associate IDs between `lost_preds` and `new_preds` by minimum-weight matching of bboxes
    :param new_preds: bbox predictions of new predictions on current frame
    :param lost_preds: bbox predictions of tracklets that have been "lost"
    :param is_coco: whether bbox format is in COCO-format
    :param take_max: whether to make radiuses equal to max(lenght,width)//2 or min 
    :return selected_prevIDs, selected_newIDs: matched pairs 
    """
    if len(lost_preds) == 0:
        return 
    

    # compute center coordinates of previos bounding box predictions 
    lost_xywhs = lost_preds[:, 2:6] # min_x, min_y, width, height coordinates
    lost_radius, lost_centers = bbox2circle(lost_xywhs, is_coco = is_coco, take_max = take_max)

    # compute center coordinates of previos bounding box predictions 
    new_centers = bbox_centers(new_preds[:, 2:6], is_coco = is_coco)

    # compute pairwise distances between all previous and current IDs
    pair_dists = cdist(lost_centers, new_centers)
    n, m = pair_dists.shape

    final_picks_new: np.ndarray = spatial_bias_assignment(pair_dists) # sorted by prevIDs

    '''
    Phase II:
    make sure pairs are within bounds of prevID radius and afterwards,
    assert there is a 1:1 mapping between prevID and currID pairs
    '''


    # are selected pairs within the radius of the previous IDs?
    selected_prevIDs, selected_newIDs = is_within_radius(pair_dists, final_picks_new, lost_radius)

    # are there any duplicate currIDs?
    selected_prevIDs, selected_newIDs = resolve_duplicate_currIDs(pair_dists, selected_prevIDs, selected_newIDs, m)

    return selected_prevIDs, selected_newIDs



def rm_dead_tracklets(frame_idx: int, lost_preds: np.ndarray, TRACKLET_LIFE: int = 30) -> Optional[np.ndarray]:
    """
    :obj: delete entries in `lost_preds` where its frame start is above the specified `TRACKLET_LIFE`
    :param frame_idx: current index frame
    :param lost_preds: 
    """
    if lost_preds is None:
        return lost_preds
    frame_starts = lost_preds[:, -1]
    time_offsets = frame_idx - frame_starts
    is_dead = time_offsets > TRACKLET_LIFE
    lost_preds = np.delete(lost_preds, is_dead, axis = 0)
    return lost_preds


def rm_found_tracklets(lost_preds, selected_lostIDs):
    # what are the found tracklets?
    found_tracklets = lost_preds[selected_lostIDs, 1]
    # delete re-linked tracklet ID from `lost_preds`
    lost_tracklet_IDs = lost_preds[:, 1]
    is_lost_tracklet = np.in1d(lost_tracklet_IDs, found_tracklets)
    lost_preds = np.delete(lost_preds, is_lost_tracklet, axis = 0)
    return lost_preds


def resolve_ID_multiplicity(preds_perframe: List[np.ndarray], lost_trackID: np.int, curr_trackID, frame_idx: int) -> None:
    """
    :obj: replace all instances of `lost_trackID` with new tracklet for all frames from `frame_idx` to end of video
          only if frame contains `lost_trackID` and `curr_trackID` IDs.
    :param preds_perframe: nx7 array containing the predictions of all frames of a given video
    :param lost_trackID: ID of the tracklet in the "lost" category that will replace all instances of 
                         `curr_trackID` with `lost_trackID` for all frames from `frame_idx` to end.
    :param curr_trackID: ID of the tracklet that will be replaced by `lost_trackID` for all frames from
                         `frame_idx` to end.
    :param frame_idx: index of frame that we are analyzing.
    """
    # compute new tracklet by calculating the maximum existing tracklet and adding 1
    new_trackID: int = max_trackID(preds_perframe) + 1
    n_frames = len(preds_perframe)
    # iterate through all frames from `frame_idx` to `n_frames`
    for curr_frame_idx in range(frame_idx, n_frames):
        # extract prediction from frame idx `curr_frame_idx`
        frame_preds: np.ndarray = preds_perframe[curr_frame_idx] # shape: (n, 7)
        # extract IDs of current frame
        frame_trackIDs: np.ndarray = frame_preds[:, 1] # shape: (n,)

        # are there any tracks containing the `lost_trackID`?
        is_lost_in_frame: np.ndarray = (frame_trackIDs == lost_trackID) # shape: (n,)
        n_tracks_containing_lost: np.int = np.sum(is_lost_in_frame)
        
        # are there any tracks containing the `curr_trackID`?
        is_curr_in_frame: np.ndarray = (frame_trackIDs == curr_trackID) # shape: (n,)
        n_tracks_containing_curr: np.int = np.sum(is_curr_in_frame)

        # if track contains both `lost_trackID` and `curr_trackID`
        if (n_tracks_containing_lost > 0) and (n_tracks_containing_curr > 0):
            
            # safety check: assert only one track contains `lost_trackID` and `curr_trackID`
            assert n_tracks_containing_lost == 1, f"video frame {curr_frame_idx} contains {n_tracks_containing_lost} with the same trackID of {lost_trackID}"
            assert n_tracks_containing_curr == 1, f"video frame {curr_frame_idx} contains {n_tracks_containing_curr} with the same trackID of {curr_trackID}"

            # replace all occurences of `lost_trackID` with `new_trackID`
            frame_preds[is_lost_in_frame, 1] = new_trackID





def modify_ahead(preds_perframe: List[np.ndarray], curr_preds: np.ndarray, lost_preds: np.ndarray, 
                 frame_idx: int, selected_lostIDs: np.ndarray, selected_currIDs: np.ndarray) -> None:
    """
    :obj: modify new tracklets to contain lost tracklet for all frames ahead till tracklet dissappears
    :assmpt 1: modifes all frames from `frame_idx` to end to replace any of them that contain a newly 
               assigned tracklet with their corresponding lost tracklet. This means that if the model
               has the capability to "eventually" recover a newly assigned tracklet to a lost tracklet,
               then the behavior of this module would not be so defined as the replaced tracklets might
               mess up the behavior of an "eventual recovery" of a tracklet ID (or they might align). 
    :param preds_perframe: nx7 array containing the predictions of all frames of a given video
    :param curr_preds: nx7 array containing predictions of all detections that have received new tracklet IDs
    :param lost_preds: nx7 array containing predictions of all tracklet IDs that have been 
                       discontinued in subsequent frames up to threshold
    :param frame_idx: current frame of a video that we are in 
    :param selected_lostIDs: (m,) array containing indices of corresponding `lost_preds` prediction whose 
                             tracklet ID has been re-linked to a newly assinged tracklet ID in `curr_preds`
    :param selected_currIDs: (m,) array containing indices of corresponding `curr_preds` prediction whose 
                             tracklet ID has been re-linked to a lost tracklet ID in `lost_preds`
    ----------------------------------------------
    steps:
    * for each pair of selected lost ID index and selected current (new) ID index:
        *  
    """


    # iterate through each corresponding lost and curr tracklet IDs
    for selected_lostID, selected_currID in zip(selected_lostIDs, selected_currIDs):
        # what is the lost tracklet pred that corresponds with `selected_lostID`?
        lost_pred: np.ndarray = lost_preds[selected_lostID, :] # lost tracklet prediction corresponding to `selected_lostIDs`
        lost_trackID: np.int = lost_pred[1] # lost tracklet ID corresponding to `selected_lostIDs`

        # what is the current tracklet preds that corresponds with `selected_currID`?
        curr_pred: np.ndarray = curr_preds[selected_currID, :] # new tracklet prediction corresponding to `selected_currIDs`
        curr_trackID: np.int = curr_pred[1] # new tracklet ID corresponding to `selected_currIDs`

        # for all frames from `frame_idx` to end, 
        for i in range(len(preds_perframe[frame_idx:])):
            # what are all the predictions associated with frame of `frame_idx + i` indice?
            frame_preds: np.ndarray = preds_perframe[frame_idx + i] # shape: (n, 7)

            # what are all predicted tracklets of frame `frame_idx + i` indice?
            frame_trackIDs: np.ndarray = frame_preds[:, 1] # shape: (n,)

            # do predictions in `frame_idx + i` frame contain the newly assigned `curr_trackID`?
            is_curr_tracklet: np.ndarray = (frame_trackIDs == curr_trackID) # shape: (n,)
            n_tracks_with_curr_trackID: np.int = np.sum(is_curr_tracklet)
            if n_tracks_with_curr_trackID > 0:

                assert n_tracks_with_curr_trackID == 1, f"video frame {frame_idx+i} contains {n_tracks_with_curr_trackID} with the same curr_trackID of {curr_trackID}"

                # is `lost_trackID` already in `frame_trackIDs`?
                if np.sum(frame_trackIDs == lost_trackID):
                    # if we re-link `curr_trackID` with `lost_trackID`, then two different tracks
                    #  with the same `lost_trackID` will exist since this tracklet was re-linked
                    #  by parent model. As such, we will give the `lost_trackID` in `frame_trackIDs`
                    #  a unique tracklet that has not yet been assigned before we relink
                    #  `curr_trackID` as `lost_trackID` as will be done by Spatial Hold
                    resolve_ID_multiplicity(preds_perframe, lost_trackID, curr_trackID, frame_idx + i)

                # replace `curr_tracklet` with `lost_tracklet`
                preds_perframe[frame_idx + i][is_curr_tracklet, 1] = lost_trackID





def assert_unique_tracklets_per_frame(frame_splits: List[np.ndarray]) -> None:
    """
    :obj: assert that each set of annotations belonging to a frame contain a single tracklet ID
    :param frame_splits: contains a unique element per set of annotation belonging to a frame;
                         shape per element: (n, 7) where the 7 columns represent:
                         ["frame", "tracklet", "tl_x", "tl_y", "width", "height", "conf"]
    """

    tracklets_perframe = [frame_pred[:, 1] for frame_pred in frame_splits]
    for i,frame_tracklet in enumerate(tracklets_perframe):
        IDs, counts = np.unique(frame_tracklet, return_counts=True)
        n_IDs = len(IDs)
        assert n_IDs == np.sum(counts), f"Frame {i} contains duplicate tracklets.\nIDs: {IDs}\nCounts: {counts}"






class spatial_hold_v1:
    def __init__(self, frame_splits: List[np.ndarray], TRACKLET_LIFE: int = 120, take_max: bool = False):
        """
        :obj: initiate spatial hold parameters
        :param frame_splits: list containing frame tracklet predictions of a video; each array element is of shape nx7 where 
                             `n` represents number of tracklets in the given frame and `7` stand for ["frame_idx", "tracklet_pred", "tl_x_bbox", "tl_y_bbox", "width", "height", "conf"]
        :param TRACKLET_LIFE: determines the life of lost tracklets in terms of number of frames
        :param take_max: when converting each bbox to a circle, this decides whether radius will equal maximum of bbox dim or minimum 
        """
        self.frame_splits = frame_splits
        self.TRACKLET_LIFE = TRACKLET_LIFE
        self.take_max = take_max
        self.lost_preds: Optional[np.ndarray] = None # if lost predictions exist, this will be an `nx7` array 
        self.glost_preds: List[Optional[np.ndarray]] = [] # global lost preds (stores lost prediction of each annotated frame; if no lost predictions in a frame, `None` will be the element)
    
    def run(self) -> List[np.ndarray]:
        """
        :obj: modify `self.frame_splits` predictions with spatial hold
        """
        
        # init parameters
        prev_preds = curr_preds = None # no predictions yet
        assg_tracklets = [] # keeps track of all assigned tracklet IDs
        
        n_annt_frames = len(self.frame_splits) # number of frames with bbox predictions or annotations

        # iterate through each frame with bboxes
        for i in range(n_annt_frames):

            frame_preds: np.ndarray = self.frame_splits[i]

            # is this our first set of predictions?
            if prev_preds is None:
                prev_preds = frame_preds # make `prev_preds` hold all `nx7` detections of the previous frame
                prevIDs = prev_preds[:, 1].tolist() # extract all track IDs of previous frame
                assg_tracklets.extend(prevIDs) # record tracklets of first frame with bboxes
                self.glost_preds.append(None) # no lost predictions on first frame with bboxes
                continue # continue to next frame

            # has parent model "re-linked" a tracklet ID that has been categorized as "lost" by deep hold?
            self.rm_found_tracklets_from_lost_ctgry(self.lost_preds, frame_preds) # remove found tracklets from lost category
            
            # have any tracklets in `lost_preds` "died"?
            self.lost_preds: Optional[np.ndarray] = rm_dead_tracklets(i, self.lost_preds, self.TRACKLET_LIFE)
                
            # append lost predictions of current frame `i` to `self.glost_preds`
            self.track_curr_lost_preds()

            # what are the the tracklets of the current and previous frame detections?
            currIDs: np.ndarray = frame_preds[:, 1]
            prevIDs: np.ndarray = prev_preds[:, 1]

            # are there any new tracklets in current frame (`currIDs`) that have never appeared before (not in `assg_tracklets`)?
            #   if so, are there any tracklets in `lost_preds`? 
            #       if so, re-link (associate) lost tracklets with new tracklets that have never appeared before
            # NOTE: `frame_preds` points to `self.frame_splits[i]`. Whatever changes to `self.frame_splits[i]` are reflected in `frame_preds`
            currIDs: np.ndarray = self.relink_lost_tracklets(frame_preds, currIDs, assg_tracklets, self.lost_preds, self.take_max, i)


            # are there any new tracklets in `currIDs` that have not appeared before?
            new_tracklets: np.ndarray = np.setdiff1d(currIDs, assg_tracklets)
            assg_tracklets.extend(new_tracklets) # record new tracklets to `assg_tracklets`


            # are there any new lost tracklets from frame 0 to `i`? (if so, append them to `self.lost_preds`)
            self.find_lost_tracklets(self.lost_preds, prevIDs, currIDs, prev_preds, i)

            prev_preds: np.ndarray = frame_preds
            
        # safety checks
        assert_unique_tracklets_per_frame(self.frame_splits)
        assert len(self.frame_splits) == len(self.glost_preds)
        
        
        return self.frame_splits, self.glost_preds 
            
    def rm_found_tracklets_from_lost_ctgry(self, lost_preds: np.ndarray, frame_preds: np.ndarray) -> None:
        """
        :obj: remove found (or re-linked) tracklets by parent model from lost category
        :param lost_preds: predictions in the lost category
        :param frame_preds: predictions in the current frame 
        """

        if lost_preds is None:
            return None

        frame_trackIDs = frame_preds[:, 1]
        lost_trackIDs = lost_preds[:, 1]

        is_not_found = np.in1d(lost_trackIDs, frame_trackIDs, invert = True)

        self.lost_preds = lost_preds[is_not_found, :].copy()
    
    def find_lost_tracklets(self, lost_preds: np.ndarray, 
                            prevIDs: np.ndarray, currIDs: np.ndarray, prev_preds: Optional[np.ndarray], i: int) -> None:
        """
        :obj: find if there are any lost tracklets from previous frame to current frame;
              if so, append new found tracklets to `self.lost_preds`;
              else, do nothing
        :param lost_preds: array containing the lost predictions as of frame `i`
        :param prevIDs: tracklets in previous frame
        :param currIDs: tracklets in current frame
        :param prev_preds: contains predictions of previous frame. If no predictions, this equals `None`
        :param i: indicates current frame of video
        """
        
        # are there any lost tracklets (aka any tracklets in `prevIDs` but not in `currIDs`)?
        is_lost = np.in1d(prevIDs, currIDs, invert = True) 
        if np.sum(is_lost) > 0:
            # what are the "lost" tracklets?
            lost_prev_preds = prev_preds[is_lost, :]
            # add a column to `lost_prev_preds` that indicates frame idx where it was recorded
            n_lostpreds = len(lost_prev_preds)
            frame_idx = np.repeat(i, n_lostpreds).reshape(-1, 1)
            meta_prev_preds = np.concatenate([lost_prev_preds, frame_idx], axis = 1)
            if lost_preds is None:
                self.lost_preds: np.ndarray = meta_prev_preds.copy()
            else:
                self.lost_preds: np.ndarray = np.concatenate([lost_preds, meta_prev_preds], axis = 0)
            
        
        
    def relink_lost_tracklets(self, frame_preds: np.ndarray, currIDs: np.ndarray, 
                              assg_tracklets: list, lost_preds: np.ndarray, 
                              take_max: bool, i: int) -> np.ndarray:
        """
        :obj: are there any new tracklets in frame `i` that actually correspond to a detection in `lost_preds`?
              if so, in-place replace the element with the new tracklet in `self.frame_splits` with the lost tracklet in `lost_preds`
              else, do nothing 
        :param frame_preds: predictions of current frame `i`
        :param currIDs: tracklets of current frame `i`
        :param assg_tracklets: list containing ALL assigned tracklets from frame 0 to `i`
        :param lost_preds: array containing the lost predictions as of frame `i`
        :param take_max: when converting each bbox to a circle, this decides whether radius will equal maximum of bbox dim
        :param i: indicates current frame of video
        """
        
        # are there any new tracklets in current frame (`currIDs`) that have never appeared before (not in `assg_tracklets`)?
        is_newtracklet: np.ndarray = np.in1d(currIDs, assg_tracklets, invert = True) 
        new_fp: np.ndarray = frame_preds[is_newtracklet, :] # new frame preds (preds that have received new tracklets)


        # are there any predictions with new tracklets AND are there any previous tracklets that have been "lost"?
        if len(new_fp) > 0 and lost_preds is not None and len(lost_preds) > 0:
            # are any of the newly assigned tracklets actually from previous tracklets?
            selected_lostIDs, selected_currIDs = proximity(new_fp, lost_preds, take_max = take_max)
            if len(selected_lostIDs) > 0:
                print(f"frame_idx: {i}")
                print(f"current frame predictions b4 re-linking: \n{frame_preds}")
                print(f"IDs of current frame b4 re-linking: \n{currIDs}")
                print(f"current frame predictions with new tracklets: \n{new_fp}")
                print(f"lost_preds b4 removal of found tracklets \n{lost_preds}")
                print(f"selected_lostIDs relative to preds with lost tracklets: {selected_lostIDs}")
                print(f"selected_currIDs relative to preds with new tracklets:  {selected_currIDs}")


                # safety check: ensure each frame contains no duplicate tracklets
                assert_unique_tracklets_per_frame(self.frame_splits)

                # for each pair of lost and current ID that need to be re-linked, apply this re-linking to all contigous subsequent frames
                modify_ahead(self.frame_splits, new_fp, lost_preds, i, selected_lostIDs, selected_currIDs) # modifies elements of `frame_splits` inplace

                print(f"current frame preds AFTER re-linking: \n{frame_preds}")

                # safety check: ensure each frame contains no duplicate tracklets
                assert_unique_tracklets_per_frame(self.frame_splits)
                
                # delete entry from lost_preds
                self.lost_preds = rm_found_tracklets(lost_preds, selected_lostIDs)

                print("lost_preds after removal of found tracklets")
                print(self.lost_preds)
                print("-----------------------")

                # re-update necessary variables since `frame_splits` elements were modified inplace
                # re-compute tracklet IDs of current frame predictions since we modified `frame_preds` inplace during `modify_ahead()` call
                currIDs: np.ndarray = frame_preds[:, 1].copy()
                
        return currIDs
    
    def track_curr_lost_preds(self) -> None:
        """
        :obj: track lost predictions of frame in which this function is invoked in
              if lost predictions exist, append them to `self.glost_preds` (global lost predictions)
              else, nothing
        """
        n_lost_preds = len(self.lost_preds) if self.lost_preds is not None else 0
        is_not_empty = (self.lost_preds is not None and n_lost_preds > 1) # NOTE: if self.lost_preds is an array, it is possible for it to contain zero lost predictions
        self.glost_preds.append(self.lost_preds.copy() if is_not_empty else None)