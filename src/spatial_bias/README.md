The `utils` directory serves to store python objects (functions and classes) that are commonly needed to implement spatial tracking or to annotate. 





**Bounding box-related functions:**

* `bbox_centers()`

  * ```python
    def bbox_centers(bboxes: Union[List[List], np.ndarray], is_coco = False, is_list = True) -> Union[List[List[int]], np.ndarray]:
        """
        :obj: attain center coordinates for each bounding box in `bboxes`
        :param bboxes: nx4 array-like object  
        :param is_coco: if False, bboxes come in [min_x, min_y, max_x, max_y] format. Else, [min_x, min_y, width, length]
        :param is_list: if False, I/O is a numpy array. Else, I/O is a list
        :return centers: nx2 array-like where features represent center coordinates of bboxes in (mid_x, mid_y) format
        ----------------------------------------------------------------------------
        steps:
        * iterate through each bbox:
            * Unpack the bbox coordinates
            * store coordinates in `centers`
        * return `centers`
        """
    ```

* `bbox2circle()`

  * ```python
    def bbox2circle(bboxes: Union[List[List], np.ndarray], is_coco = False, take_max = True, is_list = True) -> Union[Tuple[List], Tuple[np.ndarray]]:
        """
    	:obj: transform each bbox in `bboxes` to circles by returning the circle radius and circle centers
        :method: make circumference of circle equal the max(length, width)
        :param bboxes: each element contains 4 integers representing the bbox in format [min_x, min_y, max_x, max_y] if is_coco == False, else [min_x, min_y, height, width]
        :param is_coco: if False, bboxes come in [min_x, min_y, max_x, max_y] format. Else, [min_x, min_y, width, length]
        :param take_max: if False, make circumeference of each bbox equal min(length, width) dimension of bbox. Else, take max
        :param is_list: if False, Input is a numpy array. Else, Input is a list
        :return circle_params: each tuple element contains circle parameters in the form of (radius, center_coordinate)
        ----------------------------------------------------------------------------
        steps:
        * iterate through each bbox:
            * Unpack the bbox coordinates
            * calculate radius and center coordinates
            * store circle parameters
        * return `circle_params`
        """
    ```

* `annotate_tracklets()`

  * ```python
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
    ```

* `are_tracklets_continuous()`

  * ```python
    def are_tracklets_continuous(curr_tracklets: list, continuity: dict) -> None:
        """
        :obj: did any of the current tracklets appear in previous frame?
        :purpose: this is to know whether to draw a line of past coordinates of current tracklets if they have continuity
        :method: the keys in `continuity` represent tracklets that have had continuity up to the previous frame. If any of the past tracklet keys are not in the current tracklets, we delete them as they no longer have continuity
        :param curr_tracklets: list of shape (n) where `n` represents the number of tracklets associated with each detection in current frame
        :param continuity: tracks the center coordinates of all recurrent tracklets (tracklets that have continuously appeared in previous and current frame) (modified inplace if a tracklet does not have continuity)
        :return: None, modifies `continuity` inplace if a tracklet no longer has continuity
        """
    ```

  * 



**HeadHunter-related code**

* `coerce_preds()`

  * ```python
    def coerce_preds(preds: List[dict]) -> List[dict]: 
        """
        :obj: coerce HeadHunter preds from torch tensors to list objects
        :param preds: single-item list that contains dictionary with predictions as torch objects
        """
    ```

* `draw_bboxes()`

  * ```python
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
    ```

* 



**COCO-related code:**

* `skipped_frames()`

  * ```python
    def skipped_frames(annts: list) -> list:
        """
        :obj: are there any frames that did not contain any annotations?
        :param annts: computed by accesssing `coco["annotations"]` section of coco annotations
        :property: if a frame was skipped, then the `frame_id`s in `annts` will NOT be linear
        :return global_skipped_IDs: returns the `image_id`s that were skipped
                                    NOTE: `image_id`s start at idx 1 (NOT zero)
        """
    ```

  * 



**Spatial Hold-related code:**

* `resolve_conflict()`

  * ```python
    def resolve_conflict(draft_picks_curr: np.ndarray, currID_indice_duplicate: int, conflicted_prevIDs: np.ndarray) -> np.int:
        """
        :obj: if >=2 prevIDs are fithing over a currID, resolve conflict by choosing the pair that minimizes distance w.r.t. currID
        :param draft_picks_curr: created through `np.argsort(pair_dists, axis = 0)`
        :param currID_indice_duplicate: column indice that is getting "fought" over by >=2 prev indices
        :param conflicted_prevIDs: prevIDs that are "fighting" over `currID_indice_duplicate`
        :return conf_winner: `conf_winner` contains the row indice of prevID that keeps the currID pair 
        """
    ```

* `associate()`

  * ```python
    def associate(pair_dists: np.ndarray) -> np.ndarray:
        """
        :obj: implement minimum weight matching between the pairwise distances of the previous and current bboxes
        :param pair_dists: pairwise weight matrix between previous and current bboxes
        :return top_curr_indice_pick_per_prevID: 1d array where indices represent prevIDs and 
                                                 values represent their currID indice pairings
        """
    ```

* `is_within_radius()`

  * ```python
    # which are pairs are within the radius of their associated previus ID?
    def is_within_radius(pair_dists: np.ndarray, 
                         top_curr_indice_pick_per_prevID: np.ndarray, 
                         prev_params: Union[list, np.ndarray]) -> Tuple[np.ndarray]:
        """
        :obj: if `pair_dists[prevID, currID] < radius_of_prevID`, keep pair; else, remove pair
        :param pair_dists: pairwise distance matrix
        :param top_curr_indice_pick_per_prevID: indice represents prevID while value represents col idx of `pair_dists`. Together, pairing is defined
        :param prev_params: radiuses of each prevID
        :return selected_prevIDs, selected_currIDs: both arrays contain indices of IDs whose distance is < radius_detection_of_prevID
        """
    ```

* `resolve_duplicate_currIDs()`

  * ```python
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
    ```

* `assert_pairs_are_one_to_one()`

  * ```python
    def assert_pairs_are_one_to_one(selected_prevIDs: np.ndarray, selected_currIDs: np.ndarray, m: int) -> None:
        """
        :obj: assert that there exist no duplicates currIDs and that there are as many prevIDs as currIDs (come in pairs)
        :param selected_prevIDs: array whose values represent prevIDs in the form of row indices of `pair_dists`
        :param selected_currIDs: array whose values represent currIDs in the form of col indices of `pair_dists`
        :param m: number of detections in current frame
        :predecessor func: `resolve_duplicate_currIDs`
        """
    ```

* `assign_new_tracklets_to_currIDs_wo_pairs()`

  * ```python
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
    ```

* `make_curr_prev()`

  * ```python
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
    ```

* `spatial_track()`

  * ```python
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
    ```

* `proximity()`

  * ```python
    def proximity(new_preds: np.ndarray, lost_preds: np.ndarray, is_coco = True, take_max = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        :obj: associate IDs between `lost_preds` and `new_preds` by minimum-weight matching of bboxes
        :param new_preds: bbox predictions of new predictions on current frame
        :param lost_preds: bbox predictions of tracklets that have been "lost"
        :param is_coco: whether bbox format is in COCO-format
        :param take_max: whether to make radiuses equal to max(lenght,width)//2 or min 
        :return selected_prevIDs, selected_newIDs: matched pairs 
        """
    ```

* `def rm_dead_tracklets()`

  * ```python
    def rm_dead_tracklets(frame_idx: int, lost_preds: np.ndarray, TRACKLET_LIFE: int = 30) -> Optional[np.ndarray]:
        """
        :obj: delete entries in `lost_preds` where its frame start is above the specified `TRACKLET_LIFE`
        :param frame_idx: current index frame
        :param lost_preds: bboxes that are categorized as "lost" and will be judged whether they have been around longer than `TRACKLET_LIFE`
        :param TRACKLET_LIFE: maximum life of each tracklet
        :return lost_preds: returns `lost_preds` with deleted entry if life > TRACKLET_LIFE or None
        """
    ```

* `def rm_found_tracklets()`

  * ```python
    def rm_found_tracklets(lost_preds: np.ndarray, selected_lostIDs: np.ndarray) -> np.ndarray:
        """
        :obj: if tracklet that was categorized as "lost" has been re-linked, remove it from "lost" category
        :param lost_preds: predictions that are categorized as "lost"
        :param selected_lostIDs: arrays whose indices indices the row indices of `lost_preds` of preds that have been "found"
        :return lost_preds: modified `lost_preds` with deleted entries of tracklets that have been "found"
        """
    ```

* `def modify_ahead()`

  * ```python
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
    ```

* `def assert_unique_tracklets_per_frame()`

  * ```python
    def assert_unique_tracklets_per_frame(frame_splits: List[np.ndarray]) -> None:
        """
        :obj: assert that each set of annotations belonging to a frame contain a single tracklet ID
        :param frame_splits: contains a unique element per set of annotation belonging to a frame;
                             shape per element: (n, 7) where the 7 columns represent:
                             ["frame", "tracklet", "tl_x", "tl_y", "width", "height", "conf"]
        """
    ```

  * 