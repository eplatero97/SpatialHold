# bash ./spatial_hold_runs.sh > spatial_hold_runs_logs.txt


# track_spatialhold_embed_hungarian
CONFIG_FILE=configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-public-half.py
WORK_DIR=/media/erick/0123-4567/Github/mmlab/mmtracking/MOT17Logs/Public/track_spatialhold_embed_hungarian/
SHOW_DIR=/media/erick/0123-4567/Github/mmlab/mmtracking/MOT17Logs/Public/track_spatialhold_embed_hungarian/vizs
python tools/test.py $CONFIG_FILE --eval track bbox --work-dir=$WORK_DIR --show-dir=$SHOW_DIR
cd $WORK_DIR
rm ./vizs/*/img1/*.jpg
cd -


# track_hungarian_embed_spatialhold
CONFIG_FILE=configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-public-half_embed_spatialhold.py
WORK_DIR=/media/erick/0123-4567/Github/mmlab/mmtracking/MOT17Logs/Public/track_hungarian_embed_spatialhold/
SHOW_DIR=/media/erick/0123-4567/Github/mmlab/mmtracking/MOT17Logs/Public/track_hungarian_embed_spatialhold/vizs
python tools/test.py $CONFIG_FILE --eval track bbox --work-dir=$WORK_DIR --show-dir=$SHOW_DIR
cd $WORK_DIR
rm ./vizs/*/img1/*.jpg
cd -



# track_embed_spatialhold
CONFIG_FILE=configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-public-half_track_embed_spatialhold.py
WORK_DIR=/media/erick/0123-4567/Github/mmlab/mmtracking/MOT17Logs/Public/track_embed_spatialhold/
SHOW_DIR=/media/erick/0123-4567/Github/mmlab/mmtracking/MOT17Logs/Public/track_embed_spatialhold/vizs
python tools/test.py $CONFIG_FILE --eval track bbox --work-dir=$WORK_DIR --show-dir=$SHOW_DIR
cd $WORK_DIR
rm ./vizs/*/img1/*.jpg
cd -
