# python train.py \
#       --config-file configs/finetune/pvt.yaml \
#        DATA.PERCENTAGE '0.4' \
#       DATA.BATCH_SIZE "128" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Finetune_OOD/pvt_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Finetune_OOD/pvt_4/sup_vitb16_imagenet21k/lr0.0001_wd0.01/run1/val_pvt_best_model.pth"
#
#
#
# python train.py \
#       --config-file configs/finetune/pvt.yaml \
#       DATA.BATCH_SIZE "128" \
#       DATA.PERCENTAGE '0.4' \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       MODEL.ADAPTER.REDUCATION_FACTOR "128" \
#       MODEL.TRANSFER_TYPE "adapter" \
#       OUTPUT_DIR "./Adapter_OOD/pvt_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Adapter_OOD/pvt_4/sup_vitb16_imagenet21k/lr0.1_wd0.001/run1/val_pvt_best_model.pth"
#


python train.py \
      --train-type "prompt" \
      --config-file configs/prompt/pvt.yaml \
      DATA.PERCENTAGE '0.4' \
      DATA.BATCH_SIZE "128" \
      SOLVER.BASE_LR "0.0" \
      SOLVER.WEIGHT_DECAY "0.0" \
      MODEL.PROMPT.DROPOUT "0.1" \
      OUTPUT_DIR "./Prompt_OOD/pvt_4/" \
      SOLVER.TOTAL_EPOCH '0' \
      MODEL.WEIGHT_PATH "./Prompt_OOD/pvt_4/sup_vitb16_imagenet21k/lr1.0_wd0.001/run1/val_pvt_best_model.pth"





# python train.py \
#       --config-file configs/linear/pvt.yaml \
#       DATA.PERCENTAGE '0.1' \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvt_1/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/pvt_1/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_pvt_best_model.pth"

# python train.py \
#       --config-file configs/linear/pvt.yaml \
#       DATA.PERCENTAGE '0.2' \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvt_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/pvt_2/sup_vitb16_imagenet21k/lr2.5_wd0.0001/run1/val_pvt_best_model.pth"

# python train.py \
#       --config-file configs/linear/pvt.yaml \
#       DATA.PERCENTAGE '0.3' \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvt_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/pvt_3/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_pvt_best_model.pth"

# python train.py \
#       --config-file configs/linear/pvt.yaml \
#       DATA.PERCENTAGE '0.4' \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvt_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/pvt_4/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_pvt_best_model.pth"

# python train.py \
#       --config-file configs/linear/pvt.yaml \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvt/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/pvt/sup_vitb16_imagenet21k/lr2.5_wd0.0001/run1/val_pvt_best_model.pth"

# python train.py \
#       --config-file configs/linear/herbarium_19.yaml \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/herbarium_19/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/herbarium_19/sup_vitb16_imagenet21k/lr2.5_wd0.0001/run1/val_herbarium_19_best_model.pth"







# #########linear tuning ###########
# python train.py \
#       --config-file configs/linear/cotton.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/cotton_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/cotton_2/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_cotton_best_model.pth"
#
#
# python train.py \
#       --config-file configs/linear/cotton.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/cotton_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/cotton_3/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_cotton_best_model.pth"
#
#
#
# python train.py \
#       --config-file configs/linear/cotton.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/cotton_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/cotton_4/sup_vitb16_imagenet21k/lr2.5_wd0.001/run1/val_cotton_best_model.pth"
#
#
#
# ######mango #####
#
# python train.py \
#       --config-file configs/linear/mango.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/mango_2/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_mango_best_model.pth"
#
#
# python train.py \
#       --config-file configs/linear/mango.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/mango_3/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_mango_best_model.pth"
#
#
#
# python train.py \
#       --config-file configs/linear/mango.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/mango_4/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_mango_best_model.pth"
#
#
#
# python train.py \
#       --config-file configs/linear/mango.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_5/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/mango_5/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_mango_best_model.pth"
#
#
#
# python train.py \
#       --config-file configs/linear/mango.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/mango_6/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/mango_6/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_mango_best_model.pth"
#
#
# ###strawberry ####
# python train.py \
#       --config-file configs/linear/strawberry.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/strawberry_2/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_strawberry_best_model.pth"
#
#
#
# python train.py \
#       --config-file configs/linear/strawberry.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/strawberry_3/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_strawberry_best_model.pth"
#
#
#
# python train.py \
#       --config-file configs/linear/strawberry.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/strawberry_4/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_strawberry_best_model.pth"
#
#
#
# python train.py \
#       --config-file configs/linear/strawberry.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_5/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/strawberry_5/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_strawberry_best_model.pth"
#
#
#
# python train.py \
#       --config-file configs/linear/strawberry.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/strawberry_6/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/strawberry_6/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_strawberry_best_model.pth"
#
#
#
# #####PVTC #####
# python train.py \
#       --config-file configs/linear/pvtc.yaml \
#       DATA.PERCENTAGE '0.2' \
#       DATA.NUMBER_CLASSES "2" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/pvtc_2/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_pvtc_best_model.pth"
#
#
# python train.py \
#       --config-file configs/linear/pvtc.yaml \
#       DATA.PERCENTAGE '0.3' \
#       DATA.NUMBER_CLASSES "3" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/pvtc_3/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_pvtc_best_model.pth"
#
#
#
# python train.py \
#       --config-file configs/linear/pvtc.yaml \
#       DATA.PERCENTAGE '0.4' \
#       DATA.NUMBER_CLASSES "4" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/pvtc_4/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_pvtc_best_model.pth"
#
#
# python train.py \
#       --config-file configs/linear/pvtc.yaml \
#       DATA.PERCENTAGE '0.5' \
#       DATA.NUMBER_CLASSES "5" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_5/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/pvtc_5/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_pvtc_best_model.pth"
#
#
# python train.py \
#       --config-file configs/linear/pvtc.yaml \
#       DATA.PERCENTAGE '0.6' \
#       DATA.NUMBER_CLASSES "6" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_6/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/pvtc_6/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_pvtc_best_model.pth"
#
#
# python train.py \
#       --config-file configs/linear/pvtc.yaml \
#       DATA.PERCENTAGE '0.7' \
#       DATA.NUMBER_CLASSES "7" \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/pvtc_7/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/pvtc_7/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_pvtc_best_model.pth"
#
#
# python train.py \
#       --config-file configs/linear/plant_village.yaml \
#       DATA.PERCENTAGE '0.1' \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/plant_village_1/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/plant_village_1/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_plant_village_best_model.pth"
#
#
# python train.py \
#       --config-file configs/linear/plant_village.yaml \
#       DATA.PERCENTAGE '0.2' \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/plant_village_2/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/plant_village_2/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_plant_village_best_model.pth"
#
#
# python train.py \
#       --config-file configs/linear/plant_village.yaml \
#       DATA.PERCENTAGE '0.3' \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/plant_village_3/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/plant_village_3/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_plant_village_best_model.pth"
#
#
# python train.py \
#       --config-file configs/linear/plant_village.yaml \
#       DATA.PERCENTAGE '0.4' \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/plant_village_4/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/plant_village_4/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_plant_village_best_model.pth"
#
#
#
#
# python train.py \
#       --config-file configs/linear/plant_village.yaml \
#       SOLVER.BASE_LR "0.0" \
#       SOLVER.WEIGHT_DECAY "0.0" \
#       DATA.FEATURE "sup_vitb16_imagenet21k" \
#       MODEL.MODEL_ROOT "models/" \
#       OUTPUT_DIR "./Linear_OOD/plant_village_all/" \
#       SOLVER.TOTAL_EPOCH '0' \
#       MODEL.WEIGHT_PATH "./Linear_OOD/plant_village_all/sup_vitb16_imagenet21k/lr5.0_wd0.001/run1/val_plant_village_best_model.pth"

