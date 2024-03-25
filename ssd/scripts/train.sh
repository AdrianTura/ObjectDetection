# Train mb1-ssd-lite with batch size 1
python train_ssd.py \
    --dataset_type fddb \
    --datasets ~/datasets/FDDB/ \
    --validation_dataset ~/datasets/FDDB/ \
    --net mb1-ssd-lite\
    --batch_size 4 \
    --num_epochs 200 \
    --scheduler "multi-step" \
    --milestones "120,160" \
    --out_folder results

# Train mb1-ssd-lite with batch size 2
python train_ssd.py \
    --dataset_type fddb \
    --datasets ~/datasets/FDDB/ \
    --validation_dataset ~/datasets/FDDB/ \
    --net mb1-ssd-lite\
    --batch_size 8 \
    --num_epochs 200 \
    --scheduler "multi-step" \
    --milestones "120,160" \
    --out_folder results

# conda create -y -n train_env python=3.6 -c conda-forge
# conda activate train_env
# pip install torch