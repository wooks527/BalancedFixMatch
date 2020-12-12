for random_seed in {0..2} # for random seed
do
    for mu in 6 4 2
    do
        python train.py --purpose baseline \
                        --images_dir /mnt/e/00.dataset/CT \
                        --data_dir ./data/CT/$mu \
                        --num_labeled 25 \
                        --mu $mu \
                        --fold 5 \
                        --epochs 20 \
                        --batch_size 8 \
                        --random_seed $random_seed \
                        --print_to_file \
                        --metric_types acc ppv recall f1 \
                        --dataset_types train val

        python train.py --purpose fixmatch \
                        --images_dir /mnt/e/00.dataset/CT \
                        --data_dir ./data/CT/$mu \
                        --num_labeled 25 \
                        --mu $mu \
                        --fold 5 \
                        --epochs 20 \
                        --batch_size 8 \
                        --random_seed $random_seed \
                        --print_to_file \
                        --metric_types acc ppv recall f1 \
                        --dataset_types train val
    done
done

# # completed
# python train.py --purpose baseline \
#                 --data_dir /media/aiffel0042/SSD256/temp/AVIDNet/data/CT \
#                 --num_labeled 25 \
#                 --mu 2 \
#                 --fold 5 \
#                 --epochs 20 \
#                 --batch_size 16 \
#                 --random_seed 0 \
#                 --overwrite \
#                 --print_to_file \
#                 --metric_types acc ppv recall f1 \
#                 --dataset_types train test

# # to-do list
# python train.py --purpose baseline \
#                 --data_dir /media/aiffel0042/SSD256/temp/AVIDNet/data/CT \
#                 --num_labeled 25 \
#                 --mu 4 \
#                 --fold 5 \
#                 --epochs 20 \
#                 --batch_size 16 \
#                 --random_seed 0 \
#                 --overwrite \
#                 --print_to_file \
#                 --metric_types acc ppv recall f1 \
#                 --dataset_types train test

# python train.py --purpose baseline \
#                 --data_dir /media/aiffel0042/SSD256/temp/AVIDNet/data/CT \
#                 --num_labeled 25 \
#                 --mu 8 \
#                 --fold 5 \
#                 --epochs 20 \
#                 --batch_size 16 \
#                 --random_seed 0 \
#                 --overwrite \
#                 --print_to_file \
#                 --metric_types acc ppv recall f1 \
#                 --dataset_types train test