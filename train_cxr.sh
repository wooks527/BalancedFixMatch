images_dir="/media/aiffel0042/SSD256/temp/AVIDNet/data/CXR/ori"
data_dir="./data/CXR"

epoch=20
batch_size=8
num_labeled=25
fold=2
scheduler='step'

# for random_seed in {0..2} # for random seed
for random_seed in 0 # for random seed
do
    for mu in 1
    do
        for lambda_u in 1.0 #0.75 0.5
        do
            for threshold in 0.95 #0.98
            do
                python train.py --purpose fixmatch \
                                --images_dir $images_dir \
                                --data_dir $data_dir/$mu \
                                --num_labeled $num_labeled \
                                --scheduler $scheduler \
                                --mu $mu \
                                --lambda_u $lambda_u \
                                --threshold $threshold \
                                --fold $fold \
                                --epochs $epoch \
                                --batch_size $batch_size \
                                --random_seed $random_seed \
                                --is_finetuning \
                                --print_to_file \
                                --metric_types acc ppv recall f1 \
                                --dataset_types train test

                python train.py --purpose baseline \
                                --images_dir $images_dir \
                                --data_dir $data_dir/$mu \
                                --num_labeled $num_labeled \
                                --scheduler $scheduler \
                                --mu $mu \
                                --lambda_u $lambda_u \
                                --threshold $threshold \
                                --fold $fold \
                                --epochs $epoch \
                                --batch_size $batch_size \
                                --random_seed $random_seed \
                                --is_finetuning \
                                --print_to_file \
                                --metric_types acc ppv recall f1 \
                                --dataset_types train test
            done
        done
    done
done