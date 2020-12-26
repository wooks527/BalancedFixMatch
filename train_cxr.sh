images_dir="/mnt/f/CXR"
data_dir="./data/CXR"

epoch=50
num_labeled=25
fold=5
scheduler='step'

# for random_seed in {0..2} # for random seed
for random_seed in 0 1 2 # for random seed
do
	for batch_size in 4 6 8
	do
	    for mu in 1 2 3
	    do
        	# Baseline
	        python train.py --purpose baseline \
	                        --images_dir $images_dir \
        	                --data_dir $data_dir/$mu \
                	        --num_labeled $num_labeled \
                        	--scheduler $scheduler \
	                        --mu $mu \
	                        --lambda_u 1.0 \
	                        --threshold 0.95 \
	                        --fold $fold \
	                        --epochs $epoch \
	                        --batch_size $batch_size \
	                        --random_seed $random_seed \
	                        --is_finetuning \
	                        --print_to_file \
	                        --metric_types acc ppv recall f1 \
	                        --dataset_types train test
	
	        for lambda_u in 1.0 0.75 0.5
	        do
	            for threshold in 0.95
	            do
	                # FixMatch
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
	            done
	        done
	    done
	done
done