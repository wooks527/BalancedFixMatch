images_dir="/mnt/f/CXR"

epoch=20
fold=4
scheduler='cosine'

for num_labeled in 100 150
do
	data_dir="./data/CXR/imbalance_sample/nl$num_labeled"
	for random_seed in 0 1 2 3 # for random seed
	do
		for batch_size in 16 # 4 6 8
		do
			# Baseline
			mu=1
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
							--print_to_file \
							--metric_types acc ppv recall f1 \
							--dataset_types train test
			for mu in 1 2 3
			do
				for lambda_u in 1.0 #0.75 0.5
				do
					for gamma in 0.5 1.0 1.5 #1.0 #2.0
					do
						for threshold in 0.95 #0.99 0.93 0.91
						do
							# FixMatch
							python train.py --purpose fixmatch \
											--images_dir $images_dir \
											--data_dir $data_dir/$mu \
											--num_labeled $num_labeled \
											--scheduler $scheduler \
											--mu $mu \
											--focal_loss \
											--gamma $gamma \
											--lambda_u $lambda_u \
											--threshold $threshold \
											--fold $fold \
											--epochs $epoch \
											--batch_size $batch_size \
											--random_seed $random_seed \
											--print_to_file \
											--metric_types acc ppv recall f1 \
											--dataset_types train test
						done
					done
				done
			done
		done
	done
done
