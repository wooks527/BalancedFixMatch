images_dir="./data/CXR"
# data_dir="./data/CXR/balance_sample"
# data_dir="./data/CXR/imbalance_sample"

epoch=20
fold=4
scheduler='Adam'
lr=0.00005

for num_labeled in 50 25
do
	data_dir="./data/CXR/imbalance_sample/nl$num_labeled"
	for random_seed in 0 1 2 3 # for random seed
	do
		for batch_size in 16 # 4 6 8
		do
			for mu in 1 #2 3
			do
				for lambda_u in 1.0
				do
					for gamma in 0.5 1.0 1.5 2.0
					do
						for threshold in 0.95
						do
							python train.py --purpose fixmatch \
											--images_dir $images_dir \
											--data_dir $data_dir/$mu \
											--fold $fold \
											--epochs $epoch \
											--batch_size $batch_size \
											--num_labeled $num_labeled \
											--mu $mu \
											--lambda_u $lambda_u \
											--threshold $threshold \
											--focal_loss \
											--gamma $gamma \
											--scheduler $scheduler \
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
