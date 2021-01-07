images_dir="./data/CXR"
data_dir="./data/CXR"

epoch=40
num_labeled=100
fold=3
scheduler='step'

for lr in 0.001
do
	# for random_seed in {0..2} # for random seed
	for random_seed in 0 1 2 # for random seed
	do
		for batch_size in 16
		do
			for mu in 1 #2 3
			do
				# for baseline_flag in {0..20}
				# for baseline_flag in "16,9" "16,9,3"
				# for baseline_flag in "13,9" "9,13" "13,9,7"
				for baseline_flag in "9,14" "14,9" "9,14,3" "9,3,14" "14,9,3" "14,3,9" "3,9,14" "3,14,9"
				do
					# Baseline
					python train.py --purpose baseline \
									--images_dir $images_dir \
									--data_dir $data_dir/$mu \
									--num_labeled $num_labeled \
									--lr $lr \
									--scheduler $scheduler \
									--mu $mu \
									--lambda_u 1.0 \
									--threshold 0.95 \
									--fold $fold \
									--epochs $epoch \
									--batch_size $batch_size \
									--random_seed $random_seed \
									--baseline_flag $baseline_flag \
									--print_to_file \
									--metric_types acc ppv recall f1 \
									--dataset_types train test
				done

				# for lambda_u in 1.0 #0.75 0.5
				# do
				# 	for threshold in 0.95
				# 	do
				# 		for purpose in "fixaug1" "fixaug2" "fixaug3"
				# 		do
				# 			# FixMatch
				# 			python train.py --purpose $purpose \
				# 							--images_dir $images_dir \
				# 							--data_dir $data_dir/$mu \
				# 							--num_labeled $num_labeled \
				# 							--lr $lr \
				# 							--scheduler $scheduler \
				# 							--mu $mu \
				# 							--lambda_u $lambda_u \
				# 							--threshold $threshold \
				# 							--fold $fold \
				# 							--epochs $epoch \
				# 							--batch_size $batch_size \
				# 							--random_seed $random_seed \
				# 							--print_to_file \
				# 							--metric_types acc ppv recall f1 \
				# 							--dataset_types train test
				# 		done
				# 	done
				# done
			done
		done
	done
done
