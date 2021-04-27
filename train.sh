images_dir="./data/CXR"

epoch=20
fold=4
scheduler='step'
# scheduler='cos'

for num_labeled in 50
do
	data_dir="./data/CXR/imbalance_sample/nl$num_labeled"
	for random_seed in 0 1 2 3
	do
		for batch_size in 16
		do
			for mu in 1 #2 3
			do
                # Baseline
                python train.py --purpose baseline \
                                --images_dir $images_dir \
                                --data_dir $data_dir/$mu \
                                --fold $fold \
                                --epochs $epoch \
                                --batch_size $batch_size \
                                --num_labeled $num_labeled \
                                --mu $mu \
                                --scheduler $scheduler \
                                --random_seed $random_seed \
                                --print_to_file \
                                --metric_types acc ppv recall f1 \
                                --dataset_types train test

				for lambda_u in 1.0
				do
					for gamma in 0.5
					do
						for threshold in 0.95
						do
                            # FixMatch Only
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
                                            --scheduler $scheduler \
                                            --random_seed $random_seed \
                                            --print_to_file \
                                            --metric_types acc ppv recall f1 \
                                            --dataset_types train test

                            # FixMatch with Focal Loss
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

                            # FixMatch with Focal Loss and Sharpening
							for temperature in 0.9 0.5
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
												--sharpening \
												--temperature $temperature \
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
done
