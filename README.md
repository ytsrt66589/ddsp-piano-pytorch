## Run 

1. create ```data```, ```data_cache```, ```data_cache_val``` directory 
2. download [maestro v3.0.0](https://magenta.tensorflow.org/datasets/maestro#v300) dataset
3. put it in the ```data``` directory ( data/maestro-v3.0.0 )
4. run ```bash full_training.sh``` -> start training 

```
Maybe 可以設定一下多久跑一次 validation XD, 不然會等很久才一次 validation
data 部分我不確定有沒有比全部處理成 npy 再 load 還要好的方法，我就單純直接先這樣做了
但這樣 load data 也是要等一段時間

剩下的程式碼編排部分 我可能沒有用得很好 QQ

inference code 還沒寫XD 我都先看 validation 
```

## Training Strategy 

In the original paper, there are 2-stage training and fine-tune training by the following three scripts.
However, only running the first training strategy is good enough.

```
python3 train.py \
	--batch_size $phase_1_batch_size \
	--epochs $phase_1_n_epochs \
	--lr $phase_1_learning_rate \
	--phase 1 \
	$maestro_path \
	$maestro_cache_path \
	$exp_dir
```

```
python3 train.py \
	--batch_size $phase_2_batch_size \
	--epochs $phase_2_n_epochs \
	--lr $phase_2_learning_rate \
	--phase "2" \
	--restore "$exp_dir/phase_1/ckpts/" \
	$maestro_path \
	$maestro_cache_path \
	$exp_dir
```

```
python3 train.py \
	--batch_size $phase_3_batch_size \
	--epochs $phase_3_n_epochs \
	--lr $phase_3_learning_rate \
	--phase "3" \
	--restore "$exp_dir/phase_2/ckpts/" \
	$maestro_path \
	$maestro_cache_path \
	$exp_dir
```
