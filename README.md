## Introduction 

This is the project of implementing DaFX20in22 paper: DDSP-Piano: Differentiable Piano model for MIDI-to-Audio Performance Synthesis by the Pytorch framework.

### Official information 
* Official implementation is by TensorFlow: [original implementation](https://github.com/lrenault/ddsp-piano) 
* Official paper link: [paper link](https://dafx2020.mdw.ac.at/proceedings/papers/DAFx20in22_paper_48.pdf) 

### Future To-Do 
- [?] Inference.py 
- [?] Better data organization 
- [?] Better code organization  


## Run 

1. create ```data```, ```data_cache```, ```data_cache_val``` directory 
2. download [maestro v3.0.0](https://magenta.tensorflow.org/datasets/maestro#v300) dataset
3. put it in the ```data``` directory ( data/maestro-v3.0.0 )
4. run ```bash full_training.sh``` -> start training 

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
