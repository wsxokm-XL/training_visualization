# training_visualization
A progress for visualizing large model training with visdom

安装visdom (https://github.com/fossasia/visdom)

```{shell}
pip install visdom
python -m visdom.server >/dev/null 2>&1 &     # 8097端口
```

通过log_file读取训练信息，需要修改 `npu_num` 和 `log_file_name`

可以实时生成Loss曲线、表格（Loss、每个iteration消耗时间、TGS）
<img width="878" alt="visdom" src="https://github.com/user-attachments/assets/2c68bb29-136f-4829-a4e6-82a7fc6100b9">

日志文件格式样式参考：
```
------------------------ arguments ------------------------
  train_iters ..................................... 5000
  global_batch_size ............................... 16
  seq_length ...................................... 4096
-------------------- end of arguments ---------------------
 iteration        2/    5000 | consumed samples:           32 | elapsed time per iteration (ms): 54568.1 | learning rate: 2.500E-08 | global batch size:    16 | lm loss: 1.519366E+00 | loss scale: 1.0 | grad norm: 41.408 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration        3/    5000 | consumed samples:           48 | elapsed time per iteration (ms): 8493.8 | learning rate: 5.000E-08 | global batch size:    16 | lm loss: 1.502081E+00 | loss scale: 1.0 | grad norm: 40.992 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration        4/    5000 | consumed samples:           64 | elapsed time per iteration (ms): 2878.8 | learning rate: 7.500E-08 | global batch size:    16 | lm loss: 1.499536E+00 | loss scale: 1.0 | grad norm: 39.024 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration        5/    5000 | consumed samples:           80 | elapsed time per iteration (ms): 2881.4 | learning rate: 1.000E-07 | global batch size:    16 | lm loss: 1.527644E+00 | loss scale: 1.0 | grad norm: 41.789 | number of skipped iterations:   0 | number of nan iterations:   0 |
 iteration        6/    5000 | consumed samples:           96 | elapsed time per iteration (ms): 2882.2 | learning rate: 1.250E-07 | global batch size:    16 | lm loss: 1.532800E+00 | loss scale: 1.0 | grad norm: 43.485 | number of skipped iterations:   0 | number of nan iterations:   0 |
```
