# training_visualization
A progress for visualizing large model training with visdom

安装visdom (https://github.com/fossasia/visdom)
'''{shell}
pip install visdom
python -m visdom.server >/dev/null 2>&1 &     # 8097端口
'''

通过log_file读取训练信息，需要修改 `npu_num` 和 `log_file_name`

可以实时生成Loss曲线、表格（Loss、每个iteration消耗时间、TGS）
