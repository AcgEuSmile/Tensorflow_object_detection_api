# Tensorboard_object_detection_api

## Dataset preparing

![flow chart](https://chtseng.files.wordpress.com/2019/02/6340_ynevl46ceg.png?w=760&zoom=2)

1. Label xml->csv using 
    * Set config file "workspace/cfg/xml2csv_config.json"
    * Run "workspace/src/xml_to_csv.py"
2. Generate .record
    * Set config file "workspace/generate_tfrecord_config.json"
    * Run "workspace/src/generate_tfrecord.py"
3. Build your pbtxt, follow the style as below
```
item{
  id: 1
  name: 'bike'
}
item{
  id: 2
  name: 'bus'
}
item{
  id: 3
  name: 'car'
}
item{
  id: 4
  name: 'motor'
}
item{
  id: 5
  name: 'person'
}
item{
  id: 6
  name: 'rider'
}
item{
  id: 7
  name: 'truck'
}
```

## Train your model

1. Download the fine-tune model [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

2. Download the config file [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)

3. Modify the config file
```
# 分類數量，與pbtxt檔內的id最大值一樣。
num_classes: 7
# 第二階段的batch大小
second_stage_batch_size: 4
# fine-tune model的路徑
fine_tune_checkpoint: <path_of_fine_tune_model>
# 輸入的batch大小
batch_size: 4
# .record的路徑
tf_record_input_reader {
    input_path: <path_of_train/eval.record>
}
# pbtxt的路徑
label_map_path: <path_of_pbtxt>
```

4. Edit the script file "/workspace/scripts/train.bash" and run it

```bash
# 設定檔路徑
--pipeline_checkpoint_prefix <config_path>
# 指定GPU使用哪些
CUDA_VISIBLE_DEVICES=0,1
# 訓練的檔案
python <path_of_train.py>
# 輸出位置與偵錯
--train_dir=<output_path> 2>&1 | tee logs/train_$now.txt
```

5. Use tensorboard to observe the model

```bash
tensorboard --logdir=<output_path>
```

# Use this model !!!

1. Convert model.ckpt to inference.pb using "/workspace/scripts/export_model.bash"
```bash
# 要轉換的模型位置
--trained_checkpoint_prefix <path_of_input_model>
# 輸出位置
--output_directry <output_path>
```

