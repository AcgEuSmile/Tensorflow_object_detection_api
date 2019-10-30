# ＧＯＯＧＬＥ　ＡＰＩ　ＧＵＩＤＥＳ

## Dataset preparing

![flow chart](https://chtseng.files.wordpress.com/2019/02/6340_ynevl46ceg.png?w=760&zoom=2)

1. Label xml->csv using 
    * Set config file "workspace/cfg/xml2csv_config.json"
      * label_path: 資料集的標註檔案位置(.xml)
      * out_path: 輸出csv檔案的位置與檔名(.csv)
        > e.g.
        > "label_path": "/workspace/datasets/BDD100k/VOC_simple/Annotations/"
        > "out_path": "/workspace/yo/google_od_api/csv/test.csv"
    * Run "workspace/src/xml_to_csv.py"
2. Generate .record
    * Set config file "workspace/generate_tfrecord_config.json"
      * csv_path: 上個步驟輸出的csv檔案位置
      * img_path: 資料集的圖片路徑
      * out_path: 輸出record檔案的位置與檔名(.record)
        >e.g.
        >"csv_path": "/workspace/yo/google_od_api/csv/test.csv",
        >"img_path": "/workspace/datasets/BDD100k/VOC_simple/JPEGImages/",
        >"out_path": "/workspace/yo/google_od_api/tfRecord/test2.record"
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
**恭喜，你已經完成了GOOGLE API的資料集了!!**

## Train your model

1. Download the fine-tune model [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)

2. Download the config file [here](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs)

3. Modify the config file
```text
# 第一階段圖片的大小
  fixed_shape_resizer {
    height: 1280
    width: 720
  }
# 分類數量，與pbtxt檔內的id最大值一樣。
num_classes: 7
# 第二階段的batch大小
second_stage_batch_size: 48
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
# 指定GPU使用哪些，本例使用兩顆GPU
CUDA_VISIBLE_DEVICES=0,1
# 訓練的檔案
python <path_of_train.py>
# 輸出位置與偵錯
--train_dir=<output_path> 2>&1 | tee logs/train_$now.txt
# 如果要使用多GPU的模式，請加入下面兩行
--num_clones=2 # 數量依照GPU數量而定
--ps_tasks=1
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

