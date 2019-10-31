# ＧＯＯＧＬＥ　ＡＰＩ　ＧＵＩＤＥＳ

## Dataset preparing

![flow chart](https://chtseng.files.wordpress.com/2019/02/6340_ynevl46ceg.png?w=760&zoom=2)

1. Label xml->csv using 
    A.更改設定檔 **cfg/xml2csv_config.json**
      ```json
      * label_path: 資料集的標註檔案位置(.xml)
      * out_path: 輸出csv檔案的位置與檔名(.csv)
      ```
    B. 執行程式 command: `python src/node/xml_to_csv.py`
2. Generate .record
    A.更改設定檔 **cfg/generate_tfrecord_config.json**
      ```json
      * csv_path: 上個步驟輸出的csv檔案位置
      * img_path: 資料集的圖片路徑
      * out_path: 輸出record檔案的位置與檔名(.record)
      ```
    B. 執行程式 command: `python workspace/src/generate_tfrecord.py`
3. Build your pbtxt, follow the style as below
```txt {.numberLines}
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
# 執行訓練的檔案
python <path_of_train.py>
# 設定參數檔的位置
pipeline_config_path=<config_path>
# 輸出位置與偵錯
--train_dir=<output_path> 2>&1 | tee logs/train_$now.txt
# 如果要使用多GPU的模式，請加入下面兩行
--num_clones=2 # 數量依照GPU數量而定
--ps_tasks=1
```
執行程式 command: `./scripts/train.bash`

5. Use tensorboard to observe the model

執行程式 command: `tensorboard --logdir=<output_path>`

**恭喜，你已經完成了神經網路的訓練了!!**

## Use this model !!!

1. Convert model.ckpt to inference.pb using "/workspace/scripts/export_model.bash"
```bash
# 建立輸出的資料夾
mkdir -p output
# 指定GPU使用哪些，本例使用兩顆GPU
CUDA_VISIBLE_DEVICES=0,1
# 要轉換的模型位置，通常選數字比較大的。
--trained_checkpoint_prefix <path_of_input_model>
# 設定參數檔的位置
pipeline_config_path=<config_path>
# 輸出位置
--output_directry <output_path>
```

command: `./scripts/export_model.bash`
這邊會生成**frozen_inference_graph.pb的檔案**，這個檔案是推論(inference)用的權重檔。

2. 開始框圖片，輸出json檔案。

  A. 更改設定檔 **cfg/model_inference.json**

    * PATH_OUTPUT: 預測的所有輸出結果，存為json檔案。
    * PATH_TO_CKPT: 上一步生成的權重檔**frozen_inference_graph.pb**。
    * DIR_IMAGE: 要預測的圖片的位置。
    * PATH_TO_LABELS: 生成資料集時的那個.pbtxtx檔。
    * NUM_CLASSES: 預測的數量(我們使用7種)。

  B. 執行程式 command: `python src/node/model_inference.py`

3. 將預測出來的.json轉成.txt
  
  本步驟為了利用[Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics)來測試mAP。

  A. 更改設定檔 **cfg/txt_generator.json**

    * DETECTION_PATH: 上一步生成的json檔案。
    * ANNOTATIONS_PATH: Ground truth的Annotation位置，若需要比較mAP則需要認真填，若僅欲生成預測的TXT，則可隨便填寫。
    * OUTPUT_PATH: 輸出的TXT放置的位置，會生成groundtruths, detections。

  B. 以下為分歧點，選一個執行就好
  
  a. 假如你是要測試mAP，那你需要同時有groundtruths與detections的txt
    command: 
      `python src/node/txt_generator.py -l` 
      or
      `python src/node/txt_generator.py --label`
  b. 只要生成資料庫的情況
    command:  `python src/node/txt_generator.py`

4. 將txt轉為最後我們要的xml。
  **TODO**

5. 統計生成的xml標籤的各個種類數量

  A. 更改設定檔 **cfg/categories_statistics.json**
  
    * xml_path: 上一步的xml annotation的位置。
  B. 執行程式 command: `python src/node/categories_statistics.py`
  
6. 輸出有帶有boundingbox的圖片

  A. 更改設定檔 **cfg/Image_with_bb.json**

    * DISPLAY: "SAVE"代表儲存、"SHOW"代表直接顯示
    * USAGE: "PREDICT"代表框預測的圖；"LABELS"代表要框LABEL的圖
    * THRESHOLD: 信心度大於多少才框
    * IMAGE_PATH: 圖片的路徑
    * DETECTIONS_PATH: TXT檔的路徑，(groundtruths, detections)的父資料夾
    * SAVE_PATH: 框出來的圖片要放置的位置
    * SAVE_FILENAME_EXTENSION: 副檔名要用什麼，通常是.jpg
    * BOUNDINGBOX_WIDTH: 框框的寬度
    * LIMIT_NUM: 限制輸出的圖片書量
    * CLASS_COLOR: 框框顏色，內建使用彩虹7色。
  B. 執行程式 command: `python src/node/Image_with_bb.py`

7. 輸出有帶有boundingbox的影片

  A. 更改設定檔 **cfg/inference_vids.json**

    * PATH_TO_CKPT: 第一步的frozen_inference_graph.pb
    * VIDEO_PATH: 輸入的影像
    * OUTPUT_PATH: 輸出的影像(副檔名為.mp4)
    * PATH_TO_LABELS: 生成資料集時的那個.pbtxtx檔
    * CLASS_COLOR: 框框顏色，內建使用彩虹7色。
    * THRESHOLD: 信心度大於多少才框
    * NUM_CLASSES: 分類數(我們使用7)

  B. 執行程式 command: `python src/node/inference_vids.py`

**恭喜，你已經完成了所有的部分了!!**

## 備註

1. 如果要特別指定要使用哪個GPU，請在python前加上CUDA_VISIBLE_DEVICES
  > e.g. 我想要用第二個GPU!!!  指令為:CUDA_VISIBLE_DEVICES=1
  > e.g. 我想要使用第二個加第三個GPU!! 指令為:CUDA_VISIBLE_DEVICES=1, 2
