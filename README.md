# PM2.5Forecast
This repository represents the final project of pattern recognition in 2020 spring for PM2.5 prediction with a  Seq2Seq network and attention.

## Dataset
The data can be downloaded from [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/d/f519a587a6d943fa9aa0/).

## Requirements
Python 3.7 or later with all requirements.txt dependencies installed, including torch >= 1.3. To install run:
<pre>
<code>
$ pip install -U -r requirements.txt
</code>
</pre>

## Modules
* create.py: read the csv files and convert them into a h5 file.
* clean.py: clean the data and split the data for dataloader.
* train.py: train the model.
* test.py: evaluate and test the model.
* dataloader.py: define the AirDataset.
* model.py: define the model of encoder and decoder.

## Prepare data
The data before 20140429 and on 20141231 is abandoned. Please delete them at first.
To prepare the database:
<pre><code>
$ python create.py 'trainval'
$ python create.py 'test'
</pre></code>
The prepared database can be downloaded from [PM2.5 database](https://cloud.tsinghua.edu.cn/d/a840d1bfbffe4647bc57/).


To get the final train/val/test split:
<pre><code>
$ python clean.py create_dataset --seed=0
</pre></code>

## Train and test
To train the model, please run:
<pre><code>
$ python train.py --lr=1e-3
                  --epochs=100
                  --hidden=8
</pre></code>
To evaluate the model, please run:
<pre><code>
$ python test.py --mode=val
                 --checkpoints = YOUR_MODEL_PATH
                 --model= YOUR_MODEL_NUMBER
</pre></code>
To test the model, please run:
<pre><code>
$ python test.py --mode=test
                 --checkpoints = YOUR_MODEL_PATH
                 --model= YOUR_MODEL_NUMBER
</pre></code>

## Results
Our baseline get 83.29% acc on val dataset and 64.61% acc on test dataset.
### mAP
|    | 1hour mAP| 2hour mAP| 3hour mAP| 4hour mAP| 5hour mAP| 6hour mAP| total mAP|
|----|----------|----------|----------|----------|----------|----------|----------|
|val |83.69|83.10|83.50|83.70|83.56|82.21|83.29|
|test|76.09|70.33|65.53|61.65|58.53|55.51|64.61|

### MAE
|    | 1hour MAE| 2hour MAE| 3hour MAE| 4hour MAE| 5hour MAE| 6hour MAE| total MAE|
|----|----------|----------|----------|----------|----------|----------|----------|
|val |8.718|8.946|8.731|8.601|8.736|9.395|8.855|
|test|13.86|17.53|20.96|24.01|27.78|29.35|22.08|
## Contact
Please don't contact me if you find some bugs. If not,
my mailbox will be overflowing.