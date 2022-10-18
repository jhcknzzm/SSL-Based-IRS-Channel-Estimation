# SSL-Based-IRS-Channel-Estimation

This repository implements "A Self-Supervised Learning-Based Channel Estimation for IRS-Aided Communication Without Ground Truth". 
The code runs on Python 3.9.7 with PyTorch 1.9.1 and torchvision 0.10.1.


First download the dataset from https://drive.google.com/drive/folders/1lCFGizQetXrO9nj4Mee1JJJE_KN3fTBl?usp=sharing, and save it in the /data/ folder.
One can also generate IRS channel data from other open source Repo. like: https://github.com/XML124/CDRN-channel-estimation-IRS.

You can run the following command to run our code to train a neuranl network for Channel Estimation in IRS-Aided Communication System Without Ground Truth, and save the results in the /results/ folder.

`nohup python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/channel_estimation.yaml --ckpt_dir ~/.cache/ --hide_progress --method sim_x --SNR 0.0 --std 0.5 `

To run the main script main.py, at least 2 GPUs are required.

Parameters:

--method: The training method. method == sim_x: use our self-supervised learning method; method == supervise, use supervised learning.

--SNR: Transmit signal to noise ratio.

--std: Standard deviation of the noise added to the received signal


The results will be saved at /results.

You also can run the following .sh file to reproduce our experimental results.

`nohup bash run_main.sh`

## Citation

We appreciate it if you would please cite the following paper if you found the repository useful for your work:

```
@ARTICLE{zhang2022SSLChannelEstimation,
  title={A Self-Supervised Learning-Based Channel Estimation for IRS-Aided Communication Without Ground Truth},
  author={Zhengming Zhang, Taotao Ji, Haoqing Shi, Chunguo Li, Yongming Huang, Luxi Yang}
  year={2022}
}
```


