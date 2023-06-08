在命令行修改-t 可以改task id

- task id =0 , 用长度只有5行的jsonl文件测试能不能跑通。
- task id =1 , 在dataset 1上train, dataset 1上test，dataset2上test2
- task id =2 , 在dataset 2上train, dataset 2上test, dataset1上test1
- 不同的task id在wandb上会保存到不同的project下面。