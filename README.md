# Open-World KG
The official codes of Rethinking Knowledge Graph Evaluation Under the Open-World Assumption (NeurIPS 2022).


## Main experiments

As the main part of experiments in our paper, we trained several models and test them on the artificial family tree KG. We need `python=3.9` and the other environment requirements is shown in `requirements.txt`. The KG and the correlated version is in `data/family_gene` and `data/family_gene_cor`.

Our experiments need the complete training path for several models. To train the models and test them during training, run 
```bash
python script/auto.py
```
 for independent KG and 
 ```bash
python script/auto_cor.py
```
for correlated KG.

In this script, we continuously detect the remaining GPU-Util and memory of the GPUs, and submit new tasks when there is remaining computing power, until all the given models have been run. You can set the excluded GPU ids in the script.

We use the codes at [KGReasoning](https://github.com/snap-stanford/KGReasoning) for the BetaE and [kge-master](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding) for RotatE, ComplEx and pRotatE. We change the test process to test on both closed-world and open-world KGs.

The test results will be recorded in `logs` and `logs_cor`. Note that the process will be extremely time-consuming because we need to train tens of different models and test them frequently. The codes could spend about **20** **GPU days**. After all the models finish training, the complete training path is recorded in the Tensorboard file, then you can use the Jupyter Notebook `visualization.ipynb` to generate the figure which we show in the article.

## Data generation

We provide our artificial FamilyTree KG in directory `data`. If you want to generate your own FamilyTree KG, you can follow the steps below.

1. Use the codes at [Family tree data generator](https://github.com/phohenecker/) to generate the raw KG data. Please follow the instructions in this pages to generate the data. We use the following command to generate our KG.

    ```bash
    sh run-data-gen.sh path/to/dlv --max-branching-factor 20 --max-tree-depth 3 --max-tree-size 300 --num-samples 20 --stop-prob 0
    ```

2. Change the `path` in `data_generate/reform_to_raw.py` to the output directory of the previous step. Then run
    
    ```bash
    python data_generate/reform_to_raw.py
    ```
    
     to generate the KG data in the format of `data/family_gene`. The output file is a `.txt` file and each line is a triple in the format of `<head> <relation> <tail>`.

3. Split the training and test graph, queries and answers. Run
    
    ```bash
    python data_generate.py --path PATH --loadname FILENAME --sparse 95 85 75 65 --thre 10 --test_queries 500
    ```
    
    and you can look for more information by running

    ```bash
    python data_generate.py --help
    ```
4. We also provide the codes to re-format the generated data to the format of KGReasoning models and kge-master models. You can run

    ```bash
    python reform.py --sparse 95 85 75 65 --type q2b
    ```

    to KGReasoning format and

    ```bash
    python reform.py --sparse 95 85 75 65 --type transe
    ```

    to kge-master format.

    Note that the KGReasoning needs a `stat.txt` file to record the statistics of the KG. You need to create the file and write the following information in it.

    ```
    numentity: NUM_OF_ENTITIES
    numrelation: NUM_OF_RELATIONS
   ```
 
 5. We generate the **correlated** FamilyTree KG using the prediction given by a trained KGC model. Once you have a checkpoint, you can follow the procedure in `data_generate/cor_generate/cor_generate.ipynb` to generate the correlated KG. 

## Other experiments

There are several auxiliary experiments in our article. 

### Visualize the prediction

The codes to generate the detailed prediction report on FB15k237 are shown in `other_experiments/report`. When you have a checkpoint of a KGReasoning model (Q2B, BetaE) trained on FB15k237, you can run

```bash
python other_experiments/report/report.py --data_path PATH_TO_DATA --model_path PATH_TO_CHECKPOINT --detail --cuda
```

to generate the report. And you can use the Jupyter Notebook `other_experiments/report/name.ipynb` to find what objection the index refers to.

The program need the file `fb2w.nt` and you should **firstly unzip** the `fb2w.zip` to get the file.

Note the process is also **time-consuming**. We provide the detailed report which we generated in `other_experiments/report/report/output.txt`.

### Numerical simulation

The codes are in `other_experiments/simulation`. Firstly, run
```bash
python other_experiments/simulation/simu.py
```
to simulate the event in independent situation and then save the simulating results in a `.pkl` file. And use the Jupyter Notebook `other_experiments/simulation/visual.ipynb` to generate the figures.