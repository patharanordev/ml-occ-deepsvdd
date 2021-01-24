# **Custom DeepSVDD for Image Classification**

Custom Deep Support Vector Data Description(or DeepSVDD) for One-Class Classification(OCC) in Machine Learning (Ref. Deep OCC ICML 2018 paper).

This repository forked from [lukasruff repository](https://github.com/lukasruff/Deep-SVDD), allows user to custom dataset.

## **Prepare Environment**

```bash
$ python3 -m venv env
$ source env/bin/activate
$ pip3 install -r requirements.txt
```

## **Usage**

 - Adding train list in `train.csv` in `./data/custom/` directory.
 - Adding train list in `test.csv` in `./data/custom/` directory.
 - Adding your image dataset to `./data/custom/` directory.

### **Example**

train.csv

```csv
path,label
normal/1.png,0
abnormal/2.png,1
...
```

test.csv

```csv
path,label
test/3.png,0
test/4.png,1
...
```

## **Train Data**

```bash
$ sh run-custom.sh
```