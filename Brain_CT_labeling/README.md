# Brain 3D CT images labeling

## Task

You have several brain CT images, labeled by 20 classes. Images were rotated before labeling, and each image has respective txt file with rotation data.

Your task is to create a tool, which labels new images in a same way.

Provide results in Jupyter Notebook, write report in pdf or other format.

**Описание.pdf** - full task text.


## Hi-level solution algoritm

1. De-rotate images.
2. Make an exploratory analysis and pre-process images if needed.
3. Select the classifier architecture and metrics, explain selection.
4. Train classifier and estimate its quality by selected metrics.

## Files

Source data is too large to fit this repository. All data can be downloaded from [here](https://drive.google.com/drive/folders/1jfoFSAHDMd55cK-qsOhgETMjPngIzQ66?usp=sharing).


**Отчет.pdf** - final report with analyticts and lots of text and pictures, no code there.


**main_notebook.ipynb** - notebook with solution.

**baseline_metrics.py** - saved baseline metrics for model training.

**modules.py** - all supplementary functions are here. File contains:

	- Translation functions from Euler angles to rotation matrix and back. 
	- Function for reading transform txt files
	- Image preprocessing loader: it loads transform data for image, setup and verify transform, then transform segmentation file.
	- Batch generator
	- Load and save Pickle utilities
	- Dice loss function
	- Metrics Class
	- Dataset creation function
	
**XZ train in Colab.ipynb, YZ train in Colab.ipynb** - notebooks with trainings, conducted in Google Colab - see report for details.

## Results example

<img src = "https://github.com/2326wz/Test_tasks/blob/master/Brain_CT_labeling/images/Capture.PNG?raw=true">



