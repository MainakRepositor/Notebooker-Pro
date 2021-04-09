# Notebooker-Pro
The best notebook maker

<hr>

## Abstract:

The main idea of the software named, “Notebooker Pro” is to make the task of preparing a data science notebook, super easy and fast. With this software, any person can quickly get the insights about a data set displayed in front of him. This software also provides effective visualization facilities with 6 types of commonly used graphs. Anyone can also compare through 30 different Machine Learning models of both regression and classification types to choose the best model that can provide the highest accuracy for a data set of a given size and split ratio, (which can also be set by the user as per his/her requirements). 

<hr>

## Keywords:

Data Analysis, Machine Learning, Data Visualization, Model Building, Web Application, Data Science Notebooks, Education, Speed of Development.

<hr>

## Primary issues with existing techs:

<ol>
<li>Manual planning :  For making a good notebook, the creator needs to plan it well. The notebook must be informative and include all necessary details yet not over-burdening it with a flood of information which are irrelevant.<li>
<li>Time taken to code : Once the planning is done, now comes the most tedious and brain-storming part, that is to code the notebook into existence. This can take hours, and also result in aches in various body parts and a bit of monotony, working for the same thing for so long.<li>
<li>Finding a good accuracy : It is definitely a hard job to find the appropriate train-test-split ratio in order to increase the accuracy of the model. At times, due to a poor size selection, the test set and the train set data do not agree with each other, resulting in high training accuracy but low test set accuracy.</li>
<li>4.Messing up with syntax to make a proper graph : Making a good graph with proper axes/parameters is also a great challenge for a beginner or a person with a limited time frame.</li>
</ol>

<hr>

## UML Diagram

<img src="https://user-images.githubusercontent.com/64016811/114149670-c955e500-9938-11eb-9f47-7dbf0564f649.png">

<hr>

## WHAT IS NOTEBOOKER PRO

The notebooker pro is a user-friendly software designed to help you make a good data science notebook in few steps.
Well, notebooker pro will not be making a notebook for you, but will provide you with all the data insights that you 
will need to put in your kernel. The notebooker pro has been provided with 4 major sections:

i.  EDA (Explanatory Data Analysis)  --> used to find important data and statistical insights from the uploaded files

ii. Visualization --> Used to perform data visualization with 5 basic important types of graphs

iii.Regression --> Loops through 30 different regression models and returns the complexity statistics of the result
		   of regression modelling for your dataset for chosen seed values and size. The only thing to keep in
		   mind while using this is that, the data must be fitting with a regression modelling. Datasets used
		   for classification algorithm might generate vague results. So use a proper dataset.
		   **[eg.: do not use iris,cancer,penguins etc. classifier dataset]**

iv. Classification --> Loops through 30 different classification models and returns the complexity statistics of the result
		   of classification modelling for your dataset for chosen seed values and size. The only thing to keep in
		   mind while using this is that, the data must be fitting with a classification modelling. Datasets used
		   for non-classification algorithm might generate vague results. So use a proper dataset.
		   

Features:
Upload file => Upload only csv files.
Data split  => This is a linear slidebar, that will let you choose split ratio between 0 to 1
Random seed => Helps to randomize the data in training and testing data samples. 
	       You may change to get the best accuracy of for a particular model.
         
         
**You do not necessarily need to know coding to use this webapp**

<hr>

## Light Mode

<img src="https://user-images.githubusercontent.com/64016811/112420037-7da91600-8d52-11eb-8fd0-d8c916e6b313.jpg">

<hr>

## Dark Mode

<img src="https://user-images.githubusercontent.com/64016811/112760003-179ee600-9013-11eb-9a0f-9b0f4c701be3.jpg">

<hr>



The site is live at : https://share.streamlit.io/mainakrepositor/notebooker-pro/app.py

![ss1](https://user-images.githubusercontent.com/64016811/114223200-27acb300-998d-11eb-8cc5-2e4865102971.jpg)

