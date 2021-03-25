# Notebooker-Pro
The best notebook maker
<img src="https://user-images.githubusercontent.com/64016811/112420037-7da91600-8d52-11eb-8fd0-d8c916e6b313.jpg">

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

The site is live at : https://share.streamlit.io/mainakrepositor/notebooker-pro/app.py
