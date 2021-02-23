![Image](./images/ted_logo.png)

# TED Talk Text Generator

## 1. Executive Summary

There are presently several popular topics in data science, especially in the category of Natural Language Processing. Current areas are research include sentiment analysis, chat bots, speech recognition, and text generation.

Interest in Artificial Intelligence and text generation spiked in May of 2020, when Open AI Labs released a new version of their deep learning model called GPT-3. GPT-3 is a highly trained text generation engine built by reading 45 TB of data from public web sites, and electronic books and other sources. GPT-3 uses 175 billion parameters, and produces an amazingly high level of quality text.  

This project attempts to pursue if and how a simple text generator could be built, and whether the quality of the output text could pass as something legible and understood by humans.

## 2. Text Generator Inner Workings

There are many published articles, blog posts, and YouTube video tutorials to assist in describing how to build a text generator (cited below). Most articles describe a technique in how one could employ a recurrent neural network (RNN), specifically a Long Short Term Memory RNN.

A node in a LSTM layer consists of "remember" and "forget" gates, which allow it to carry forward information to the next node in the network. This helps the network keep track of values it has seen in it's recent history, and helps it predict the next value.

Christopher Olah has written a detailed blog post [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/),
 posted on August 27, 2015 which explains in great detail the inner workings of the individual cells in a LSTM model.

As someone who is the relatively early stages of his data science learning, I was somewhat surprised to realize the real work being done by a text generator, can be simply summarized as as multi-class classification problem.

Borrowed from a [Towards Data Science post by Javaid Nabi](https://towardsdatascience.com/machine-learning-multiclass-classification-with-imbalanced-data-set-29f6a177c1a):
> Multiclass Classification: A classification task with more than two classes; e.g., classify a set of images of fruits which may be oranges, apples, or pears. Multi-class classification makes the assumption that each sample is assigned to one and only one label: a fruit can be either an apple or a pear but not both at the same time.

In the case of a text generator, the multiple classes are the labels assigned to the next character or word a model is predicting, also known as it's target variable (y).

Current generators can choose to base the model training on either:
* Single characters
* Words

A single character model, for example, would use a sequence of characters as its input variables (features), then predict a single target character.

For example, in the sentence "Welcome to my TED talk.", a model uses a sequence length of 8, would use the first 8 characters as its input variables, in order to predict the 9th character.

![Image](./images/welcome_1.png)

## 2. Data Collection

All text generation models require a group of text, also called a corpus



## 3. Data Cleaning & Pre-Processing


**Columns included in feature set:**

|Feature|Type|Description|
|---|---|---|
|protestnumber|int|Number of protests that year in specific country|
|protesterviolence|int(binary)|Indicates whether protester enacted violence|
|pop_total|float|Population of country where protest takes place|
|pop_density|float|Population density of country where protest takes place|
|prosperity_2020|float|Prosperity index of country the year protest takes place|
|region_Africa|int (binary)|Country is in Africa|
|region_Asia|int (binary)|Country is in Asia|
|region_Central America|int  (binary)|Country is in Central America|
|region_Europe|int (binary)|Country is in Europe|
|region_MENA|int (binary)|Country is in the Middle East/North Africa|
|region_North America|int (binary)|Country is in North America|
|region_Oceania|int (binary)|Country is in Oceania|
|region_South America|int (binary)|Country is in South America|
|protest_size_category_Less than 50|int (binary)|Number of participants <50|
|protest_size_category_50-99|int (binary)|50 - 99 participants|
|protest_size_category_100-999|int (binary)|100 - 999 participants|
|protest_size_category_1,000-4,999|int (binary)|1,000 - 4,999 participants|
|protest_size_category_5,000-9,999|int (binary)|5,000 - 9,999 participants|
|protest_size_category_10,000-100,000|int (binary)|10,000 - 100,000 participants|
|protest_size_category_Over 100,000|int (binary)|Number of participants >100,000|
|protester_id_type_civil_human_rights|int (binary)|Indicates participants are civil/human rights group|  
|protester_id_type_ethnic_group|int (binary)|Indicates participants are ethnic group|
|protester_id_type_locals_residents|int(binary)|Indicates participants are local residents|
|protester_id_type_pensioners_retirees|int (binary)|Indicates participants are pensioner/retirees|
|protester_id_type_prisoners|int (binary)|Indicates participants are prisoners|
|protester_id_type_protestors_generic|int (binary)|Indicates participants are generic group|
|protester_id_type_religious_group|int (binary)|Indicates participants are from religious group|    
|protester_id_type_soldiers_veterans|int (binary)|Indicates participants are soldiers/veterans|
|protester_id_type_students_youth|int (binary)|Indicates participants are students/youth|
|protester_id_type_victims_families|int (binary)|Indicates participants are families of victims|
|protester_id_type_women|int (binary)|Indicates participants are primarily groups of women|
|protester_id_type_workers_unions|int (binary)|Indicates participants are union members or workers|
|labor_wage_dispute|int( binary)|Protest motivation is labor & wage disputes|
|land_farm_issue|int (binary)|Protest motivation is land and farming conflict|
|police_brutality|int (binary)|Protest motivation is police brutality|
|political_behavior_process|int (binary)|Protest motivation is political behavior or process|
|price increases_tax_policy|int (binary)|Protest motivation is tax increase/tax policy|
|removal_of_politician|int (binary)|Protest motivation is removal of politician(s)|
|social_restrictions|int (binary)|Protest motivation associated with social restrictions|




## 4. Exploratory Datat Anaylsis

After the end of the data cleaning phase, we are left with one large main dataframe, which contains around 15,000 rows of data. At this stage, all categorical features columns have been encoded into a large collection of binary columns to represent each country, protestor category, protester type, etc. The new total number of columns at this state becomes 235.

Some analysis reveals the distribution of some of the input features.

![Protester ID Types](./images/Graph-ProtesterIdType.png)

<img src="./images/protest_size_categories.png" width="500" height="250">



<br>

After considering the possible effects using all of the encoded country names, we determined that doing that might add too many unimportant features, since a protest's general location is captured in a 'Region' field.

Some extra analysis was done running the text contained in the **notes** field through a Natural Language Processor, but this could have introduced information that is too highly correlated to the model's target variables, and was Ultimately not used as an input feature.  For example, a negative description in the **notes** field might skew the results.

After removing all country and any other features deemed unnecessary, the set of data to be passed into our models was reduced to approximately 70 columns.


## 5. Modeling

At first glance, we were presented with a multi-label classification problem i.e. a government can have multiple responses to the same protest. For these reasons, we initially built a couple of models: a Neural Network and a sklearn Multilabel Classifier using bagging and random forests ensemble methods. Our neural network model was promising given the first iteration, but we quickly realized we could not interpret any meaningful insights from it. Our Multilabel Classifier did not perform well.

For the second modeling effort, we built seven different logistic regressions for each of our target variables, effectively running a binary classification on each class. The results for these models varied widely and predictions were not reliable for classes that were less frequent in the data. This is a pitfall for having massively imbalanced data. More importantly to our interests, the violent responses were massively underrepresented.

The third modeling effort was a slight spinoff from our second modeling effort. First, we ran a binary logistic regression on whether or not the state government ignored a protest. After running this model, we filtered our data to exclude points where the government did not ignore a protest (i.e. which can be interpreted as when the government did respond). We then ran six different binary logistic regressions for each of the remaining six variables. Treating the ignore logistic regression separately reduced imbalances in our data and improved our scores across the board, especially for our columns of interest: the violent responses.

Given a decent product, we ran a gridsearch on several hyperparameters with an emphasis on the 'class_weight' of our logistic regression, which effectively penalizes wrong predictions for the class at hand based on a ratio default is 1:1).

We even went a step further and grouped all three of the violent responses--beatings, killings, and shootings--into one 'violent_response' column in order to reduce imbalances even more. However, this did not increase the scores of our model.

Please see the images folder in our repo for our final modeling technique's performance metrics, confusion matrices, and AUC plots.


## 6. Evaluation & Analysis


## 7. Conclusion


## 8. Next Steps & Future work


## 9. Acknowledgements

