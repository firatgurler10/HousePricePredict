# HousePricePredict

This Python code is used to load and examine California housing price data and visualize the correlation (relationship) between the data. The code creates a correlation map (heatmap) and rotates the text on the axes to improve readability. Let's see what it does step by step:

## General Operation of the Code:

- Loading Data:
The fetch_california_housing() function loads the California housing price data. This dataset includes features such as house prices and environmental factors.<br>The data is transferred to pandas.DataFrame, so it can be processed in a table form.Housing prices (PRICE) are added to the data frame.

- Exploring the Data:
The california_df.head() and california_df.describe() commands print the first few rows of the data and summary statistics to the screen. This provides information about the general characteristics of the data.

- Correlation Map:
A correlation map visualizes the relationship between each feature in the dataset. The map shows how much the features are related to each other (for example, whether two variables have a positive or negative correlation).The correlation map is created with the sns.heatmap function. This map shows the relationship between the features with colors. The annot=True parameter prints the correlation coefficients on the map.

- Rotation of Text on the X and Y Axes:
To prevent the long labels on the X and Y axes from overlapping on the map, these labels are rotated 45 degrees (rotation=45). This increases the readability of the text.

### What Do You Get When You Run the Code?
You will see the data titles and summary statistics on the screen.A heatmap will be created showing the correlation between all the features. Colors indicate the strength of the relationship between variables. For example, you can see the relationship between price and number of rooms in this map.This type of analysis is used especially in data science and machine learning projects to understand which variables contribute to the model.
