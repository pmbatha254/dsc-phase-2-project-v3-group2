## Project title
### Box Office Blueprint: *Data-Driven Insights for Smarter Movie Production*

## Collaborators
1. PAULINE KIMENZU
2. JORAM MUGESA
3. EDINAH OGOTI
4. KELVIN SESERY

## Business Problem
Box Office Blueprint company now sees all the big companies creating original video content and they want to get in on the fun. They have decided to create a new movie studio, but they don’t know anything about creating movies. You are charged with exploring what types of films are currently doing the best at the box office. You must then translate those findings into actionable insights that the head of your company's new movie studio can use to help decide what type of films to create.

## Main Objective
To analyze current box office performance and identify the types of films that achieve the greatest financial success and audience appeal, providing actionable insights for the company’s new movie studio.

## Specific Objective 
1. Which genres are more likely to get highest critics ratings? 

2. Is there a relationship between the production budget and revenue in worldwide gross? 

3. Which original languages are more popular in screening in box office?


## Business Understanding
Box Office Blueprint is establishing a new movie studio and needs to understand the current market landscape to produce successful films. Analyzing box office performance of existing movies will provide data-driven insights into profitable film types, helping the new studio make informed decisions about genre, budget, and release strategy.

### The primary goal of the  project is:
To provide data-driven insights that will help Box Office Blueprint new film studio identify what types of movies are most likely to succeed at the box office, so the company can make smarter investment and production decisions.
In nutshell:  use data to reduce risk and increase the chances of producing profitable films.

## Data Understanding
The analysis will utilize movie data from various sources to understand factors influencing box office success. This includes information on movie titles, genres, release dates, budgets, and worldwide gross revenue. The data will need to be explored to assess its completeness, accuracy, and suitability for addressing the specific objectives.

## Loading Datasets

#### Load and explore the available datasets to understand their content and identify relevant data for the analysis.
Five Datasets are stored in specified CSV file and one is a SQL Database. We will load the data into a pandas DataFrame, display the first few rows, display column information, and display a summary of the DataFrames.


```python
#import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sqlite3
import statsmodels.api as sm
import re

```

### Loading bom.movie_gross.csv dataset and displaying summary Information


```python
# Load the data from bom.movie_gross.csv
df1 = pd.read_csv("zippedData/bom.movie_gross.csv.gz",low_memory=False)        

# Display the first 5 rows
print("First 5 rows of the DataFrame (bom.movie_gross.csv):")
display(df1.head())
# Display column names and their data types



### Loading tn.movie_budgets.csv dataset and displaying summary Information


```python
# Load the data from tn.movie_budgets.csv
df2 = pd.read_csv("zippedData/tn.movie_budgets.csv.gz")         

# Display the first 5 rows
print("First 5 rows of the DataFrame (tn.movie_budgets.csv):")
display(df2.head())

# Display column names and their data types
print("\nColumn names and data types (tn.movie_budgets.csv):")
display(df2.info())
```

### Loading tmdb.movies.csv dataset and displaying summary Information


```python
# Load the data from tmdb.movies.csv
df3 = pd.read_csv("zippedData/tmdb.movies.csv.gz")         

# Display the first 5 rows
print("First 5 rows of the DataFrame (tmdb.movies.csv):")
display(df3.head())

# Display column names and their data types
print("\nColumn names and data types (tmdb.movies.csv):")
display(df3.info())
```

### Loading rt.movie_info.tsv dataset and displaying summary Information


```python
# Load the data from rt.movie_info.tsv
df4 = pd.read_csv("zippedData/rt.movie_info.tsv.gz", sep="\t",compression="gzip",encoding="latin-1")      

# Display the first 5 rows
print("First 5 rows of the DataFrame (rt.movie_info.tsv):")
display(df4.head())

# Display column names and their data types
print("\nColumn names and data types (rt.movie_info.tsv):")
display(df4.info())
```

### Loading rt.reviews.tsv dataset and displaying summary Information


```python

# Load the data from rt.reviews.tsv
df5 = pd.read_csv("zippedData/rt.reviews.tsv.gz", sep="\t",compression="gzip",encoding="latin-1")     

# Display the first 5 rows
print("First 5 rows of the DataFrame (rt.reviews.tsv):")
display(df5.head())

# Display column names and their data types
print("\nColumn names and data types (rt.reviews.tsv):")
display(df5.info())
```


### Loading im.db Database(SQLite)


```python
#Establish a connection to database and list available tables
conn = sqlite3.connect("zippedData/im.db")
tables = pd.read_sql("SELECT name FROM sqlite_master", conn)
tables
```


```python
# Connecting and reading from a specified table
df_basics = pd.read_sql("SELECT * FROM movie_basics;", conn)
df_basics
```


```python
# Connecting and reading from a specified table
df_rating = pd.read_sql("SELECT * FROM movie_ratings;", conn)
df_rating
```


## Data Preparation, Cleaning, Exploration and Visualization

 #### Objective 1: Which genres are more likely to get highest critics ratings? 


```python
df_rating = pd.read_sql("SELECT * FROM movie_ratings;", conn)
df_rating
```


```python
#Check missing Values  on the Review Dataset

# Count of nulls per column
df_rating_clean = df_rating.isna().sum()
df_rating

```



```python
# Explore Data on the Movie ratings data with the movie basics
query ="""SELECT 
    b.genres,
    r.movie_id,
    AVG(r.averagerating) AS avg_rating,
    SUM(r.numvotes) AS total_votes
FROM  movie_ratings r
JOIN movie_basics b
    ON r.movie_id = b.movie_id
    GROUP BY b.genres
    ORDER BY avg_rating DESC;"""
df_genre = pd.read_sql(query,conn)
df_genre
```


```python
# Visualize the data
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_genre, x='avg_rating', y='total_votes', hue='genres', legend=False)
plt.title('Average Movie Rating vs. Total Votes by Genre')
plt.xlabel('Average Rating')
plt.ylabel('Total Votes')
plt.grid(True)
plt.show()
```


    
<img width="1001" height="699" alt="output_30_0" src="https://github.com/user-attachments/assets/3820dd68-28f3-44e9-91b1-c11fe50cd490" />

    



```python
# Determine a suitable threshold for total_votes
# Looking at the scatter plot, a threshold of 10000 seems reasonable to exclude genres with very few votes.
threshold = 10000

# Filter the DataFrame based on the chosen total_votes threshold
df_filtered_genre = df_genre[df_genre['total_votes'] > threshold].copy()

# Display the filtered DataFrame
display(df_filtered_genre)
```



#### Split genres
Create a new DataFrame where each row represents a single genre from the 'genres' column of the filtered data.


```python
# Split the 'genres' column by comma and stack the results
genres_split = df_filtered_genre['genres'].str.split(',').explode()

# Reset the index of the resulting Series
genres_split = genres_split.reset_index()

# Create a new DataFrame by merging the original filtered DataFrame with the split genres
df_split_genres = pd.merge(df_filtered_genre, genres_split, left_index=True, right_on='index')

# Rename the column containing the individual genres
df_split_genres = df_split_genres.rename(columns={'genres_y': 'single_genre'})

# Drop the unnecessary index column from the merge
df_split_genres = df_split_genres.drop(columns=['index', 'genres_x'])

# Display the new DataFrame
display(df_split_genres)
```

#### Analyze individual genres
Calculate the average rating and total votes for each individual genre.


```python
df_individual_genre_analysis = df_split_genres.groupby('single_genre').agg(
    avg_rating=('avg_rating', 'mean'),
    total_votes=('total_votes', 'sum')
).reset_index()

display(df_individual_genre_analysis)
```

#### Visualize individual genres
Creating visualizations to show the distribution of average ratings and total votes for individual genres.


```python
plt.figure(figsize=(15, 7))
sns.barplot(data=df_individual_genre_analysis.sort_values('avg_rating', ascending=False), x='single_genre', y='avg_rating')
plt.title('Average Rating by Individual Genre')
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```


    
![png](output_38_0.png)
    



```python
plt.figure(figsize=(15, 7))
sns.barplot (data=df_individual_genre_analysis.sort_values('total_votes', ascending=False), x='single_genre', y='total_votes')
plt.title('Total Votes by Individual Genre')
plt.xlabel('Genre')
plt.ylabel('Total Votes')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```


    
![png](output_39_0.png)
    


#### Summary:
##### Data Analysis Key Findings
After filtering, the dataset contained 356 rows representing movies with more than 10,000 total votes.
The analysis calculated the average rating and total votes for each individual genre present in the filtered dataset.
Two bar plots were generated, visualizing the average rating per genre and the total votes per genre.

Looking at the "Average Rating by Individual Genre" plot and the df_individual_genre_analysis DataFrame, the genres with the highest average ratings appear to be:

Documentary: (Highest average rating) News: (Second highest average rating) Biography: History: Music: 
Most Popular Genres (by Total Votes):

Examining the "Total Votes by Individual Genre" plot and the df_individual_genre_analysis DataFrame, the genres with the most total votes are:

Drama: (Highest total votes) Action: (Second highest total votes) Adventure: Comedy: Thriller: 

Summary:
The analysis of individual genres, after filtering for movies with more than 10,000 votes, reveals a distinction between genres that are highly rated and those that are most popular in terms of the sheer number of votes. Documentaries and News tend to have the highest average ratings among individual genres, while Drama, Action, and Adventure receive the most total votes. This suggests that while some genres are critically acclaimed, others have a broader appeal and larger viewership base.

We can also note that some genres like 'Horror' have a relatively low average rating but a high number of total votes, indicating a large audience despite lower average scores. Conversely, genres like 'News' have high average ratings but significantly fewer total votes compared to the most popular genres.

This analysis highlights that focusing solely on average rating or total votes would provide an incomplete picture. A comprehensive understanding requires considering both metrics.

#### Analyze Relationship between Average Rating and Total Votes
Create a scatter plot to visualize the relationship between average rating and total votes for individual genres and calculate the correlation coefficient to quantify the strength and direction of this relationship.


```python
# Using the df_split_genres from the previous steps.

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_individual_genre_analysis, x='avg_rating', y='total_votes', hue='single_genre', legend=False)
plt.title('Average Rating vs. Total Votes for Individual Genres')
plt.xlabel('Average Rating')
plt.ylabel('Total Votes')
plt.grid(True)
plt.show()

# Calculate the correlation coefficient
correlation = df_individual_genre_analysis['avg_rating'].corr(df_individual_genre_analysis['total_votes'])

print(f"The correlation coefficient between average rating and total votes for individual genres is: {correlation:.2f}")
```


    
![png](output_43_0.png)
    
The correlation coefficient between average rating and total votes for individual genres is: -0.40


The correlation coefficient of -0.40 suggests a weak negative linear relationship. In other words, there is a slight tendency for genres with higher average ratings to have fewer total votes, and vice versa. However, this relationship is not very strong, as indicated by the value being closer to 0 than to -1.
Data Analysis Key Findings
The analysis focused on individual genres with at least 500 movies.
The top-rated genres (highest average rating) are War, History, and Western.
The most popular genres (highest total votes) are Drama, Comedy, and Action.

## Objective 2: Is there a relationship between the production budget and revenue in worldwide gross?

#### Step 1: Loading the Data
Load the tn.movie_budgets.csv file into a pandas DataFrame.


```python
df2 = pd.read_csv("zippedData/tn.movie_budgets.csv.gz")        
df2.head()
```

```python
#check for missing values
df2.isnull().sum()
```


#### Step 2: Data Cleaning and Preprocessing
The production_budget and worldwide_gross columns are currently stored as strings with dollar signs and commas, which prevents them from being used in numerical calculations. This step converts these columns to a numerical format (integers) for analysis.


```python
#Convert the data type to float
df2["production_budget"] = pd.to_numeric(
    df2["production_budget"].replace({r'\$': '', ',': ''}, regex=True),
    errors="coerce"
)
```


```python
#Convert the data type to float
df2["worldwide_gross"] = pd.to_numeric(
    df2["worldwide_gross"].replace({r'\$': '', ',': ''}, regex=True),
    errors="coerce"
)
```


```python
#Convert the data type to float
df2["domestic_gross"] = pd.to_numeric(
    df2["domestic_gross"].replace({r'\$': '', ',': ''}, regex=True),
    errors="coerce"
)
```


```python
df2.info()
```
#### Step 3: Exploratory Data Analysis (EDA)
Here, we will analyze the relationship between the production_budget and worldwide_gross using a scatter plot and a correlation coefficient. The scatter plot helps visualize the relationship, while the correlation coefficient provides a quantitative measure.


```python
# Compute correlation
correlation = df2["production_budget"].corr(df2["worldwide_gross"])
print("Correlation between production budget and worldwide gross:", correlation)
```

    Correlation between production budget and worldwide gross: 0.7483059765694761


#### Strong positive relationship – as the production budget increases, the worldwide gross tends to increase as well.


```python
r_squared = correlation**2
plt.figure(figsize=(8,6))
sns.regplot(
    data=df2,
    x="production_budget",
    y="worldwide_gross",
    scatter_kws={"alpha":0.5},
    line_kws={"color":"red"}
)

# Add R² value as text on the plot
plt.text(
    x=df2["production_budget"].max()*0.6,   # position X
    y=df2["worldwide_gross"].max()*0.9,     # position Y
    s=f"R² = {r_squared:.2f}",
    fontsize=12,
    color="black",
    bbox=dict(facecolor="white", alpha=0.7)  # add background box
)

plt.title("Production Budget vs Worldwide Gross")
plt.xlabel("Production Budget")
plt.ylabel("Worldwide Gross")
plt.show()
```


    
![png](output_58_0.png)
    


#### Key Findings 
Positive slope of the red line → as production budget increases, worldwide gross tends to increase.

Clustering near the line → strong correlation (confirmed by your r = 0.7483).

Scatter (variance) → not all movies with big budgets make big grosses. Some may underperform, and a few low-budget films may outperform expectations.

Outliers → if you see points far above/below the line, those represent unusually successful or unsuccessful films compared to their budgets.

#### R- Squared Interpretation
56% of the variation in worldwide grosses can be explained by differences in production budgets.

The other 44% is due to other factors (marketing, distribution, timing, reviews, genre, star power, etc.).

Strong positive relationship → big-budget films tend to perform well, but success is not guaranteed.


```python
# Define predictor (X) and response (y)
X = df2["production_budget"]
y = df2["worldwide_gross"]

# Add constant (intercept) to the model
X = sm.add_constant(X)

# Fit linear regression model
model = sm.OLS(y, X).fit()

# Show summary of results
print(model.summary())
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:        worldwide_gross   R-squared:                       0.560
    Model:                            OLS   Adj. R-squared:                  0.560
    Method:                 Least Squares   F-statistic:                     7355.
    Date:                Fri, 12 Sep 2025   Prob (F-statistic):               0.00
    Time:                        19:21:07   Log-Likelihood:            -1.1557e+05
    No. Observations:                5782   AIC:                         2.311e+05
    Df Residuals:                    5780   BIC:                         2.311e+05
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    =====================================================================================
                            coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------
    const             -7.286e+06   1.91e+06     -3.813      0.000    -1.1e+07   -3.54e+06
    production_budget     3.1269      0.036     85.763      0.000       3.055       3.198
    ==============================================================================
    Omnibus:                     4232.022   Durbin-Watson:                   1.005
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           172398.262
    Skew:                           3.053   Prob(JB):                         0.00
    Kurtosis:                      29.044   Cond. No.                     6.57e+07
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 6.57e+07. This might indicate that there are
    strong multicollinearity or other numerical problems.


#### Analysis and Recommendations
Based on the analysis, a strong positive correlation exists between a movie's production budget and its worldwide gross revenue. The correlation coefficient of approximately 0.74 suggests that as the production budget increases, the worldwide gross revenue also tends to increase. However, it's important to note that correlation does not imply causation. A higher budget doesn't guarantee a blockbuster, as other factors such as marketing, cast, and audience reception play significant roles.

### Objective 3: Which original languages are more popular in screening in box office?

#### Step 1: Data Preparation and Merging
The analysis begins by loading two datasets: tmdb.movies.csv (which contains original language information) and bom.movie_gross.csv (which contains box office data). The data from both files is then merged on the title column to combine language and gross revenue information for each movie.


```python
# Clean the 'title' column in `bom_df` to remove the year in parentheses for better merging.
df1['title'] = df1['title'].apply(lambda x: re.sub(r' \(\d{4}\)', '', x))
```


```python
# Clean the 'foreign_gross' column in `df1` to a numeric type.
df1['foreign_gross'] = df1['foreign_gross'].str.replace(',', '', regex=False).astype(float)

# Merge the two DataFrames on the 'title' column.
merged_df = pd.merge(df1, df3, on='title', how='inner')

# Create a new column for worldwide gross by summing domestic and foreign gross.
merged_df['worldwide_gross'] = merged_df['domestic_gross'].fillna(0) + merged_df['foreign_gross'].fillna(0)

# Display the first few rows of the merged data.
merged_df.head()
```

#### Step 2: Analysis and Visualization
After preparing the data, we can now analyze which original languages have generated the most revenue in the box office. We'll group the data by original_language and sum the worldwide_gross for each language. The results will then be visualized using a bar chart.


```python
# Group by original_language and sum worldwide gross
language_gross = merged_df.groupby('original_language')['worldwide_gross'].sum().sort_values(ascending=False)

# Create a DataFrame from the grouped series for easier plotting
language_gross_df = language_gross.head(10).reset_index()
language_gross_df.columns = ['original_language', 'worldwide_gross']

# Create the bar chart to visualize the results
plt.figure(figsize=(12, 7))
sns.barplot(x='original_language', y='worldwide_gross', data=language_gross_df)
plt.title('Worldwide Gross by Original Language (Top 10)', fontsize=16)
plt.xlabel('Original Language', fontsize=12)
plt.ylabel('Worldwide Gross (in Billions USD)', fontsize=12)
plt.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


    
![png](output_69_0.png)
    


#### Results 
The bar chart shows a clear dominance of English (en) language films in terms of worldwide box office gross.


#### Recommendation
Based on this analysis we can recommend that movies be produced in English as it is the dominant/popular original language in the box office market worldwide.

## Conclusion and Business Recommendations
##### 1.Prioritize Documentaries and Drama films
The analysis shows that Drama dominates in audience votes, making it the most popular genre.
However, Documentaries lead in ratings, showing they resonate more deeply with critics or dedicated viewers.
Therefore, a dual approach is advisable: leverage Drama for broad audience engagement while investing in Documentaries to strengthen brand reputation and quality perception.
This strategic mix would maximize both commercial returns and critical recognition, ensuring long-term success for the company.

##### 2. Invest in a Higher Production Budget
There is a positive relationship between a film's production budget and its worldwide gross revenue. To increase the potential for a high return on investment, the new venture should consider allocating a higher budget to its film productions. However, it is important to note that a high budget does not guarantee success, as some high-budget films underperform, and some low-budget films exceed expectations.

##### 3. Produce Films in English
The analysis found that English is the dominant and most popular original language for films in the worldwide box office market. To maximize the film’s potential reach and commercial success, the studio should prioritize producing movies in English.


