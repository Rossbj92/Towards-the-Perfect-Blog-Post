# Towards the Perfect Data Science Blog Post

Scraping nearly 10,000 blog posts from the the [Towardsdatascience](https://towardsdatascience.com/) archives, I attempted to build a machine learning model to predict the claps that a post received. 

## Replication Instructions

1. Clone repo
2. Run [webscraper](Notebooks/Scraping.ipynb) notebook
3. Process data with [cleaning](Notebooks/Cleaning.ipynb) notebook
4. Model ([modeling](Notebooks/Modeling.ipynb) notebook)

## Directory Descriptions

```Data```
- ```medium_2020_posts.pkl``` - raw scraped data used in the cleaning notebook
- ```medium_2020_processed.pkl``` - cleaned data used in the modeling notebook

```Notebooks```
- [Scraping](Notebooks/Scraping.ipynb) - used for scraping the Towardsdatascience archives
- [Cleaning](Notebooks/Cleaning.ipynb) - text processing, feature engineering, EDA
- [Modeling](Notebooks/Modeling.ipynb) - model building and evaluation

```Fonts```
- Custom fonts used in notebooks for plotting

```Presentation```
- pdf and ppt format of final project presentation

## Conclusions

- Break posts into sections
- Use visuals
- Avoid passive voice
- Tutorials and overviews garner more interest than news pieces (e.g., trends in tech)

Model performance was, however, relatively poor in terms of both r-squared and MAE. Future directions will focus on new data sources, such as author characteristics (e.g., followers) and social media factors (e.g., tweets of post). 

### Methods Used
- Linear modeling (linear regression, Ridge, Lasso, ElasticNet)
- Poisson regression
- Random forest
- Topic modeling
- Cross validation


