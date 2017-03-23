import numpy as np
import pandas as pd
import matplotlib.pyplot as plt; plt.rcdefaults()
# import plotly
# from plotly.graph_objs import Scatter, Layout, Bar
# import matplotlib.pyplot as plt
# import plotly.plotly as py
# import plotly.graph_objs as go

input_train_ds = pd.read_csv('./train.csv', header=0)
input_test_ds = pd.read_csv('./test.csv', header=0)

columns = ['AnimalID', 'Name', 'DateTime', 'OutcomeType', 'OutcomeSubtype', 'AnimalType', 'SexuponOutcome',
           'AgeuponOutcome', 'Breed', 'Color']

outcomeTypes = ['Return_to_owner', 'Euthanasia', 'Adoption', 'Transfer', 'Died']
animalTypes = ['Dog', 'Cat']
sexUponOutcomes = ['Neutered Male', 'Spayed Female', 'Intact Male', 'Intact Female', 'Unknown']

cat_colors = ['Agouti', 'Apricot', 'Black', 'Blue', 'Brown', 'Buff', 'Calico', 'Chocolate', 'Cream', 'Flame', 'Gray',
              'Lilac', 'Lynx', 'Orange', 'Pink', 'Point', 'Seal', 'Silver', 'Smoke', 'Tabby', 'Tan', 'Tiger', 'Torbie',
              'Tortie', 'Tricolor', 'White', 'Yellow']

Y_FIELDS = ["Adoption", "Died", "Euthanasia", "Return_to_owner", "Transfer"]
X_FIELDS = []

# 1  |+| separate data into 2 parts Dogs and Cats - training classifier
# 2  |+| create field dummies for outcomes
# 3  |+| break age to regions (different for cats and dogs), create dummy fields for it
# 4  |+| create dummies for 'SexuponOutcome' field
# 5  |+| Break color to color dummies (different for cats and dogs)
# 6  |+| Break bread to bread dummies (different for cats and dogs)
# 7  |+| Organize above operation in one function
# 8  |-| Train two separate NNs (dogs and cats)
# 9  |-| ???
# 10 |-| PROFIT


# We need to create copy
dogs_ds = pd.DataFrame(input_train_ds[input_train_ds['AnimalType'] == 'Dog'])
cats_ds = pd.DataFrame(input_train_ds[input_train_ds['AnimalType'] == 'Cat'])
# cats_ds = input_train_ds[input_train_ds['AnimalType'] == 'Cat']
input_train_ds['SexuponOutcome'] = input_train_ds['SexuponOutcome'].fillna('Unknown')

# print('Uniques sex: ', input_train_ds['SexuponOutcome'].unique())


# Ploting data with sex outcomes
cats_x = outcomeTypes
cat_traces = []
dog_traces = []
for sex in sexUponOutcomes:
    cat_y = []
    dog_y = []
    for outcomeType in outcomeTypes:
        cat_result = sum((cats_ds['SexuponOutcome'] == sex) & (cats_ds['OutcomeType'] == outcomeType))
        dog_result = sum((dogs_ds['SexuponOutcome'] == sex) & (dogs_ds['OutcomeType'] == outcomeType))
        # print('Sum for ', sex, ' cats is: ', result)
        cat_y.append(cat_result)
        dog_y.append(dog_result)

    # cat_traces.append(go.Bar(x=outcomeTypes, y=cat_y, name=sex))
    # dog_traces.append(go.Bar(x=outcomeTypes, y=dog_y, name=sex))


# layout = go.Layout(barmode='group')
# cat_fig = go.Figure(data=cat_traces, layout=layout)
# dog_fig = go.Figure(data=dog_traces, layout=layout)
# PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS
# PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS
# plotly.offline.plot(cat_fig, filename='Sex-Outcome_cat_graph.html')
# plotly.offline.plot(dog_fig, filename='Sex-Outcome_dog_graph.html')
# PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS
# PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS

cats_sex_dummies = pd.get_dummies(cats_ds['SexuponOutcome'])
dogs_sex_dummies = pd.get_dummies(dogs_ds['SexuponOutcome'])

# print('SexDummies!!!')
# print(cats_sex_dummies.describe())

cats_sex_dummies.columns = sexUponOutcomes
dogs_sex_dummies.columns = sexUponOutcomes

cats_ds = cats_ds.join(cats_sex_dummies)
dogs_ds = dogs_ds.join(dogs_sex_dummies)

# looks like there is a pattern for every outcome separetly. Most obvious is in 'Adoption' outcome.
# So, I will start from it.

# Working with ages
def get_number(string_with_number):
    return float(string_with_number.split(' ')[0])


def transform_to_days(age_string):
    if pd.isnull(age_string):
        return 0
    if 'year' in age_string:
        return int(get_number(age_string) * 365.24)
    elif 'month' in age_string:
        return int(get_number(age_string) * 30.44)
    elif 'week' in age_string:
        return int(get_number(age_string) * 7)
    else:
        return int(get_number(age_string))

#
cats_ds.loc[:, 'AgeuponOutcome'] = cats_ds.loc[:, 'AgeuponOutcome'].apply(transform_to_days)
dogs_ds.loc[:, 'AgeuponOutcome'] = dogs_ds.loc[:, 'AgeuponOutcome'].apply(transform_to_days)

cats_nullAges = cats_ds['AgeuponOutcome'].isnull()
dogs_nullAges = dogs_ds['AgeuponOutcome'].isnull()

# print('NULL AGES ', len(catsNullAges))

cats_notNullAges = cats_ds['AgeuponOutcome'].notnull()
dogs_notNullAges = dogs_ds['AgeuponOutcome'].notnull()
cats_ds = cats_ds.replace({'AgeuponOutcome': 0}, int(round(cats_ds[cats_notNullAges]['AgeuponOutcome'].mean(), 0)))
dogs_ds = dogs_ds.replace({'AgeuponOutcome': 0}, int(round(dogs_ds[dogs_notNullAges]['AgeuponOutcome'].mean(), 0)))
cats_uniqueAges = cats_ds['AgeuponOutcome'].unique()
cats_uniqueAges.sort()
dogs_uniqueAges = dogs_ds['AgeuponOutcome'].unique()
dogs_uniqueAges.sort()
# print('Cats Age types: ', cats_uniqueAges)
# print('Dogs Age types: ', dogs_uniqueAges)
#
# # preparing plot for age:
number_of_cats_age_samples = len(cats_uniqueAges)
number_of_dogs_age_samples = len(dogs_uniqueAges)
# print('Cats Age types: ', number_of_cats_age_samples)
# print('Dogs Age types: ', number_of_dogs_age_samples)
cats_traces = []
dogs_traces = []
cats_x = [str(i) + ' days old' for i in cats_uniqueAges]
dogs_x = [str(i) + ' days old' for i in dogs_uniqueAges]


# Age ranges based on some empyrical analysis
cats_ds['less5Days'] = 0
cats_ds.loc[cats_ds['AgeuponOutcome'] < 5, 'less5Days'] = 1

cats_ds['5to28Days'] = 0
cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 5) & (cats_ds['AgeuponOutcome'] < 28), '5to28Days'] = 1

cats_ds['28to35Days'] = 0
cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 28) & (cats_ds['AgeuponOutcome'] < 35), '28to35Days'] = 1

cats_ds['35to152Days'] = 0
cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 35) & (cats_ds['AgeuponOutcome'] < 152), '35to152Days'] = 1

cats_ds['152to213Days'] = 0
cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 152) & (cats_ds['AgeuponOutcome'] < 213), '152to213Days'] = 1

cats_ds['213to273Days'] = 0
cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 213) & (cats_ds['AgeuponOutcome'] < 273), '213to273Days'] = 1

cats_ds['273to334Days'] = 0
cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 273) & (cats_ds['AgeuponOutcome'] < 334), '273to334Days'] = 1

cats_ds['334to496Days'] = 0
cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 334) & (cats_ds['AgeuponOutcome'] < 496), '334to496Days'] = 1

cats_ds['496to1460Days'] = 0
cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 496) & (cats_ds['AgeuponOutcome'] < 1460), '334to1460Days'] = 1

cats_ds['1460to2191Days'] = 0
cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 1460) & (cats_ds['AgeuponOutcome'] < 2191), '1460to2191Days'] = 1

cats_ds['2191to3287Days'] = 0
cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 2191) & (cats_ds['AgeuponOutcome'] < 3287), '2191to3287Days'] = 1

cats_ds['3287to4017Days'] = 0
cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 3287) & (cats_ds['AgeuponOutcome'] < 4017), '3287to4017Days'] = 1

cats_ds['4017to5113Days'] = 0
cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 4017) & (cats_ds['AgeuponOutcome'] < 5113), '4017to5113Days'] = 1

cats_ds['more5113Days'] = 0
cats_ds.loc[cats_ds['AgeuponOutcome'] >= 5113, 'more5113Days'] = 1


dogs_ds['less4Days'] = 0
dogs_ds.loc[dogs_ds['AgeuponOutcome'] < 4, 'less4Days'] = 1

dogs_ds['4to21Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 4) & (dogs_ds['AgeuponOutcome'] < 21), '4to21Days'] = 1

dogs_ds['35to91Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 35) & (dogs_ds['AgeuponOutcome'] < 91), '35to91Days'] = 1

dogs_ds['91to121Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 91) & (dogs_ds['AgeuponOutcome'] < 121), '91to121Days'] = 1

dogs_ds['121to213Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 121) & (dogs_ds['AgeuponOutcome'] < 213), '121to213Days'] = 1

dogs_ds['213to273Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 213) & (dogs_ds['AgeuponOutcome'] < 273), '213to273Days'] = 1

dogs_ds['273to334Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 273) & (dogs_ds['AgeuponOutcome'] < 334), '273to334Days'] = 1

dogs_ds['334to1007Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 334) & (dogs_ds['AgeuponOutcome'] < 1007), '334to1007Days'] = 1

dogs_ds['1007to1460Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 1007) & (dogs_ds['AgeuponOutcome'] < 1460), '1007to1460Days'] = 1

dogs_ds['1460to2556Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 1460) & (dogs_ds['AgeuponOutcome'] < 2556), '1460to2556Days'] = 1

dogs_ds['2556to3287Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 2556) & (dogs_ds['AgeuponOutcome'] < 3287), '2556to3287Days'] = 1

dogs_ds['3287to4017Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 3287) & (dogs_ds['AgeuponOutcome'] < 4017), '3287to4017Days'] = 1

dogs_ds['4017to4748Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 4017) & (dogs_ds['AgeuponOutcome'] < 4748), '4017to4748Days'] = 1

dogs_ds['4748to5843Days'] = 0
dogs_ds.loc[(dogs_ds['AgeuponOutcome'] >= 4748) & (dogs_ds['AgeuponOutcome'] < 5843), '4748to5843Days'] = 1

dogs_ds['more5843Days'] = 0
dogs_ds.loc[dogs_ds['AgeuponOutcome'] >= 5843, 'more5843Days'] = 1


# for outcomeType in outcomeTypes:
# for outcomeTypeIndex in np.linspace(0, len(outcomeTypes) - 1, num=len(outcomeTypes), dtype=int):
#     cats_y = np.zeros(number_of_cats_age_samples)
#     for index in np.linspace(0, len(cats_y) - 1, num=len(cats_y), dtype=int):
#         cats_y[index] = sum((cats_ds['AgeuponOutcome'] == cats_uniqueAges[index])
#                             & (cats_ds['OutcomeType'] == outcomeTypes[outcomeTypeIndex]))
#     cats_traces.append(go.Scatter(x=cats_x, y=cats_y, name=outcomeTypes[outcomeTypeIndex], fill='tozeroy', mode='none'))
#
#     dogs_y = np.zeros(number_of_dogs_age_samples)
#     for index in np.linspace(0, len(dogs_y) - 1, num=len(dogs_y), dtype=int):
#         dogs_y[index] = sum((dogs_ds['AgeuponOutcome'] == dogs_uniqueAges[index])
#                             & (dogs_ds['OutcomeType'] == outcomeTypes[outcomeTypeIndex]))
#     dogs_traces.append(go.Scatter(x=dogs_x, y=dogs_y, name=outcomeTypes[outcomeTypeIndex], fill='tozeroy', mode='none'))

# PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS
# PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS
# plotly.offline.plot(cats_traces, filename='Cats_Age-Outcome_scatter_graph.html')
# plotly.offline.plot(dogs_traces, filename='Dogs_Age-Outcome_scatter_graph.html')
# PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS
# PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS # PLOTS PLOTS PLOTS

# I have transformed age values from string to number of days.  And it looks like we should create several age groups
# as dummy features instead of plain age value. Unfortunately I was not able to find any patterns in Age-Outcome....

# Working with Breeds
unique_cats_breeds = cats_ds['Breed'].unique()
unique_cats_breeds.sort()
unique_dogs_breeds = dogs_ds['Breed'].unique()
unique_dogs_breeds.sort()
# print(unique_cats_breeds)
# print(unique_dogs_breeds)
#
# numberOfCatBreedSamples = len(unique_cats_breeds)
# numberOfDogBreedSamples = len(unique_dogs_breeds)
# print('Cat Breed types: ', numberOfCatBreedSamples)
# print('Dog Breed types: ', numberOfDogBreedSamples)
#
cat_breed_values = set()
dog_breed_values = set()
#
for breed_combination in unique_cats_breeds:
    cat_breed_values |= set(breed_combination.replace('/', ' ').split(' '))

for cat_breed_single_value in cat_breed_values:
    cats_ds[cat_breed_single_value] = 0
    cats_ds.loc[cats_ds['Breed'].str.contains(cat_breed_single_value, na=False), cat_breed_single_value] = 1


for breed_combination in unique_dogs_breeds:
    dog_breed_values |= set(breed_combination.replace('/', ' ').split(' '))

for dog_breed_single_value in dog_breed_values:
    dogs_ds[dog_breed_single_value] = 0
    dogs_ds.loc[dogs_ds['Breed'].str.contains(dog_breed_single_value, na=False), dog_breed_single_value] = 1





#     # colorValues.extend(color_combination.split(' '))
#
list_of_cat_breed_values = list(cat_breed_values)
list_of_cat_breed_values.sort()

list_of_dog_breed_values = list(dog_breed_values)
list_of_dog_breed_values.sort()

# print('Cat breeds')
# print(", ".join("'%s'" % x for x in list_of_cat_breed_values))
# print('Dog breeds')
# print(", ".join("'%s'" % x for x in list_of_dog_breed_values))


#
# # print('Cat Breed values: ', list_of_cat_breed_values)
# print('number of cat breed values: ', len(list_of_cat_breed_values))
#
# # print('Dog Breed values: ', list_of_dog_breed_values)
# print('number of dog breed values: ', len(list_of_dog_breed_values))
#
# # breed_traces = []
# # x = unique_cats_breeds
# #
# # for outcomeTypeIndex in np.linspace(0, len(outcomeTypes) - 1, num=len(outcomeTypes), dtype=int):
# #     y = np.zeros(numberOfCatBreedSamples)
# #     for index in np.linspace(0, len(y) - 1, num=len(y), dtype=int):
# #         y[index] = sum((cats_ds['Breed'] == unique_cats_breeds[index])
# #                        & (cats_ds['OutcomeType'] == outcomeTypes[outcomeTypeIndex]))
# #     breed_traces.append(go.Scatter(x=x, y=y, name=outcomeTypes[outcomeTypeIndex], fill='tozeroy', mode='none'))
# #
# # plotly.offline.plot(breed_traces, filename='CatBreed-Outcome_scatter_graph.html')
#
#
# # Working with colors
unique_cat_colors = cats_ds['Color'].unique()
unique_cat_colors.sort()

unique_dog_colors = dogs_ds['Color'].unique()
unique_dog_colors.sort()

number_of_cat_color_samples = len(unique_cat_colors)
print('Cat color types: ', number_of_cat_color_samples)

number_of_dog_color_samples = len(unique_dog_colors)
print('Dog color types: ', number_of_dog_color_samples)


cat_color_values = set()
dog_color_values = set()

for cat_color_combination in unique_cat_colors:
    cat_color_values |= set(cat_color_combination.replace('/', ' ').split(' '))

for cat_color_single_value in cat_color_values:
    cats_ds[cat_color_single_value] = 0
    cats_ds.loc[cats_ds['Color'].str.contains(cat_color_single_value, na=False), cat_color_single_value] = 1

for dog_color_combination in unique_dog_colors:
    dog_color_values |= set(dog_color_combination.replace('/', ' ').split(' '))

for dog_color_single_value in dog_color_values:
    dogs_ds[dog_color_single_value] = 0
    dogs_ds.loc[dogs_ds['Color'].str.contains(dog_color_single_value, na=False), dog_color_single_value] = 1

# print(cats_ds[cat_colors].describe())


#
# list_of_cat_color_values = list(cat_color_values)
# list_of_cat_color_values.sort()
# print('Cat Dummies: ', len(list_of_cat_color_values))
# print(", ".join("'%s'" % x for x in list_of_cat_color_values))


# l = ['a', 'b', 'c']
# print(", ".join("'%s'" % x for x in l))

# print "asdasda"

# for color in list_of_cat_color_values: print("\"" + color + "\",")

#
# list_of_dog_color_values = list(dog_color_values)
# list_of_dog_color_values.sort()
# print(", ".join("'%s'" % x for x in list_of_dog_color_values))
#
# print('Cat color types after transformation: ', len(list_of_cat_color_values))
# print('Dog color types after transformation: ', len(list_of_dog_color_values))
#
cats_outcome_dummies = pd.get_dummies(cats_ds['OutcomeType'])
dogs_outcome_dummies = pd.get_dummies(dogs_ds['OutcomeType'])

# print('Cat dummies: ', cats_outcome_dummies.describe())

cats_outcome_dummies.columns = Y_FIELDS
dogs_outcome_dummies.columns = Y_FIELDS

cats_ds = cats_ds.join(cats_outcome_dummies)
dogs_ds = dogs_ds.join(dogs_outcome_dummies)

cats_outcome_dummies = pd.get_dummies(cats_ds['OutcomeType'])
dogs_outcome_dummies = pd.get_dummies(dogs_ds['OutcomeType'])




