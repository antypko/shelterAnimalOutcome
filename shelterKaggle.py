import pandas as pd

input_train_ds = pd.read_csv('./train.csv', header=0)
input_test_ds = pd.read_csv('./test.csv', header=0)

age_cat_values = ['less5Days', '5to28Days', '28to35Days', '35to152Days', '152to213Days', '213to273Days',
                  '273to334Days', '334to496Days', '496to1460Days', '1460to2191Days', '2191to3287Days',
                  '3287to4017Days', '4017to5113Days', 'more5113Days']
age_dog_values = ['less4Days', '4to21Days', '35to91Days', '91to121Days', '121to213Days', '213to273Days', '273to334Days',
                  '334to1007Days', '1007to1460Days', '1460to2556Days', '2556to3287Days',
                  '3287to4017Days', '4017to4748Days', '4748to5843Days', 'more5843Days']

cat_colors = ['Agouti', 'Apricot', 'Black', 'Blue', 'Brown', 'Buff', 'Calico', 'Chocolate', 'Cream', 'Flame', 'Gray',
              'Lilac', 'Lynx', 'Orange', 'Pink', 'Point', 'Seal', 'Silver', 'Smoke', 'Tabby', 'Tan', 'Tiger', 'Torbie',
              'Tortie', 'Tricolor', 'White', 'Yellow']

dog_colors = ['Apricot', 'Black', 'Blue', 'Brindle', 'Brown', 'Buff', 'Chocolate', 'Cream', 'Fawn', 'Gold', 'Gray',
              'Liver', 'Merle', 'Orange', 'Pink', 'Red', 'Ruddy', 'Sable', 'Silver', 'Smoke', 'Tabby', 'Tan', 'Tick',
              'Tiger', 'Tricolor', 'White', 'Yellow']

cat_breeds = ['Abyssinian', 'American', 'Angora', 'Balinese', 'Bengal', 'Blue', 'Bobtail', 'Bombay', 'British', 'Brown',
              'Burmese', 'Cat', 'Coon', 'Cornish', 'Cymric', 'Devon', 'Domestic', 'Exotic', 'Forest', 'Hair', 'Havana',
              'Himalayan', 'Japanese', 'Javanese', 'Longhair', 'Maine', 'Manx', 'Medium', 'Mix', 'Munchkin',
              'Norwegian', 'Ocicat', 'Persian', 'Pixiebob', 'Ragdoll', 'Rex', 'Russian', 'Shorthair', 'Siamese',
              'Snowshoe', 'Sphynx', 'Tonkinese', 'Turkish', 'Van']

dog_breeds = ['Affenpinscher', 'Afghan', 'Airedale', 'Akita', 'Alaskan', 'American', 'Anatol', 'Apso', 'Argentino',
              'Australian', 'Basenji', 'Basset', 'Bay', 'Beagle', 'Bearded', 'Beauceron', 'Bedlington', 'Belgian',
              'Bernard', 'Bernese', 'Bichon', 'Black', 'Bloodhound', 'Blue', 'Bluetick', 'Boerboel', 'Bordeaux',
              'Border', 'Borzoi', 'Boston', 'Boxer', 'Boykin', 'Brindle', 'Brittany', 'Bruss', 'Bull', 'Bulldog',
              'Bullmastiff', 'Cairn', 'Canaan', 'Canario', 'Cane', 'Cardigan', 'Carolina', 'Catahoula', 'Cattle',
              'Cavalier', 'Chesa', 'Chihuahua', 'Chin', 'Chinese', 'Chow', 'Coat', 'Coated', 'Cocker', 'Collie',
              'Coonhound', 'Corgi', 'Corso', 'Crested', 'Cur', 'Dachshund', 'Dalmatian', 'Dane', 'De', 'Doberman',
              'Dog', 'Dogo', 'Dogue', 'Duck', 'Dutch', 'Elkhound', 'English', 'Entlebucher', 'Eskimo', 'Feist', 'Field',
              'Finnish', 'Flat', 'Fox', 'Foxhound', 'French', 'Frise', 'German', 'Giant', 'Glen', 'Golden', 'Great',
              'Greater', 'Greyhound', 'Griffon', 'Hair', 'Hairless', 'Harrier', 'Havanese', 'Heeler', 'Highland',
              'Hound', 'Hovawart', 'Husky', 'Ibizan', 'Imaal', 'Inu', 'Irish', 'Italian', 'Italiano', 'Jack',
              'Japanese', 'Jindo', 'Keeshond', 'Kelpie', 'Kuvasz', 'Labrador', 'Lacy', 'Landseer', 'Leonberger',
              'Lhasa', 'Longhair', 'Lowchen', 'Malamute', 'Malinois', 'Maltese', 'Manchester', 'Mastiff', 'Mexican',
              'Miniature', 'Mix', 'Mountain', 'Mouth', 'Neapolitan', 'Newfoundland', 'Norfolk', 'Norwegian', 'Norwich',
              'Nova', 'Of', 'Old', 'Otterhound', 'Papillon', 'Parson', 'Patterdale', 'Pbgv', 'Pekingese', 'Pembroke',
              'Pequeno', 'Pharaoh', 'Picardy', 'Pinsch', 'Pinscher', 'Pit', 'Plott', 'Podengo', 'Pointer', 'Pointing',
              'Pomeranian', 'Poodle', 'Port', 'Presa', 'Pug', 'Pyrenees', 'Queensland', 'Rat', 'Redbone', 'Retr',
              'Retriever', 'Rhod', 'Ridgeback', 'Rottweiler', 'Rough', 'Russell', 'Saluki', 'Samoyed', 'Schipperke',
              'Schnauzer', 'Scotia', 'Scottish', 'Sealyham', 'Setter', 'Sharpei', 'Sheepdog', 'Shepherd', 'Shetland',
              'Shiba', 'Shih', 'Shorthair', 'Siberian', 'Silky', 'Skye', 'Smooth', 'Soft', 'Span', 'Spaniel', 'Spanish',
              'Spinone', 'Spitz', 'Springer', 'St.', 'Staffordshire', 'Standard', 'Swedish', 'Swiss', 'Tan', 'Tennesse',
              'Terr', 'Terrier', 'Tervuren', 'Tibetan', 'Tolling', 'Toy', 'Treeing', 'Tzu', 'Unknown', 'Vallhund',
              'Vizsla', 'Walker', 'Water', 'Weimaraner', 'Welsh', 'West', 'Wheaten', 'Whippet', 'Wire', 'Wirehair',
              'Wirehaired', 'Wolfhound', 'Yorkshire']

sex_upon_outcome = ["Intact Female", "Intact Male", "Neutered Male", "Spayed Female", "Unknown"]

Y_LABELS = ["Adoption", "Died", "Euthanasia", "Return_to_owner", "Transfer"]


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


def prepare_shelter_data(input_data, train_data=True):
    dogs_ds = pd.DataFrame(input_data[input_data['AnimalType'] == 'Dog'])
    cats_ds = pd.DataFrame(input_data[input_data['AnimalType'] == 'Cat'])

    # print(dogs_ds.columns())

    if train_data:
        cats_outcome_dummies = pd.get_dummies(cats_ds['OutcomeType'])
        dogs_outcome_dummies = pd.get_dummies(dogs_ds['OutcomeType'])

        cats_outcome_dummies.columns = Y_LABELS
        dogs_outcome_dummies.columns = Y_LABELS

        cats_ds = cats_ds.join(cats_outcome_dummies)
        dogs_ds = dogs_ds.join(dogs_outcome_dummies)
    else:
        for label in Y_LABELS:
            cats_ds[label] = 0
            dogs_ds[label] = 0

    cats_ds.loc[:, 'AgeuponOutcome'] = cats_ds.loc[:, 'AgeuponOutcome'].apply(transform_to_days)
    dogs_ds.loc[:, 'AgeuponOutcome'] = dogs_ds.loc[:, 'AgeuponOutcome'].apply(transform_to_days)
    cats_not_null_ages = cats_ds['AgeuponOutcome'].notnull()
    dogs_not_null_ages = dogs_ds['AgeuponOutcome'].notnull()
    cats_ds = cats_ds.replace({'AgeuponOutcome': 0},
                              int(round(cats_ds[cats_not_null_ages]['AgeuponOutcome'].mean(), 0)))
    dogs_ds = dogs_ds.replace({'AgeuponOutcome': 0},
                              int(round(dogs_ds[dogs_not_null_ages]['AgeuponOutcome'].mean(), 0)))

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
    cats_ds.loc[(cats_ds['AgeuponOutcome'] >= 496) & (cats_ds['AgeuponOutcome'] < 1460), '496to1460Days'] = 1
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

    cats_sex_dummies = pd.get_dummies(cats_ds['SexuponOutcome'])
    dogs_sex_dummies = pd.get_dummies(dogs_ds['SexuponOutcome'])

    cats_sex_dummies.columns = sex_upon_outcome
    dogs_sex_dummies.columns = sex_upon_outcome

    cats_ds = cats_ds.join(cats_sex_dummies)
    dogs_ds = dogs_ds.join(dogs_sex_dummies)

    for cat_color_single_value in cat_colors:
        cats_ds[cat_color_single_value] = 0
        cats_ds.loc[cats_ds['Color'].str.contains(cat_color_single_value, na=False), cat_color_single_value] = 1

    for dog_color_single_value in dog_colors:
        dogs_ds[dog_color_single_value] = 0
        dogs_ds.loc[dogs_ds['Color'].str.contains(dog_color_single_value, na=False), dog_color_single_value] = 1

    for cat_breed_single_value in cat_breeds:
        cats_ds[cat_breed_single_value] = 0
        cats_ds.loc[cats_ds['Breed'].str.contains(cat_breed_single_value, na=False), cat_breed_single_value] = 1

    for dog_breed_single_value in dog_breeds:
        dogs_ds[dog_breed_single_value] = 0
        dogs_ds.loc[dogs_ds['Breed'].str.contains(dog_breed_single_value, na=False), dog_breed_single_value] = 1

    return {'cats': cats_ds, 'dogs': dogs_ds}


train_result = prepare_shelter_data(input_train_ds)
#
train_cats_columns = train_result['cats'].columns.values
train_dogs_columns = train_result['dogs'].columns.values

print('Number of train cats columns', len(train_cats_columns))
print('Number of train dogs columns', len(train_dogs_columns))

test_result = prepare_shelter_data(input_test_ds, train_data=False)

test_cats_columns = test_result['cats'].columns.values
test_dogs_columns = test_result['dogs'].columns.values

print('Number of test cats columns', len(test_cats_columns))
print('Number of test dogs columns', len(test_dogs_columns))

print('Symmetric Difference:')
print(set(train_cats_columns).symmetric_difference(set(test_cats_columns)))

print('Train from Test Difference:')
print(set(train_cats_columns) - set(test_cats_columns))

print('Test from Train Difference:')
print(set(test_cats_columns) - set(train_cats_columns))

# print(result['cats'].columns.values)
# print(result['cats'].columns.values)
