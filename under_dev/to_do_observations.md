Guidelines to proofread the observations subpackage in ***oceanicospy***

# Standard logic per class

Each model of each sensor type has a particular class whether be derived from an abstract or normally defined. The naming convention usually follows the name od the sensor, not the brand.

Each class usually has two main methods: .*get_raw_records()* and .*get_clean_records()*:

* What we usually agree about *.get_raw_records()* should retrieve the minimal readable pandas DataFrame from the records file with no changes/tweaks.
* Regarding the .*get_clean_records()*, we interpret this method to get a ready-to-use pandas DataFrame. Some features we identify as standard are:
  * Having a pd.Timeindex as the DataFrame index as much as possible.
  * Trim the time record based on the start_date and end_date provided by user.
  * No matter if there NaNs in the cleaned DataFrame, the identification and deletion of NaNs is left to the user for now.
  * We've stablished a naming formatting for the columns but further efforts should be done on this:
    * each column should be like: variable_name[units]

## Documentation

The documentation is expected to follow the numpy-style. This means that parameters, return and notes are the minimal accepted. Also watch the if you want to highlight some variables in the documentation website the *rst syntaxis should be followed

***If anything does not follow this logic or can be improved, do not hesitate to modify it.***

# Things to check/implement

First of all, behaviour of the classes ***should be tested as much as possible*** so each effort on check the performance of the classes is more than welcome.

Regarding the particular classes:

## Weather stations

* There are lots of variables in the Davis station that are not easily readable in terms of convention. We need to paraphrase the column names to make the DataFrame more standard. For example, what is Temp1, Temp2, Dir2 or Dir1? are they the same?
* The naming convention needs to be treated carefully to not mix things up. Let's say we have a rain column in the Davis weather station so we rename it as rain[mm], if we have a rain or precipitation in another station, ***we need to be sure it points to the same physical thing (units and/or method), otherwise the rain column from one will not be comparable to another.***

## Spotter

* There are some column names from both sources which do not follow a naming convention yet, for example: how are we going to call: significant_wave_height_sofar_model?. Naming convention for much used variables has still to be defined to keep consistency.
* Dataframe from SOFAR brings the wind speed and direction. Should it be cool if those variables have the same names than the ones used in the weather stations to keep consistency?

## AWAC

- It is quite weird that the single wad file in the *observsations/AWAC* folder does not start at 00:00 (minutes/seconds)
- There is an option to compute the currents direction that uses another function in *utils*. This needs to be validated with data that has been already processed.

# Final disclaimer

**Everything that has been code is not a straitjacket, feel free to purpose something that can lead to a stronger version. Any improvement is encouraged on the source code and the guide example on examples/reading_field_data.ipynb.** 

Add claim your credits in the notes section for the methods you refactor or create. Making yourself visible is key to tracking efforts.
