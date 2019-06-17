# DDNN (Double Dummy Neural Network)

## Usage
We use https://github.com/dds-bridge/dds and https://github.com/Afwas/python-dds.  

Make directory dds/dds-bridge and clone https://github.com/dds-bridge/dds.  
dds/dds.py, dds/functions.py and dds/dds_hands.py are modified from https://github.com/dds-bridge/dds.

## Dataset
### Create dataset
Create the .csv file which includes sets of hands data and dds data.
```
$ python dataset_maker.py
```

### Format
```
N hand (52), E hand (52), S hand (52), W hand (52), N DDS (5), E DDS (5), S DDS (5), W DDS (5)  
```

Every bid is separated by blank.


### Hand format
52 dimensions binary data  
```
club 2 ... A, diamond 2 ... A, heart 2 ... A, spade 2 ... A
```

If the player has the card, the bit is 1.

### DDS format
5 dimensions 0-13 integer data  
club, diamond, heart, spade, no trump  
If the player is the declarer and the suit is trump, it is the number of tricks the declarer's team will take 

## Train DDNN
```
$ python netwrok.py
```