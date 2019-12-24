#### Analyzing Chicago's Rideshare Trips
as part of my STAT525 Course Project at Purdue
*** 
The dataset, provided by the City of Chicago, can be downloaded from
[here](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips/m6dm-c72p).
It consists of all the rideshare (presumably Uber/Lyft) trips taken in
Chicago since November 2018, totalling to a whopping 101 million trips.
I haven't yet singled in on what exactly I want to study.

The description of the data, as given on the CoC website,

<i> <p> &ensp;&ensp;&ensp;&ensp; "All trips, starting November 2018,
reported by Transportation Network Providers (sometimes called rideshare
companies) to the City of Chicago as part of routine reporting required
by ordinance." </i>

A visual overview of the dataset can be found
[here](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips-Dashboard/pvbr-dkbf).


Important notes regarding the data,

- **Community Area** and **Census Tract** values are *null* for
  locations outside Chicago; Same for the lat/lon coordinates.
- **Tips** are rounded off to the nearest $1.00; **Fares** to the
  nearest $2.50.

Some notable characteristics of the dataset that were put into place for
privacy considerations,

- Trip start/stop times are rounded to the nearest 15-minute interval.
- Rather than exact coordinates, the centroids of the **census tracts**
  of pickup/dropoff locations are provided. Each census tract is at
  least 89,000 sq feet.
-  Further, for approx. one-third of the data, only the **community
   areas** of pickup/dropoff are provided, and each spans about 3 sq
   miles.
  