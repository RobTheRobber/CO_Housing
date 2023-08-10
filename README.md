# Satellite Life Expectancies

## Overview
There are many different vital services services that satellites provide. 

## Question and Hypothesis
- Has the expected lifetime of satellites increased over the years?
- What are some of the factors that are affecting the expected lifetime of satellites?

## Cleaning
When originally importing this dataset, there were many columns that were missing too much data or entirely empty. With these columns removed I lookec to the columns I decided to look towards cleaning up the columns I decided to keep. With the missing values for my Expected Lifetime column I noticed that all the values were rounded to whole numbers and there were very few outliers. With that information I decided to use the median to fill in the missing data in order to keep my values consistent. Seeing how that this was the only numerical column I used, I looked at my categorical columns to fill in the missing data with 'Unknowns'. This allowed me to map the columns in order to make comparisions using my model. Seeing how I was going to be separating the satellites by decade, I used the 'Launch Date' column to create my 'Decade' column to group my data. After adding the mapping columns for each of my categorical columns I was ready to start my analysis.

## Visualization
 Starting off with a surface level analysis of the 'Life Expectancy' we can see the vast difference between the  2000's mean and the 2020's mean with the average dropping from 10.8 years to 4.8 years. We can also see the huge increase in satellite launches with 2020-2022 dwarfing the amount of launches conducted from 2000-2019. Seeing how many launches there were I decided to breakdown those launches to see where they were being sent.
<br>
<img src="img/decades.png", width=40% height=30%/>
<br>
Looking at the data we can see that the majority of the satellites launched for the 2020's were into the LEO orbit. Looking at how heaviliy the data was skewed with LEO satellites I decided to look at the life expectancy of a satellite in each orbit.
<br>
<img src="img/orbits.png" width=40% height=30%/>
<br>
There is a very clear distinction between the average life expectancy of a satellite and the orbit. Comparing this graph with the previous one we can start to understand why the average expected life expectancy dropped so drastically for the 2020's.
<br>
<img src="img/mean_orbit.png" width=40% height=30%/>
<br>
Reviewing the change in life expectancy for each decade for each orbit there is a better picture. This shows how the life expectancy has been increasing for satellites in both for MEO and GEO while LEO has declined. This is very interesting especially when we look at the operator who has been launching the majority of the satellites.
<br>
<img src="img/mean_orbit_decade.png" width=40% height=30%/>
<br>

But as every realtor will tell you, the most important thing about a property is location, location, location. Now that we have identified the most common type of property along with the district we can start looking at the cost of these properties and break down where the value of these properties are coming from. First looking at the average cost of each type of property per district we can get a rough idea of the value of the location. By viewing these averages with our previous total value graph, we are able to get a more clear picture of the prices available of each property type per area.
<br>
<img src="img/operators.png" width=40% height=30%/>
<br>

## Conclusion
It is quite clear that the most common type of within this data are some type of home with single family homes being by far the most common. Looking at the districts we can see that the majority of properties are located in the Urban District and the General Services District having the vast majority of the properties within them. Looking at the costs of each property we could have a bit of skewed data due to the low the amount of data points in some of the property types but a pretty clear view that in most cases the value of a property comes from the building with some exceptions. Finally looking at the difference between the sale price and the total value of the home we can get an idea of how much these properties are being sold for and whether the property type increases or decreases with value. Finally we can view the general location of each property by tax district and see the average value of each property type within its district and see where the most and least expensive properties on average can be found.


##  Future Project Plans
Ways I can improve on this project in the future is by finding more data on the other property types to have a better breakdown of all property values and make sure I have better averages. I can also find a map of the tax districts in order to show where these locations are and maybe see why some of the price averages are higher than others. A big improvement that could be made is consolidating the code in order to improve reusability as even with the little code I was able to refactor into functions were made much more flexible and could be implemented in later projects.
