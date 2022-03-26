
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import seaborn as sns
import re

#the delimeter is ; not , so be carefull
LGA = pd.read_csv("LGAs_new.csv", sep = ";")
#different LGA tags
LGA.to_csv("LGA_names.csv")


# import the csv for response time as pd df and assign the data according to the response code 
response_time= pd.read_csv("LGA-Response-Time-Performance-FY-2019.csv")

rt_code1 = response_time.head(79)
rt_code2 = response_time[88:167]


# merge the tags table to response time tables for future mergers
rt_code1 = rt_code1.merge(LGA, left_on = "LGA Name", right_on ="LGA", how ="left") 
rt_code2 = rt_code2.merge(LGA, left_on = "LGA Name", right_on ="LGA", how ="left")


#adjust the datatypes 
rt_code1["AVG RT - Seconds"] = rt_code1["AVG RT - Seconds"].astype(int)
rt_code1.to_csv("rt_code1.csv")


#adjust the datatypes
rt_code2["AVG RT - Seconds"] = rt_code2["AVG RT - Seconds"].astype(int)
rt_code2.to_csv("rt_code2.csv")


# get desriptive statistics and overall info
rt_code1["AVG RT - Seconds"].describe().to_csv("AVG_RT_Statistics_Code1.csv")
rt_code2["AVG RT - Seconds"].describe().to_csv("AVG_RT_Statistics_Code2.csv")



# top 10 LGA with the worst Code 1 average response times
worst1= rt_code1.sort_values("AVG RT - Seconds", ascending=False).head(10)
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(x="LGA",y ="AVG RT - Seconds", data = worst1, palette ='plasma')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Worst Average Response Time for Code 1')
plt.savefig('worst_Code1_Average-RT.png', bbox_inches='tight')
plt.close()

# top 10 LGA with the best Code 1 average response times
best1= rt_code1.sort_values("AVG RT - Seconds", ascending=True).head(10)
fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.barplot(x="LGA",y ="AVG RT - Seconds", data = best1, palette ='plasma')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Best Average Response Time for Code 1')
plt.savefig('best_Code1_Average-RT.png', bbox_inches='tight')
plt.close()


# top 10 LGA with the worst Code 2 average response times
worst2 = rt_code2.sort_values("AVG RT - Seconds", ascending=False).head(10)
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.barplot(x="LGA",y ="AVG RT - Seconds", data = worst2, palette ='plasma')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Worst Average Response Time for Code 2')
plt.savefig('worst_Code2_Average-RT.png', bbox_inches='tight')
plt.close()

# top 10 LGA with the best Code 2 average response times
best2 = rt_code2.sort_values("AVG RT - Seconds", ascending=True).head(10)
fig2, ax2 = plt.subplots(figsize=(12, 6))
sns.barplot(x="LGA",y ="AVG RT - Seconds", data = best2, palette ='plasma')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Best Average Response Time for Code 2')
plt.savefig('best_Code2_Average-RT.png', bbox_inches='tight')
plt.close()



# analyse population growth statistics
# get the expected population size in thousands for each LGA over the years
population_projection = pd.read_csv("Population_Growth.csv", sep =";")
del population_projection["Unnamed: 2"] #remove null columns
del population_projection["LGA"]
pop_pro = population_projection.merge(LGA, on = "LGA Code", how ="inner")

short_pop_pro = pop_pro[["LGA","2021","2026","2031","2036"]]


t_pop_pro = short_pop_pro[["2021","2026","2031","2036"]].T
t_pop_pro = t_pop_pro.set_axis(short_pop_pro["LGA"], axis=1)
pop_pct_change = t_pop_pro.pct_change()
pop_pct_change = pop_pct_change.set_axis(short_pop_pro["LGA"], axis=1)
pop_pct_change*=100 
pop_pct_change



# calculate percentage population change from 2021 to 2026, 2026 to 2031 and 2031 to 2036
n_pop_pct_change = pop_pct_change.T
n_pop_pct_change = n_pop_pct_change[["2026","2031","2036"]]
n_pop_pct_change = n_pop_pct_change.set_axis(["% increase from 2021 to 2026","% increase from 2026 to 2031","% increase from 2031 to 2036"], axis=1)



total_pct_change = n_pop_pct_change
total_pct_change['total % increase from 2021 to 2036'] = n_pop_pct_change.sum(axis=1, numeric_only= True)
total_pct_change.reset_index()
total_pct_change.to_csv("total_pct_change.csv")

# get descriptive statistics for percentage population change over the years 
total_pct_change.describe().to_csv("total_pct_change_statistics.csv")



highest_total_pct_change = total_pct_change.sort_values(by='total % increase from 2021 to 2036', ascending=False).head(10)
highest_total_pct_change['LGA']= highest_total_pct_change.index



# plot LGAs with the greatest total % increase from 2021 to 2036
LGAdata1 = highest_total_pct_change['LGA']
increase_to_2026 = highest_total_pct_change['% increase from 2021 to 2026'].to_numpy()
increase_to_2031 = highest_total_pct_change['% increase from 2026 to 2031'].to_numpy()
increase_to_2036 = highest_total_pct_change['% increase from 2031 to 2036'].to_numpy()
ind = [x for x, _ in enumerate(LGAdata1)]

plt.bar(ind, increase_to_2036, width=0.8, label='% increase from 2031 to 2036', color='gold', bottom=increase_to_2031+increase_to_2026)
plt.bar(ind, increase_to_2031, width=0.8, label='% increase from 2026 to 2031', color='silver', bottom=increase_to_2026)
plt.bar(ind, increase_to_2026, width=0.8, label='% increase from 2021 to 2026', color='#CD853F')
plt.xticks(ind, LGAdata1, rotation=45, ha='right')
plt.ylabel("% population change")
plt.xlabel("LGA")
plt.legend(loc="upper right")
plt.title("LGAs with Greatest % Population Increase")
plt.savefig("LGAs_with_Greatest_Percentage_Population_Increase.png", bbox_inches='tight')
plt.close()


lowest_total_pct_change = total_pct_change.sort_values(by='total % increase from 2021 to 2036', ascending=True).head(10)
lowest_total_pct_change['LGA']= lowest_total_pct_change.index




# plot LGAs with the smallest total % increase from 2021 to 2036
LGAdata2 = lowest_total_pct_change['LGA']
increase_to_2026 = lowest_total_pct_change['% increase from 2021 to 2026'].to_numpy()
increase_to_2031 = lowest_total_pct_change['% increase from 2026 to 2031'].to_numpy()
increase_to_2036 = lowest_total_pct_change['% increase from 2031 to 2036'].to_numpy()
ind = [x for x, _ in enumerate(LGAdata2)]

plt.bar(ind, increase_to_2036, width=0.8, label='% increase from 2031 to 2036', color='gold', bottom=increase_to_2031+increase_to_2026)
plt.bar(ind, increase_to_2031, width=0.8, label='% increase from 2026 to 2031', color='silver', bottom=increase_to_2026)
plt.bar(ind, increase_to_2026, width=0.8, label='% increase from 2021 to 2026', color='#CD853F')

plt.xticks(ind, LGAdata2, rotation=45, ha='right')
plt.ylabel("% population change")
plt.xlabel("LGA")
plt.legend(loc="lower right")
plt.title("LGAs with Smallest % Population Increase")
plt.savefig("LGAs_with_Smallest_Percentage_Population_Increase.png", bbox_inches='tight')
plt.close()



# find the difference in population number from 2021 to 2026, 2026 to 2031 and 2031 to 2036 for each LGA
pop_diff = t_pop_pro.diff()
pop_diff*=1000 #convert from thousands to ones
n_pop_diff = pop_diff.T
del n_pop_diff["2021"]
n_pop_diff = n_pop_diff.set_axis(["pop num diff from 2021 to 2026","pop num diff from 2026 to 2031","pop num diff from 2031 to 2036"], axis=1)


# get descriptive statistics for difference in population size over the years
n_pop_diff.describe()



total_pop_diff = n_pop_diff
total_pop_diff['total pop number difference from 2021 to 2036'] = n_pop_diff.sum(axis=1, numeric_only= True)
total_pop_diff.reset_index()
total_pop_diff.to_csv("total_pop_diff.csv")

# get descriptive statistics for population number change over the years
total_pop_diff.describe().to_csv("total_pop_diff_statistics.csv")


highest_total_pop_diff = total_pop_diff.sort_values(by='total pop number difference from 2021 to 2036', ascending=False).head(10)
highest_total_pop_diff['LGA']= highest_total_pop_diff.index


# plot LGAs with the greatest total population number difference from 2021 to 2036
LGAdata3 = highest_total_pop_diff['LGA']
increase_to_2026 = highest_total_pop_diff['pop num diff from 2021 to 2026'].to_numpy()
increase_to_2031 = highest_total_pop_diff['pop num diff from 2026 to 2031'].to_numpy()
increase_to_2036 = highest_total_pop_diff['pop num diff from 2031 to 2036'].to_numpy()
ind = [x for x, _ in enumerate(LGAdata3)]

plt.bar(ind, increase_to_2036, width=0.8, label='pop num increase from 2031 to 2036', color='gold', bottom=increase_to_2031+increase_to_2026)
plt.bar(ind, increase_to_2031, width=0.8, label='pop num increase from 2026 to 2031', color='silver', bottom=increase_to_2026)
plt.bar(ind, increase_to_2026, width=0.8, label='pop num increase from 2021 to 2026', color='#CD853F')

plt.xticks(ind, LGAdata3, rotation=45, ha='right')
plt.ylabel("population number difference")
plt.xlabel("LGA")
plt.legend(loc="upper right")
plt.title("LGAs with Greatest Increase in Population Number")
plt.savefig("LGAs_with_Greatest_Increase_in_Population_Number.png", bbox_inches='tight')
plt.close()


lowest_total_pop_diff = total_pop_diff.sort_values(by='total pop number difference from 2021 to 2036', ascending=True).head(10)
lowest_total_pop_diff['LGA']= lowest_total_pop_diff.index



# plot LGAs with the lowest total population number difference from 2021 to 2036
LGAdata4 = lowest_total_pop_diff['LGA']
increase_to_2026 = lowest_total_pop_diff['pop num diff from 2021 to 2026'].to_numpy()
increase_to_2031 = lowest_total_pop_diff['pop num diff from 2026 to 2031'].to_numpy()
increase_to_2036 = lowest_total_pop_diff['pop num diff from 2031 to 2036'].to_numpy()
ind = [x for x, _ in enumerate(LGAdata4)]

plt.bar(ind, increase_to_2036, width=0.8, label='pop num increase from 2031 to 2036', color='gold', bottom=increase_to_2031+increase_to_2026)
plt.bar(ind, increase_to_2031, width=0.8, label='pop num increase from 2026 to 2031', color='silver', bottom=increase_to_2026)
plt.bar(ind, increase_to_2026, width=0.8, label='pop num increase from 2021 to 2026', color='#CD853F')

plt.xticks(ind, LGAdata4, rotation=45, ha='right')
plt.ylabel("population number difference")
plt.xlabel("LGA")
plt.legend(loc="lower right")
plt.title("LGAs with Smallest Increase in Population Number")
plt.savefig("LGAs_with_Smallest_Increase_in_Population_Number.png", bbox_inches='tight')
plt.close()



# obtain data for number of medical practitioners in each LGA
health_force = pd.read_csv("phidu_health_workforce_lga_2018-1057657243358440567.csv")
health_force = LGA.merge(health_force, left_on = "LGA Code", right_on =" lga_code",how="left")
del health_force["LGA_b"]
del health_force[" lga_name"]
del health_force[" lga_code"]
health_force.set_index("LGA Code")
# West Wimmera data row is empy so we drop it
health_force_full = health_force.dropna()




# Create new People/Doctor Dataset
LGAlst = health_force["LGA"]
LGApop = population_projection["2021"]
LGAdrs = health_force.iloc[: , -1] # total medical practitioners 2018 number

DRSperPOP = {}
for i in range(len(LGApop)):
    DRSperPOP[LGAlst[i]] = (LGApop[i] * 1000)/LGAdrs[i]
    
data = {"LGA" : DRSperPOP.keys(), "People per Doctor" : DRSperPOP.values()}
doctor_to_pop = pd.DataFrame.from_dict(data)
doctor_to_pop.to_csv("doctor_to_pop.csv")
doctor_to_pop.describe().to_csv("doctor_to_pop_statistics.csv")



# create a lolliplot chart of the 15 LGAs with the worst people:doctor ratio
ordered_doctor_pop = doctor_to_pop.sort_values(by="People per Doctor", ascending=False).head(10)
my_range=range(0,len(ordered_doctor_pop.index))
plt.stem(ordered_doctor_pop["People per Doctor"])
plt.xticks(my_range, ordered_doctor_pop["LGA"], rotation=45, horizontalalignment='right')
plt.xlabel('LGA')
plt.ylabel('People per Doctor')
plt.title('Worst LGAs: Highest Number of People per Doctor')
plt.savefig('Worst LGAs: Highest Number of People per Doctor', bbox_inches='tight')
plt.close()


# create a lolliplot chart of the 15 LGAs with the worst people:doctor ratio
ordered_doctor_pop2 = doctor_to_pop.sort_values(by="People per Doctor", ascending=True).head(10)
ordered_doctor_pop2
my_range2=range(0,len(ordered_doctor_pop2.index))
plt.stem(ordered_doctor_pop2["People per Doctor"])
plt.xticks(my_range2, ordered_doctor_pop2["LGA"], rotation=45, horizontalalignment='right')
plt.xlabel('LGA')
plt.ylabel('People per Doctor')
plt.title('Best LGAs: Lowest Number of People per Doctor')
plt.savefig('Best LGAs: Lowest Number of People per Doctor', bbox_inches='tight')
plt.close()



# Multivarible analysis


# Function to add ranks to dataframes
def createrankset(dataset, rankval,boolpref, col_list):
    rankset = dataset.sort_values(rankval, ascending=boolpref)
    rankset['Rank'] = np.arange(len(rankset)) + 1
    rankset = rankset[col_list]
    return rankset


# Create list to iterate through during ranking scheme
listof_dataframes = []


rt1_pop = n_pop_diff.merge(rt_code1, on ="LGA", how = "inner")
rt2_pop = n_pop_diff.merge(rt_code2, on ="LGA", how = "inner")


n_pop_diff = n_pop_diff.reset_index()



# Create ranked dataframes to be used in ranking scheme
sorted_rt1_pop = createrankset(rt1_pop,"AVG RT - Seconds",False,['Rank','LGA','AVG RT - Seconds'])
listof_dataframes.append(sorted_rt1_pop)
sorted_rt2_pop = createrankset(rt2_pop,"AVG RT - Seconds",False,['Rank','LGA','AVG RT - Seconds'])
listof_dataframes.append(sorted_rt2_pop)
sorted_cap_inc_2031 = createrankset(n_pop_diff, "pop num diff from 2026 to 2031",False,["Rank", "LGA", "pop num diff from 2026 to 2031"])
listof_dataframes.append(sorted_cap_inc_2031)
sorted_cap_inc_2026 = createrankset(n_pop_diff, "pop num diff from 2021 to 2026",False,["Rank", "LGA", "pop num diff from 2021 to 2026"])
listof_dataframes.append(sorted_cap_inc_2026)
sorted_cap_inc_2036 = createrankset(n_pop_diff, "pop num diff from 2031 to 2036",False,["Rank", "LGA", "pop num diff from 2031 to 2036"])
listof_dataframes.append(sorted_cap_inc_2036)
sorted_health_workforce = createrankset(doctor_to_pop,"People per Doctor",True,["Rank", "LGA","People per Doctor"])
listof_dataframes.append(sorted_health_workforce)



# Use ranking scheme to determine areas that would most benefit from a new hospital
rankweight_dict = {}
#weight_lst = [RT1, RT2, 2031CP, 2026CP, 2036CP, PPLperDR]
weight_lst = [0.30, 0.10, 0.225, 0.135, 0.09, 0.15]
for i in range(len(listof_dataframes)):
    df = listof_dataframes[i]
    weight = weight_lst[i]
    for ind in df.index:
        if i in [2,3,4]:
            if i == 2:
                p = "pop num diff from 2026 to 2031"
            elif i == 3:
                p = "pop num diff from 2021 to 2026"
            elif i == 4:
                p = "pop num diff from 2031 to 2036"
            rankweight_dict[df['LGA'][ind]] += df[p][ind] * weight
        elif i == 1:
            rankweight_dict[df['LGA'][ind]] += df['AVG RT - Seconds'][ind] * weight
        elif i == 5:
            rankweight_dict[df['LGA'][ind]] += df['People per Doctor'][ind] * weight
        else:
            rankweight_dict[df['LGA'][ind]] = df['AVG RT - Seconds'][ind] * weight
            


# Create a ranking library and dataframe
data = {"LGA" :rankweight_dict.keys(),
        "Weight" : rankweight_dict.values()}
weighted_rank_df = pd.DataFrame.from_dict(data)
worst_list = weighted_rank_df.sort_values("Weight", ascending=False)
worst_5 = worst_list.head()
worst_5.index = np.arange(len(worst_5)) + 1
worst_5.to_csv("worst5_LGAs.csv")



spider_data = listof_dataframes[0].merge(listof_dataframes[1],how = 'outer', on='LGA').merge(listof_dataframes[3],how = 'outer', on='LGA').merge(listof_dataframes[2],how = 'outer', on='LGA').merge(listof_dataframes[4],how = 'outer', on='LGA').merge(listof_dataframes[5],how = 'outer', on='LGA')




# Change index to LGAs and drop irrelevant columns
spider_data.index = listof_dataframes[0]['LGA']
spider_data = spider_data.drop(columns = ["Rank_x", "Rank_y","LGA"])
spider_data.to_csv('spider.csv')



# Normalise response time data to improve visualisation
spider_data["AVG RT - Seconds_x"] = spider_data["AVG RT - Seconds_x"] * 50
spider_data["AVG RT - Seconds_y"] = spider_data["AVG RT - Seconds_y"] * 42
spider_data["People per Doctor"] = spider_data["People per Doctor"] * 40
spider_data["Needed"] = spider_data["AVG RT - Seconds_x"]


spider_data_T = spider_data.T
spider_data_T


spiderplot_dict = {}
lga_lst = []
for lga in worst_5["LGA"]:
    spiderplot_dict[lga] = spider_data_T[lga]
    lga_lst.append(lga)
label_loc = np.linspace(start=0, stop=2 * np.pi, num=len(spiderplot_dict[lga]))


categories = ["T1 Response","T2 Response","2026 Pop Num Increase", "2031 Pop Num Increase", "2036 Pop Num Increase","People in LGA per Doctor"]



# create spider plots for the 5 worst LGAs
colors = ['b', 'r', 'g', 'm', 'y']
i = 0;
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

for lga in lga_lst:
    title = lga + " Radar"
    figname = lga + "_Radar.png"
    plt.figure(figsize=(6, 6))
    plt.subplot(polar=True)
    plt.plot(label_loc, spiderplot_dict[lga], label=lga, color = colors[i])
    plt.fill(label_loc, spiderplot_dict[lga], label=lga, facecolor = colors[i], alpha=0.25)
    plt.title(title, size=20, y=1.05)
    plt.xticks(angles[:-1],categories)
    plt.yticks([])
    plt.savefig(figname)
    i += 1




